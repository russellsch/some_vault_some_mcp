"""LanceDB indexer — port of willfanguy's indexer.py with improvements.

Schema is preserved exactly (see docs/plans/lancedb-schema-capture.md):
- 12 columns, 768-dim float32 vectors (Ollama nomic-embed-text default)
- FTS index on content column
- tags/projects stored as comma-separated strings (not Arrow lists)

Improvements:
- SKIP_DIRS reduced to .obsidian, .git, .trash (vault-specific dirs removed)
- rglob("*.md") only — no SKIP_EXTENSIONS needed
- Dimension check at startup (§6.4 step 3)
- Single-file reindex actually limits scope (upstream bug fixed)
- embed_texts uses list batch input (Ollama supports it)
"""

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

TABLE_NAME = "vault_chunks"
EXCLUDED_DIRS = frozenset([".obsidian", ".git", ".trash"])
BATCH_SIZE = 50

# Bump when the *stored* record format changes (not query-time behaviour), so
# existing installs auto-reindex on the next boot. v2: tags/projects stored
# lowercased + sentinel-wrapped (",a,b,") for whole-token matching.
SCHEMA_VERSION = 2

_reindex_lock = threading.Lock()


def _schema_version_file(db_path: str) -> str:
    return os.path.join(db_path, "_schema_version.json")


def _write_schema_version(db_path: str) -> None:
    try:
        with open(_schema_version_file(db_path), "w", encoding="utf-8") as f:
            json.dump({"version": SCHEMA_VERSION}, f)
    except OSError as e:
        # Best-effort — a failed write must not crash indexing. Worst case the
        # next boot re-checks the data format (see _data_matches_current_format).
        logger.warning(f"Could not write schema version marker: {e}")


def _read_schema_version(db_path: str) -> int | None:
    try:
        with open(_schema_version_file(db_path), encoding="utf-8") as f:
            return int(json.load(f).get("version"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None  # missing/corrupt → treat as unknown


def _data_matches_current_format(table) -> bool:
    """True if the stored tags/projects already use the v2 sentinel format.

    Used to avoid a needless reindex (and a boot loop) when the version marker
    is merely missing/corrupt but the data was in fact already migrated.
    """
    try:
        df = table.search().select(["tags", "projects"]).to_pandas()
    except Exception:
        return False
    for col in ("tags", "projects"):
        for v in df[col]:
            if isinstance(v, str) and v and not v.startswith(","):
                return False  # old plain "a,b" format found
    return True


def check_and_maybe_migrate(db, db_path: str) -> bool:
    """Return True if the existing table must be fully reindexed for schema reasons.

    Auto-heals a lost/corrupt marker when the data is already current, so a
    successful reindex (or a healed marker) breaks any would-be boot loop.
    """
    table = _get_table(db)
    if table is None:
        return False  # no table — a fresh full_index writes the marker
    if _read_schema_version(db_path) == SCHEMA_VERSION:
        return False
    if _data_matches_current_format(table):
        _write_schema_version(db_path)  # marker lost but data fine — heal it
        return False
    return True


def _get_db(db_path: str):
    import lancedb
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    return lancedb.connect(db_path)


def _get_table(db, table_name: str = TABLE_NAME):
    if table_name in db.list_tables().tables:
        return db.open_table(table_name)
    return None


def _check_dimension_mismatch(db, provider_dims: int) -> None:
    """Check existing table vector dims against provider dims.

    Raises RuntimeError if they differ — prevents silent ArrowInvalid crash.
    If no table exists, silently passes (will create fresh).
    """
    table = _get_table(db)
    if table is None:
        return  # No table yet — skip check

    schema = table.schema
    vector_field = None
    for field in schema:
        if field.name == "vector":
            vector_field = field
            break

    if vector_field is None:
        return  # No vector column — unexpected but non-fatal

    # FixedSizeList exposes list_size; variable-size lists don't → None (skip check).
    # (The old elif branch could assign the boolean False here, producing a spurious
    # "existing index has False-dim vectors" message.)
    existing_dims = getattr(vector_field.type, "list_size", None)

    if existing_dims is not None and existing_dims != provider_dims:
        raise RuntimeError(
            f"Dimension mismatch: provider produces {provider_dims}-dim vectors "
            f"but existing index has {existing_dims}-dim vectors. "
            "Full reindex required — run with --reindex-force or delete the LanceDB directory."
        )


def _rebuild_fts_index(table) -> None:
    try:
        table.create_fts_index("content", replace=True)
        logger.info("FTS index rebuilt on 'content' column")
    except Exception as e:
        logger.warning(f"FTS index rebuild failed: {e}")


def scan_vault(vault_path: str) -> list[tuple[str, float]]:
    """Scan vault for .md files, returning [(vault_relative_path, mtime)].

    Skips excluded dirs and any file whose real path escapes the vault, so a
    symlink pointing outside the vault is never indexed (out-of-vault content
    must not enter the index / model context)."""
    from some_vault_some_mcp.core.paths import _within_vault
    vault = Path(vault_path)
    vault_root = vault.resolve()
    results = []
    for path in vault.rglob("*.md"):
        rel = str(path.relative_to(vault)).replace("\\", "/")
        parts = rel.split("/")
        if any(seg.lower() in EXCLUDED_DIRS for seg in parts):
            continue
        if not _within_vault(path, vault_root):
            continue
        results.append((rel, path.stat().st_mtime))
    return results


def _make_record(chunk: dict, vector: list[float]) -> dict:
    """Convert a chunk dict + vector to a LanceDB record dict."""
    from some_vault_some_mcp.core.filters import store_tokens
    return {
        "file_path": chunk["file_path"],
        "chunk_index": chunk["chunk_index"],
        "heading": chunk["heading"],
        "content": chunk["content"],
        "title": chunk.get("title", ""),
        # v2: lowercased + sentinel-wrapped for whole-token LIKE matching.
        "tags": store_tokens(chunk.get("tags", [])),
        "projects": store_tokens(chunk.get("projects", [])),
        "area": str(chunk.get("area") or ""),
        "status": str(chunk.get("status") or ""),
        "source": str(chunk.get("source") or ""),
        "file_mtime": chunk["file_mtime"],
        "vector": vector,
    }


def full_index(
    vault_path: str,
    db_path: str,
    provider=None,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Build complete index from scratch."""
    from some_vault_some_mcp.core.chunker import chunk_markdown
    if provider is None:
        from some_vault_some_mcp.core.embeddings import get_provider
        provider = get_provider()

    start = time.time()
    db = _get_db(db_path)

    if TABLE_NAME in db.list_tables().tables:
        db.drop_table(TABLE_NAME)

    files = scan_vault(vault_path)
    vault = Path(vault_path)
    logger.info(f"Scanning {len(files)} markdown files for full index...")

    all_chunks: list[dict] = []
    for rel_path, mtime in files:
        full_path = vault / rel_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Could not read {rel_path}: {e}")
            continue
        chunks = chunk_markdown(rel_path, content, file_mtime=mtime)
        for chunk in chunks:
            chunk["file_mtime"] = mtime
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.info("No chunks produced — empty vault or all files empty")
        return {"files_indexed": 0, "chunks_created": 0, "files_removed": 0,
                "duration_seconds": round(time.time() - start, 2)}

    logger.info(f"Embedding {len(all_chunks)} chunks in batches of {batch_size}...")
    all_vectors: list[list[float] | None] = []
    texts = [c["text_to_embed"] for c in all_chunks]
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = provider.embed_texts(batch)
        all_vectors.extend(vectors)
        logger.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    assert len(all_vectors) == len(all_chunks), "vector/chunk count mismatch — embedding misalignment"
    records = [_make_record(c, v) for c, v in zip(all_chunks, all_vectors) if v is not None]
    table = db.create_table(TABLE_NAME, data=records)
    _rebuild_fts_index(table)
    _write_schema_version(db_path)

    duration = time.time() - start
    unique_files = len(set(r["file_path"] for r in records))
    logger.info(f"Full index: {unique_files} files, {len(records)} chunks in {duration:.1f}s")
    return {
        "files_indexed": unique_files,
        "chunks_created": len(records),
        "files_removed": 0,
        "duration_seconds": round(duration, 2),
    }


def incremental_index(
    vault_path: str,
    db_path: str,
    provider=None,
    batch_size: int = BATCH_SIZE,
    single_file: str | None = None,
    only_files: set[str] | None = None,
) -> dict:
    """Update index with only changed/new/deleted files.

    Scope can be limited to a set of vault-relative paths via only_files (or a
    single path via single_file, kept as a thin alias). Reads only the
    file_path/file_mtime columns from the index — never materialises vectors.
    """
    from some_vault_some_mcp.core.chunker import chunk_markdown
    if provider is None:
        from some_vault_some_mcp.core.embeddings import get_provider
        provider = get_provider()
    if single_file is not None and only_files is None:
        only_files = {single_file}

    start = time.time()
    db = _get_db(db_path)
    table = _get_table(db)

    if table is None:
        return full_index(vault_path, db_path, provider, batch_size)

    with _reindex_lock:
        # Get current vault state
        current_files = dict(scan_vault(vault_path))
        if only_files is not None:
            current_files = {k: v for k, v in current_files.items() if k in only_files}

        # Get indexed states — project only the needed columns (never vectors).
        df = table.search().select(["file_path", "file_mtime"]).to_pandas()
        indexed_mtimes: dict[str, float] = {}
        for _, row in df.drop_duplicates("file_path").iterrows():
            indexed_mtimes[row["file_path"]] = row["file_mtime"]
        if only_files is not None:
            indexed_mtimes = {k: v for k, v in indexed_mtimes.items() if k in only_files}

        to_reindex = [
            (rel, mtime) for rel, mtime in current_files.items()
            if rel not in indexed_mtimes or mtime > indexed_mtimes[rel]
        ]
        deleted = set(indexed_mtimes.keys()) - set(current_files.keys())

        if not to_reindex and not deleted:
            return {
                "files_indexed": 0, "chunks_created": 0, "files_removed": 0,
                "duration_seconds": round(time.time() - start, 2),
            }

        # Remove old chunks for reindexed/deleted files
        paths_to_remove = {p for p, _ in to_reindex} | deleted
        if paths_to_remove:
            from some_vault_some_mcp.core.filters import escape_string
            filter_expr = " OR ".join(f'file_path = "{escape_string(p)}"' for p in paths_to_remove)
            table.delete(filter_expr)

        vault = Path(vault_path)
        new_chunks: list[dict] = []
        for rel_path, mtime in to_reindex:
            full_path = vault / rel_path
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"Could not read {rel_path}: {e}")
                continue
            chunks = chunk_markdown(rel_path, content, file_mtime=mtime)
            for chunk in chunks:
                chunk["file_mtime"] = mtime
            new_chunks.extend(chunks)

        records: list[dict] = []
        if new_chunks:
            texts = [c["text_to_embed"] for c in new_chunks]
            all_vectors: list[list[float] | None] = []
            for i in range(0, len(texts), batch_size):
                all_vectors.extend(provider.embed_texts(texts[i:i + batch_size]))

            assert len(all_vectors) == len(new_chunks), "vector/chunk count mismatch — embedding misalignment"
            records = [_make_record(c, v) for c, v in zip(new_chunks, all_vectors) if v is not None]
            table.add(records)

        _rebuild_fts_index(table)

        duration = time.time() - start
        return {
            "files_indexed": len(to_reindex),
            "chunks_created": len(records),
            "files_removed": len(deleted),
            "duration_seconds": round(duration, 2),
        }


def get_index_status(vault_path: str, db_path: str, provider_dims: int | None = None) -> dict:
    """Return index health information."""
    db = _get_db(db_path)
    table = _get_table(db)

    if table is None:
        return {
            "total_chunks": 0, "total_files": 0, "pending_reindex": 0,
            "db_size_mb": 0.0,
        }

    total_chunks = table.count_rows()
    df = table.search().select(["file_path", "file_mtime"]).to_pandas()
    total_files = df["file_path"].nunique() if not df.empty else 0

    # Pending reindex count
    pending = 0
    if vault_path:
        current_files = dict(scan_vault(vault_path))
        indexed_mtimes: dict[str, float] = {}
        for _, row in df.drop_duplicates("file_path").iterrows():
            indexed_mtimes[row["file_path"]] = row["file_mtime"]
        for rel_path, mtime in current_files.items():
            if rel_path not in indexed_mtimes or mtime > indexed_mtimes[rel_path]:
                pending += 1

    # DB size
    db_size = 0.0
    if os.path.exists(db_path):
        for dirpath, _, filenames in os.walk(db_path):
            for f in filenames:
                try:
                    db_size += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
    db_size_mb = round(db_size / (1024 * 1024), 2)

    return {
        "total_chunks": total_chunks,
        "total_files": total_files,
        "pending_reindex": pending,
        "db_size_mb": db_size_mb,
    }
