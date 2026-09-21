"""Crash-safe LanceDB indexing with generation-based publication.

Full rebuilds are constructed in a uniquely named table and become visible only
after validation and an atomic manifest update. The legacy ``vault_chunks``
table remains readable for installations that do not yet have a manifest.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TABLE_NAME = "vault_chunks"
GENERATION_PREFIX = "vault_chunks__gen_"
MANIFEST_NAME = "_index_manifest.json"
MANIFEST_FORMAT_VERSION = 1
BATCH_SIZE = 50
SCHEMA_VERSION = 3  # v3 adds content_hash

_GENERATION_RE = re.compile(r"^vault_chunks__gen_[0-9a-f]{32}$")
_reindex_lock = threading.RLock()
_database_lock_state = threading.local()
_state_lock = threading.Lock()
_cutover_locks_guard = threading.Lock()
_cutover_locks: dict[str, "_ThreadReadWriteLock"] = {}


class _ThreadReadWriteLock:
    """Process-local half of the generation cutover lock."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    def acquire(self, *, exclusive: bool) -> None:
        with self._condition:
            if exclusive:
                self._waiting_writers += 1
                try:
                    while self._writer or self._readers:
                        self._condition.wait()
                    self._writer = True
                finally:
                    self._waiting_writers -= 1
            else:
                while self._writer or self._waiting_writers:
                    self._condition.wait()
                self._readers += 1

    def release(self, *, exclusive: bool) -> None:
        with self._condition:
            if exclusive:
                self._writer = False
            else:
                self._readers -= 1
            self._condition.notify_all()


def _thread_cutover_lock(key: str) -> _ThreadReadWriteLock:
    with _cutover_locks_guard:
        return _cutover_locks.setdefault(key, _ThreadReadWriteLock())


@contextmanager
def _generation_cutover_lock(db_path: str, *, exclusive: bool):
    """Protect table readers from publication-time generation reclamation."""
    key = str(Path(db_path).expanduser().resolve())
    local_lock = _thread_cutover_lock(key)
    local_lock.acquire(exclusive=exclusive)
    try:
        lock_path = Path(key) / ".index-cutover.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                if lock_path.stat().st_size == 0:
                    handle.write(b"\0")
                    handle.flush()
                mode = msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK
                while True:
                    try:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), mode, 1)
                        break
                    except OSError as exc:
                        if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                            raise
                        time.sleep(0.05)
            else:
                import fcntl

                mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                fcntl.flock(handle.fileno(), mode)
            try:
                yield
            finally:
                if sys.platform == "win32":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
    finally:
        local_lock.release(exclusive=exclusive)


@contextmanager
def _database_lock(db_path: str):
    """Hold a reentrant, database-scoped inter-process publication lock."""
    key = str(Path(db_path).expanduser().resolve())
    held = getattr(_database_lock_state, "held", None)
    if held is None:
        held = _database_lock_state.held = set()
    if key in held:
        yield
        return

    lock_path = Path(key) / ".index-publication.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        if sys.platform == "win32":
            import msvcrt

            if lock_path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        held.add(key)
        try:
            yield
        finally:
            held.remove(key)
            if sys.platform == "win32":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


@dataclass(frozen=True)
class ScannedFile:
    relative_path: str
    mtime: float


@dataclass
class VaultScan:
    files: dict[str, ScannedFile] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    complete: bool = True


@dataclass
class _IndexRuntimeState:
    rebuild_in_progress: bool = False
    last_rebuild_error: str | None = None
    serving_degraded: bool = False


_runtime_states: dict[str, _IndexRuntimeState] = {}


def _state_key(db_path: str) -> str:
    return str(Path(db_path).expanduser().resolve())


def _get_runtime_state(db_path: str) -> _IndexRuntimeState:
    key = _state_key(db_path)
    with _state_lock:
        state = _runtime_states.setdefault(key, _IndexRuntimeState())
        return _IndexRuntimeState(**state.__dict__)


def _update_runtime_state(db_path: str, **updates: Any) -> None:
    key = _state_key(db_path)
    with _state_lock:
        state = _runtime_states.setdefault(key, _IndexRuntimeState())
        for name, value in updates.items():
            setattr(state, name, value)


def _schema_version_file(db_path: str) -> str:
    return os.path.join(db_path, "_schema_version.json")


def _write_schema_version(db_path: str) -> None:
    try:
        path = Path(_schema_version_file(db_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": SCHEMA_VERSION}), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write schema version marker: %s", exc)


def _read_schema_version(db_path: str) -> int | None:
    try:
        with open(_schema_version_file(db_path), encoding="utf-8") as handle:
            return int(json.load(handle).get("version"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _manifest_path(db_path: str) -> Path:
    return Path(db_path) / MANIFEST_NAME


def _valid_owned_table_name(name: object, *, allow_legacy: bool = True) -> bool:
    if not isinstance(name, str):
        return False
    return (allow_legacy and name == TABLE_NAME) or bool(_GENERATION_RE.fullmatch(name))


def _read_manifest(db_path: str) -> dict[str, Any] | None:
    try:
        raw = json.loads(_manifest_path(db_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("format_version") != MANIFEST_FORMAT_VERSION or raw.get("completion") != "complete":
        return None
    if not _valid_owned_table_name(raw.get("active")):
        return None
    previous = raw.get("previous")
    if previous is not None and not _valid_owned_table_name(previous):
        return None
    try:
        int(raw["schema_version"])
        int(raw["vector_dimensions"])
    except (KeyError, TypeError, ValueError):
        return None
    return raw


def _durable_replace(temp_path: Path, target_path: Path) -> None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
        move.restype = wintypes.BOOL
        if not move(str(temp_path), str(target_path), 0x1 | 0x8):
            raise OSError(ctypes.get_last_error(), "durable manifest replacement failed")
        return
    os.replace(temp_path, target_path)
    directory_fd = os.open(str(target_path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publish_manifest(db_path: str, manifest: dict[str, Any]) -> None:
    target = _manifest_path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        _durable_replace(temp, target)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _get_db(db_path: str):
    import lancedb

    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    return lancedb.connect(db_path)


def _db_path_from_connection(db) -> str:
    return str(getattr(db, "uri", getattr(db, "_uri", "")))


def _schema_dimensions(table) -> int | None:
    try:
        vector_field = next(field for field in table.schema if field.name == "vector")
    except (StopIteration, AttributeError):
        return None
    return getattr(vector_field.type, "list_size", None)


def _table_has_current_schema(table, provider_dims: int | None = None) -> bool:
    try:
        fields = {field.name for field in table.schema}
    except Exception:
        return False
    required = {
        "file_path", "chunk_index", "heading", "content", "title", "tags", "projects",
        "area", "status", "source", "file_mtime", "content_hash", "vector",
    }
    if not required.issubset(fields):
        return False
    dims = _schema_dimensions(table)
    return provider_dims is None or dims == provider_dims


def _resolve_table_name(db, db_path: str, provider_dims: int | None = None) -> tuple[str | None, bool]:
    names = set(db.list_tables().tables)
    manifest = _read_manifest(db_path)
    if manifest is not None:
        active = manifest["active"]
        if active in names:
            try:
                if _table_has_current_schema(db.open_table(active), provider_dims):
                    return active, False
            except Exception:
                pass
        previous = manifest.get("previous")
        if previous in names:
            try:
                table = db.open_table(previous)
                if provider_dims is None or _schema_dimensions(table) == provider_dims:
                    _update_runtime_state(db_path, serving_degraded=True)
                    return previous, True
            except Exception:
                pass
    if TABLE_NAME in names:
        try:
            table = db.open_table(TABLE_NAME)
            if provider_dims is None or _schema_dimensions(table) in (None, provider_dims):
                return TABLE_NAME, False
        except Exception:
            return None, False
    return None, False


def _get_table(db, table_name: str = TABLE_NAME, provider_dims: int | None = None):
    if table_name != TABLE_NAME:
        return db.open_table(table_name) if table_name in db.list_tables().tables else None
    name, _ = _resolve_table_name(db, _db_path_from_connection(db), provider_dims)
    return db.open_table(name) if name is not None else None


def resolve_active_table(db_path: str, provider_dims: int | None = None):
    db = _get_db(db_path)
    name, degraded = _resolve_table_name(db, db_path, provider_dims)
    return db, (db.open_table(name) if name else None), name, degraded


@contextmanager
def active_table_reader(db_path: str, provider_dims: int | None = None):
    """Yield the published table while preventing live generation cleanup."""
    with _generation_cutover_lock(db_path, exclusive=False):
        yield resolve_active_table(db_path, provider_dims)


def _data_matches_current_format(table) -> bool:
    if not _table_has_current_schema(table):
        return False
    try:
        frame = table.search().select(["tags", "projects"]).to_pandas()
    except Exception:
        return False
    return all(
        not (isinstance(value, str) and value and not value.startswith(","))
        for column in ("tags", "projects") for value in frame[column]
    )


def check_and_maybe_migrate(db, db_path: str) -> bool:
    table = _get_table(db)
    if table is None:
        return False
    if _read_schema_version(db_path) == SCHEMA_VERSION and _data_matches_current_format(table):
        return False
    if _data_matches_current_format(table):
        _write_schema_version(db_path)
        return False
    return True


def _check_dimension_mismatch(db, provider_dims: int) -> None:
    table = _get_table(db)
    if table is None:
        return
    existing_dims = _schema_dimensions(table)
    if existing_dims is not None and existing_dims != provider_dims:
        raise RuntimeError(
            f"Dimension mismatch: provider produces {provider_dims}-dim vectors "
            f"but existing index has {existing_dims}-dim vectors. "
            "Full reindex required — run with --reindex-force."
        )


def _arrow_schema(dimensions: int):
    import pyarrow as pa

    return pa.schema([
        pa.field("file_path", pa.string()), pa.field("chunk_index", pa.int64()),
        pa.field("heading", pa.string()), pa.field("content", pa.string()),
        pa.field("title", pa.string()), pa.field("tags", pa.string()),
        pa.field("projects", pa.string()), pa.field("area", pa.string()),
        pa.field("status", pa.string()), pa.field("source", pa.string()),
        pa.field("file_mtime", pa.float64()), pa.field("content_hash", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dimensions)),
    ])


def _rebuild_fts_index(table, *, strict: bool = False) -> None:
    if table.count_rows() == 0:
        return
    try:
        table.create_fts_index("content", replace=True)
        table.search("index-validation", query_type="fts").limit(1).to_pandas()
    except Exception as exc:
        if strict:
            raise RuntimeError(f"FTS index validation failed: {exc}") from exc
        logger.warning("FTS index rebuild failed: %s", exc)


def _scan_vault_complete(vault_path: str) -> VaultScan:
    from some_vault_some_mcp.core.paths import _within_vault, is_index_excluded

    vault = Path(vault_path)
    vault_root = vault.resolve()
    result = VaultScan()

    def visit(directory: Path, rel_dir: str = "") -> None:
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            result.complete = False
            result.errors.append(f"{rel_dir or '.'}: {exc}")
            return
        for entry in entries:
            rel = (f"{rel_dir}/{entry.name}" if rel_dir else entry.name).replace("\\", "/")
            path = Path(entry.path)
            try:
                if entry.is_dir(follow_symlinks=False):
                    if not is_index_excluded(f"{rel}/_"):
                        visit(path, rel)
                    continue
                if not entry.name.lower().endswith(".md"):
                    continue
                if is_index_excluded(rel) or not _within_vault(path, vault_root):
                    continue
                stat = entry.stat(follow_symlinks=True)
                result.files[rel] = ScannedFile(rel, stat.st_mtime)
            except OSError as exc:
                result.skipped.add(rel)
                result.errors.append(f"{rel}: {exc}")

    visit(vault)
    return result


def scan_vault(vault_path: str) -> list[tuple[str, float]]:
    result = _scan_vault_complete(vault_path)
    if not result.complete:
        raise RuntimeError("Vault scan incomplete: " + "; ".join(result.errors))
    return [(rel, item.mtime) for rel, item in sorted(result.files.items())]


def _read_note_bytes(vault_path: str, rel_path: str) -> tuple[bytes, float]:
    from some_vault_some_mcp.core.paths import resolve_vault_path

    full_path = Path(resolve_vault_path(vault_path, rel_path))
    raw = full_path.read_bytes()
    return raw, full_path.stat().st_mtime


def _make_record(chunk: dict, vector: list[float]) -> dict:
    from some_vault_some_mcp.core.filters import store_tokens

    return {
        "file_path": chunk["file_path"], "chunk_index": chunk["chunk_index"],
        "heading": chunk["heading"], "content": chunk["content"],
        "title": chunk.get("title", ""), "tags": store_tokens(chunk.get("tags", [])),
        "projects": store_tokens(chunk.get("projects", [])),
        "area": str(chunk.get("area") or ""), "status": str(chunk.get("status") or ""),
        "source": str(chunk.get("source") or ""), "file_mtime": float(chunk["file_mtime"]),
        "content_hash": str(chunk.get("content_hash", "")), "vector": vector,
    }


def _embed_chunks(chunks: list[dict], provider, batch_size: int) -> list[dict]:
    if not chunks:
        return []
    vectors: list[list[float] | None] = []
    texts = [chunk["text_to_embed"] for chunk in chunks]
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        returned = provider.embed_texts(batch)
        if len(returned) != len(batch):
            raise RuntimeError("vector/chunk count mismatch — embedding misalignment")
        vectors.extend(returned)
    if len(vectors) != len(chunks):
        raise RuntimeError("vector/chunk count mismatch — embedding misalignment")
    records: list[dict] = []
    expected = int(provider.dimensions)
    for chunk, vector in zip(chunks, vectors):
        if vector is None or len(vector) != expected:
            raise RuntimeError("embedding provider returned a missing or wrong-dimension vector")
        if any(not math.isfinite(float(value)) for value in vector):
            raise RuntimeError("embedding provider returned a non-finite vector")
        records.append(_make_record(chunk, [float(value) for value in vector]))
    return records


def _records_from_table(table, paths: set[str] | None = None) -> list[dict]:
    if table is None or table.count_rows() == 0:
        return []
    frame = table.to_pandas()
    if paths is not None:
        frame = frame[frame["file_path"].isin(paths)]
    records: list[dict] = []
    for _, row in frame.iterrows():
        record = {name: row[name] for name in frame.columns if not name.startswith("_")}
        if hasattr(record.get("vector"), "tolist"):
            record["vector"] = record["vector"].tolist()
        if record.get("chunk_index") is not None:
            record["chunk_index"] = int(record["chunk_index"])
        records.append(record)
    return records


def _indexed_hashes(table) -> dict[str, str]:
    if table is None or table.count_rows() == 0 or not _table_has_current_schema(table):
        return {}
    frame = table.search().select(["file_path", "content_hash"]).to_pandas()
    return {str(row["file_path"]): str(row["content_hash"])
            for _, row in frame.drop_duplicates("file_path").iterrows()}


def _validate_candidate(table, provider_dims: int) -> None:
    if not _table_has_current_schema(table, provider_dims):
        raise RuntimeError("candidate index schema or vector dimensions are invalid")
    if table.count_rows() > 0:
        frame = table.search().select(["file_path", "content_hash", "vector"]).to_pandas()
        for _, row in frame.iterrows():
            vector = row["vector"]
            if not isinstance(row["file_path"], str) or not isinstance(row["content_hash"], str):
                raise RuntimeError("candidate contains an invalid path or content hash")
            if len(vector) != provider_dims or any(not math.isfinite(float(v)) for v in vector):
                raise RuntimeError("candidate contains an invalid vector")
    _rebuild_fts_index(table, strict=True)


def _drop_unpublished(db, table_name: str) -> None:
    if _GENERATION_RE.fullmatch(table_name) and table_name in db.list_tables().tables:
        try:
            db.drop_table(table_name)
        except Exception as exc:
            logger.warning("Could not remove unpublished generation %s: %s", table_name, exc)


def _process_files(vault_path: str, scan: VaultScan, provider, batch_size: int, *,
                   old_table=None, allow_carry_forward: bool = False):
    from some_vault_some_mcp.core.chunker import chunk_markdown

    chunks: list[dict] = []
    processed: set[str] = set()
    skipped: set[str] = set(scan.skipped)
    for rel_path in sorted(scan.files):
        try:
            raw, mtime = _read_note_bytes(vault_path, rel_path)
            content_hash = hashlib.sha256(raw).hexdigest()
            file_chunks = chunk_markdown(rel_path, raw.decode("utf-8", errors="replace"), file_mtime=mtime)
            for chunk in file_chunks:
                chunk["file_mtime"] = mtime
                chunk["content_hash"] = content_hash
            chunks.extend(file_chunks)
            processed.add(rel_path)
        except Exception as exc:
            logger.warning("Skipping %s during rebuild: %s", rel_path, exc)
            skipped.add(rel_path)
    records = _embed_chunks(chunks, provider, batch_size)
    if allow_carry_forward and old_table is not None and skipped:
        # A yielded entry whose stat/read failed is uncertain, not deleted.
        # It already passed the traversal's boundary/exclusion checks; retain
        # its last-known-good rows until a complete later scan can decide.
        records.extend(_records_from_table(old_table, skipped))
    return records, processed, skipped


def _manifest_for(active: str, previous: str | None, dimensions: int) -> dict[str, Any]:
    return {"format_version": MANIFEST_FORMAT_VERSION, "active": active, "previous": previous,
            "schema_version": SCHEMA_VERSION, "vector_dimensions": int(dimensions),
            "completion": "complete"}


def full_index(vault_path: str, db_path: str, provider=None, batch_size: int = BATCH_SIZE, *, watcher=None) -> dict:
    if provider is None:
        from some_vault_some_mcp.core.embeddings import get_provider
        provider = get_provider()
    if watcher is None:
        try:
            from some_vault_some_mcp.core.watcher import get_watcher
            watcher = get_watcher(db_path)
        except Exception:
            watcher = None
    start = time.time()
    _update_runtime_state(db_path, rebuild_in_progress=True, last_rebuild_error=None)
    if watcher is not None:
        watcher.begin_buffering()
    with _reindex_lock, _database_lock(db_path):
        db = _get_db(db_path)
        old_name, _ = _resolve_table_name(db, db_path, provider.dimensions)
        old_table = db.open_table(old_name) if old_name is not None else None
        compatible_old = bool(old_table is not None and _table_has_current_schema(old_table, provider.dimensions))
        candidate_name = f"{GENERATION_PREFIX}{uuid.uuid4().hex}"
        publication_attempted = False
        try:
            scan = _scan_vault_complete(vault_path)
            if not scan.complete:
                raise RuntimeError("Vault scan incomplete: " + "; ".join(scan.errors))
            records, processed, skipped = _process_files(
                vault_path, scan, provider, batch_size, old_table=old_table,
                allow_carry_forward=compatible_old,
            )
            if scan.files and not processed:
                raise RuntimeError("Every scanned note failed during full indexing")
            table = db.create_table(candidate_name, data=records if records else None,
                                    schema=_arrow_schema(provider.dimensions))
            if watcher is not None:
                watcher.acquire_cutover()
            try:
                final_scan = _scan_vault_complete(vault_path)
                if not final_scan.complete:
                    raise RuntimeError("Final vault scan incomplete: " + "; ".join(final_scan.errors))
                _sync_table_to_scan(table, vault_path, final_scan, provider, batch_size, strict=True)
                _validate_candidate(table, provider.dimensions)
                old_paths = set(_indexed_hashes(old_table)) if old_table is not None else set()
                chunk_count = table.count_rows()
                publication_attempted = True
                with _generation_cutover_lock(db_path, exclusive=True):
                    _publish_manifest(
                        db_path,
                        _manifest_for(candidate_name, old_name, provider.dimensions),
                    )
                    _cleanup_inactive_generations_best_effort(db_path, db)
                _write_schema_version(db_path)
                if watcher is not None:
                    watcher.commit_cutover()
            finally:
                if watcher is not None:
                    watcher.release_cutover()
            _update_runtime_state(db_path, rebuild_in_progress=False, last_rebuild_error=None,
                                  serving_degraded=False)
            return {"files_indexed": len(processed), "chunks_created": chunk_count,
                    "files_removed": len(old_paths - set(final_scan.files)),
                    "files_skipped": len(skipped | final_scan.skipped),
                    "duration_seconds": round(time.time() - start, 2)}
        except Exception as exc:
            if not publication_attempted:
                _drop_unpublished(db, candidate_name)
            _update_runtime_state(db_path, rebuild_in_progress=False, last_rebuild_error=str(exc),
                                  serving_degraded=compatible_old or publication_attempted)
            if watcher is not None:
                watcher.abort_cutover()
            raise


def _delete_paths(table, paths: set[str]) -> set[str]:
    from some_vault_some_mcp.core.filters import escape_string

    if not paths:
        return set()
    expression = " OR ".join(f"file_path = '{escape_string(path)}'" for path in paths)
    try:
        table.delete(expression)
        return set()
    except Exception as exc:
        logger.warning("Batched delete of %d paths failed (%s); retrying per path", len(paths), exc)
    failed: set[str] = set()
    for path in paths:
        try:
            table.delete(f"file_path = '{escape_string(path)}'")
        except Exception as exc:
            logger.warning("Delete failed for %s: %s", path, exc)
            failed.add(path)
    return failed


def _sync_table_to_scan(table, vault_path: str, scan: VaultScan, provider, batch_size: int,
                        *, only_files: set[str] | None = None, strict: bool = False) -> dict:
    from some_vault_some_mcp.core.chunker import chunk_markdown

    if not scan.complete:
        raise RuntimeError("Cannot infer deletions from an incomplete vault scan")
    indexed = _indexed_hashes(table)
    current_paths = set(scan.files)
    if only_files is not None:
        current_paths &= only_files
        indexed = {path: value for path, value in indexed.items() if path in only_files}
    chunks: list[dict] = []
    processed: set[str] = set()
    skipped: set[str] = set()
    for rel_path in sorted(current_paths):
        try:
            raw, mtime = _read_note_bytes(vault_path, rel_path)
            content_hash = hashlib.sha256(raw).hexdigest()
            if indexed.get(rel_path) == content_hash:
                continue
            file_chunks = chunk_markdown(rel_path, raw.decode("utf-8", errors="replace"), file_mtime=mtime)
            for chunk in file_chunks:
                chunk["file_mtime"] = mtime
                chunk["content_hash"] = content_hash
            chunks.extend(file_chunks)
            processed.add(rel_path)
        except Exception as exc:
            logger.warning("Skipping %s during incremental indexing: %s", rel_path, exc)
            skipped.add(rel_path)
    records = _embed_chunks(chunks, provider, batch_size)
    # Paths whose individual metadata probe failed are uncertain. They must
    # not be inferred deleted from an otherwise complete directory traversal.
    deleted = set(indexed) - current_paths - scan.skipped
    failed_delete = _delete_paths(table, processed | deleted)
    if failed_delete:
        records = [record for record in records if record["file_path"] not in failed_delete]
    if records:
        table.add(records)
    _rebuild_fts_index(table, strict=strict)
    return {"files_indexed": len(processed - failed_delete), "chunks_created": len(records),
            "files_removed": len(deleted - failed_delete),
            "files_skipped": len(skipped) + len(failed_delete)}


def incremental_index(vault_path: str, db_path: str, provider=None, batch_size: int = BATCH_SIZE,
                      single_file: str | None = None, only_files: set[str] | None = None) -> dict:
    if provider is None:
        from some_vault_some_mcp.core.embeddings import get_provider
        provider = get_provider()
    if single_file is not None and only_files is None:
        only_files = {single_file}
    watcher = None
    if only_files is not None:
        try:
            from some_vault_some_mcp.core.watcher import get_watcher

            watcher = get_watcher(db_path)
        except Exception:
            watcher = None
        if watcher is not None and watcher.is_buffering:
            return full_index(
                vault_path,
                db_path,
                provider,
                batch_size,
                watcher=watcher,
            )
    start = time.time()
    with _reindex_lock, _database_lock(db_path):
        db, old_table, old_name, _ = resolve_active_table(db_path, provider.dimensions)
        if old_table is None:
            return full_index(vault_path, db_path, provider, batch_size)
        scan = _scan_vault_complete(vault_path)
        if not scan.complete:
            raise RuntimeError("Vault scan incomplete: " + "; ".join(scan.errors))
        candidate_name = f"{GENERATION_PREFIX}{uuid.uuid4().hex}"
        publication_attempted = False
        try:
            old_records = _records_from_table(old_table)
            table = db.create_table(
                candidate_name,
                data=old_records if old_records else None,
                schema=_arrow_schema(provider.dimensions),
            )
            result = _sync_table_to_scan(
                table,
                vault_path,
                scan,
                provider,
                batch_size,
                only_files=only_files,
                strict=True,
            )
            _validate_candidate(table, provider.dimensions)
            publication_attempted = True
            with _generation_cutover_lock(db_path, exclusive=True):
                _publish_manifest(
                    db_path,
                    _manifest_for(candidate_name, old_name, provider.dimensions),
                )
                _cleanup_inactive_generations_best_effort(db_path, db)
            _write_schema_version(db_path)
            # A watcher failure can race this targeted candidate after the
            # initial mode check. Never report healthy while its journal is
            # still buffering; the next targeted request will promote to a
            # full reconciliation.
            if watcher is None or not watcher.is_buffering:
                _update_runtime_state(
                    db_path,
                    last_rebuild_error=None,
                    serving_degraded=False,
                )
            result["duration_seconds"] = round(time.time() - start, 2)
            return result
        except Exception as exc:
            if not publication_attempted:
                _drop_unpublished(db, candidate_name)
            _update_runtime_state(
                db_path,
                last_rebuild_error=str(exc),
                serving_degraded=True,
            )
            raise


def _cleanup_inactive_generations_unlocked(db_path: str, db) -> None:
    """Drop old owned generations while the exclusive cutover lock is held."""
    manifest = _read_manifest(db_path)
    if manifest is None:
        return
    keep = {manifest["active"], manifest.get("previous")}
    names = set(db.list_tables().tables)
    if manifest["active"] not in names:
        return
    if not _table_has_current_schema(
        db.open_table(manifest["active"]), int(manifest["vector_dimensions"])
    ):
        return
    for name in names:
        if _GENERATION_RE.fullmatch(name) and name not in keep:
            try:
                db.drop_table(name)
            except Exception as exc:
                # Publication is already durable. Reclamation is best effort
                # and will be retried after the next publication or startup.
                logger.warning("Could not remove inactive index generation %s: %s", name, exc)


def _cleanup_inactive_generations_best_effort(db_path: str, db) -> None:
    try:
        _cleanup_inactive_generations_unlocked(db_path, db)
    except Exception as exc:
        # The new manifest may already be visible. Cleanup must never turn a
        # successful publication into an ambiguous failure response.
        logger.warning("Could not inspect inactive index generations: %s", exc)


def cleanup_inactive_generations(db_path: str) -> None:
    with _reindex_lock, _database_lock(db_path):
        db = _get_db(db_path)
        with _generation_cutover_lock(db_path, exclusive=True):
            _cleanup_inactive_generations_unlocked(db_path, db)


def get_index_status(vault_path: str, db_path: str, provider_dims: int | None = None) -> dict:
    with active_table_reader(db_path, provider_dims) as (_, table, _, degraded):
        indexed = _indexed_hashes(table) if table is not None else {}
        total_chunks = table.count_rows() if table is not None else 0
    state = _get_runtime_state(db_path)
    base = {"rebuild_in_progress": state.rebuild_in_progress,
            "last_rebuild_error": state.last_rebuild_error,
            "serving_degraded": state.serving_degraded or degraded}
    if table is None:
        return {"total_chunks": 0, "total_files": 0, "pending_reindex": 0,
                "db_size_mb": 0.0, **base}
    pending = 0
    if vault_path:
        scan = _scan_vault_complete(vault_path)
        if not scan.complete:
            pending = len(scan.errors)
        else:
            current_hashes: dict[str, str] = {}
            for rel_path in scan.files:
                try:
                    raw, _ = _read_note_bytes(vault_path, rel_path)
                    current_hashes[rel_path] = hashlib.sha256(raw).hexdigest()
                except Exception:
                    pending += 1
            pending += sum(indexed.get(path) != digest for path, digest in current_hashes.items())
            pending += len(set(indexed) - set(scan.files))
    size = 0
    if os.path.exists(db_path):
        for dirpath, _, filenames in os.walk(db_path):
            for filename in filenames:
                try:
                    size += os.path.getsize(os.path.join(dirpath, filename))
                except OSError:
                    pass
    return {"total_chunks": total_chunks, "total_files": len(indexed),
            "pending_reindex": pending, "db_size_mb": round(size / (1024 * 1024), 2), **base}
