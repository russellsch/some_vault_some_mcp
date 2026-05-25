"""Read tools: get_note and list_notes."""

import logging
from pathlib import Path

from some_vault_some_mcp.core.filters import escape_filter_value
from some_vault_some_mcp.core.frontmatter import parse_frontmatter, extract_all_tags
from some_vault_some_mcp.core.paths import (
    VaultPathError,
    ensure_md_extension,
    resolve_note_path,
    resolve_vault_path,
    walk_vault,
)
from some_vault_some_mcp.models import NoteContent, NoteMetadata

logger = logging.getLogger(__name__)


def _resolve_note_relpath(vault_path: str, path: str) -> str | None:
    """Map a user-supplied note reference to a vault-relative .md path.

    Fast path: the literal path (with .md ensured) when it exists on disk.
    Fallback: Obsidian-style resolution (exact relative path or basename)
    against every note in the vault.
    """
    candidate = ensure_md_extension(path)
    try:
        full_path = resolve_vault_path(vault_path, candidate)
    except VaultPathError:
        return None
    if Path(full_path).exists():
        return candidate
    return resolve_note_path(candidate, walk_vault(vault_path))


def get_note(vault_path: str, path: str) -> NoteContent | None:
    """Read a single note. Path is extension-agnostic and resolves
    Obsidian-style ("todo", "todo.md", and a bare basename all work)."""
    rel_path = _resolve_note_relpath(vault_path, path)
    if rel_path is None:
        return None

    try:
        full_path = resolve_vault_path(vault_path, rel_path)
    except VaultPathError:
        return None

    p = Path(full_path)
    if not p.exists():
        return None

    content = p.read_text(encoding="utf-8", errors="replace")
    fm, body = parse_frontmatter(content)
    tags = extract_all_tags(content)
    title = fm.get("title") or p.stem

    return NoteContent(
        file_path=rel_path,
        title=str(title),
        content=body,
        frontmatter=fm,
        tags=tags,
    )


def list_notes(
    vault_path: str,
    db_path: str | None = None,
    folder: str | None = None,
    tags: list[str] | None = None,
    projects: list[str] | None = None,
    status: str | None = None,
    area: str | None = None,
    frontmatter_property: str | None = None,
    frontmatter_value: str | None = None,
    include_content: bool = False,
    limit: int = 50,
) -> tuple[list[NoteMetadata], int]:
    """Enumerate vault notes with optional filtering.

    Returns (results, total_count).

    Filtering strategy:
    - index-backed fields (tags/projects/status/area): query LanceDB if db_path given
    - frontmatter_property: scan files (or filtered candidate set from index)
    - no filters: filesystem walk
    """
    use_index = db_path and (tags or projects or status or area)

    candidate_paths: list[str] | None = None

    if use_index:
        candidate_paths = _list_from_index(db_path, tags, projects, status, area)

    if frontmatter_property and frontmatter_value:
        candidate_paths = _filter_by_frontmatter(
            vault_path, candidate_paths, frontmatter_property, frontmatter_value
        )

    if candidate_paths is None:
        # Filesystem walk
        candidate_paths = walk_vault(vault_path)
        if folder:
            candidate_paths = [p for p in candidate_paths if p.startswith(folder + "/") or p.startswith(folder)]

    # Apply folder filter
    if folder and candidate_paths is not None:
        candidate_paths = [
            p for p in candidate_paths
            if p.startswith(folder.rstrip("/") + "/") or p == folder
        ]

    candidate_paths = sorted(set(candidate_paths))
    total = len(candidate_paths)
    limited = candidate_paths[:limit]

    results = []
    for rel_path in limited:
        meta = _build_metadata(vault_path, rel_path, include_content)
        if meta:
            results.append(meta)

    return results, total


def _list_from_index(
    db_path: str,
    tags: list[str] | None,
    projects: list[str] | None,
    status: str | None,
    area: str | None,
) -> list[str]:
    """Query LanceDB for file paths matching metadata filters (pre-filter)."""
    try:
        import lancedb
        db = lancedb.connect(db_path)
        if "vault_chunks" not in db.list_tables().tables:
            return []
        table = db.open_table("vault_chunks")

        conditions = []
        if tags:
            tag_conditions = [f'tags LIKE "%{escape_filter_value(t)}%"' for t in tags]
            conditions.append(f"({' OR '.join(tag_conditions)})")
        if projects:
            proj_conditions = [f'projects LIKE "%{escape_filter_value(p)}%"' for p in projects]
            conditions.append(f"({' OR '.join(proj_conditions)})")
        if status:
            conditions.append(f'status = "{escape_filter_value(status)}"')
        if area:
            conditions.append(f'area LIKE "%{escape_filter_value(area)}%"')

        where = " AND ".join(conditions)
        df = table.search().where(where).select(["file_path"]).to_pandas()
        return list(df["file_path"].unique())
    except Exception as e:
        logger.warning(f"Index filter failed: {e}")
        return []


def _filter_by_frontmatter(
    vault_path: str,
    candidate_paths: list[str] | None,
    prop: str,
    value: str,
) -> list[str]:
    """Scan files for frontmatter property=value match (case-insensitive)."""
    if candidate_paths is None:
        candidate_paths = walk_vault(vault_path)

    value_lower = value.lower()
    matching = []
    for rel_path in candidate_paths:
        full_path = Path(vault_path) / rel_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        fm, _ = parse_frontmatter(content)
        prop_val = fm.get(prop)
        if prop_val is None:
            continue
        if isinstance(prop_val, list):
            if any(str(v).lower() == value_lower for v in prop_val):
                matching.append(rel_path)
        else:
            if str(prop_val).lower() == value_lower:
                matching.append(rel_path)

    return matching


def _build_metadata(vault_path: str, rel_path: str, include_content: bool) -> NoteMetadata | None:
    full_path = Path(vault_path) / rel_path
    try:
        content = full_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    fm, _ = parse_frontmatter(content)
    tags_raw = fm.get("tags", [])
    if isinstance(tags_raw, str):
        tags_raw = [t.strip() for t in tags_raw.split(",") if t.strip()]
    elif not isinstance(tags_raw, list):
        tags_raw = []

    projects_raw = fm.get("projects", [])
    if isinstance(projects_raw, str):
        projects_raw = [p.strip() for p in projects_raw.split(",") if p.strip()]
    elif not isinstance(projects_raw, list):
        projects_raw = []

    title = fm.get("title") or full_path.stem
    status = fm.get("status")
    area = fm.get("area")
    created = (
        str(fm.get("date created") or fm.get("dateCreated") or fm.get("date") or fm.get("created") or "")
        or None
    )

    return NoteMetadata(
        file_path=rel_path,
        title=str(title),
        tags=[str(t).lower() for t in tags_raw],
        projects=[str(p) for p in projects_raw],
        status=str(status) if status else None,
        area=str(area) if area else None,
        created=created,
    )
