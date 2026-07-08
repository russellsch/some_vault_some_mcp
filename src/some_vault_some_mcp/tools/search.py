"""Search tool — hybrid, semantic, and exact text search."""

import logging
import os
from pathlib import Path

from some_vault_some_mcp.core.filters import escape_like, like_token, split_tokens
from some_vault_some_mcp.models import SearchResult, TextSearchMatch, TextSearchResult

logger = logging.getLogger(__name__)

# Hybrid scoring weights and boost factor — exposed for unit testing
_SEMANTIC_WEIGHT = 0.7
_KW_WEIGHT = 0.3
_BOOST_FACTOR = 1.2


def _get_table_and_db(db_path: str):
    import lancedb
    db = lancedb.connect(db_path)
    if "vault_chunks" not in db.list_tables().tables:
        return None, None
    return db, db.open_table("vault_chunks")


def semantic_search(
    query: str,
    db_path: str,
    provider,
    top_k: int = 10,
    tags: list[str] | None = None,
    folder: str | None = None,
) -> list[SearchResult]:
    """Pure vector similarity search with pre-filter on tags/folder."""
    db, table = _get_table_and_db(db_path)
    if table is None:
        return []

    query_vector = provider.embed_query(query)

    # Build where clause for pre-filter
    conditions = []
    if tags:
        tag_conditions = [like_token("tags", t) for t in tags]
        conditions.append(f"({' OR '.join(tag_conditions)})")
    if folder:
        conditions.append(f'file_path LIKE "{escape_like(folder.rstrip("/") + "/")}%"')

    search_q = table.search(query_vector).metric("cosine").limit(top_k)
    if conditions:
        search_q = search_q.where(" AND ".join(conditions))

    results = search_q.to_pandas()
    if results.empty:
        return []

    return [
        SearchResult(
            title=row.get("title", ""),
            file_path=row["file_path"],
            heading=row.get("heading") or None,
            snippet=row["content"][:300],
            score=round(max(0.0, 1 - float(row.get("_distance", 1))), 4),
            tags=split_tokens(row.get("tags", "")),
            projects=split_tokens(row.get("projects", "")),
            area=row.get("area") or None,
        )
        for _, row in results.iterrows()
    ]


def hybrid_search(
    query: str,
    db_path: str,
    provider,
    top_k: int = 10,
    tags: list[str] | None = None,
    folder: str | None = None,
) -> list[SearchResult]:
    """Hybrid search: 70% semantic + 30% FTS keyword, 1.2x boost on both match."""
    db, table = _get_table_and_db(db_path)
    if table is None:
        return []

    query_vector = provider.embed_query(query)

    # Build where clause for pre-filter
    conditions = []
    if tags:
        tag_conditions = [like_token("tags", t) for t in tags]
        conditions.append(f"({' OR '.join(tag_conditions)})")
    if folder:
        conditions.append(f'file_path LIKE "{escape_like(folder.rstrip("/") + "/")}%"')
    where_clause = " AND ".join(conditions) if conditions else None

    # Semantic pass
    sem_q = table.search(query_vector).metric("cosine").limit(top_k * 2)
    if where_clause:
        sem_q = sem_q.where(where_clause)
    sem_results = sem_q.to_pandas()

    # FTS pass
    try:
        fts_q = table.search(query, query_type="fts").limit(top_k * 2)
        if where_clause:
            fts_q = fts_q.where(where_clause)
        kw_results = fts_q.to_pandas()
    except Exception as e:
        logger.warning(f"FTS search failed (index may not exist): {e}")
        kw_results = sem_results.iloc[0:0]

    # Normalize FTS scores
    max_fts = 0.0
    if not kw_results.empty and "_score" in kw_results.columns:
        max_fts = float(kw_results["_score"].max())

    seen: dict[str, dict] = {}
    for _, row in sem_results.iterrows():
        key = f"{row['file_path']}:{row['chunk_index']}"
        seen[key] = {
            "row": row,
            "sem_score": max(0.0, 1 - float(row.get("_distance", 1))),
            "kw_score": 0.0,
        }
    for _, row in kw_results.iterrows():
        key = f"{row['file_path']}:{row['chunk_index']}"
        raw = float(row.get("_score", 0))
        norm = raw / max_fts if max_fts > 0 else 0.0
        if key in seen:
            seen[key]["kw_score"] = norm
        else:
            seen[key] = {"row": row, "sem_score": 0.0, "kw_score": norm}

    ranked = []
    for key, data in seen.items():
        combined = data["sem_score"] * _SEMANTIC_WEIGHT + data["kw_score"] * _KW_WEIGHT
        if data["sem_score"] > 0 and data["kw_score"] > 0:
            combined *= _BOOST_FACTOR
        combined = min(combined, 1.0)
        ranked.append((combined, data["row"]))

    ranked.sort(key=lambda x: x[0], reverse=True)

    return [
        SearchResult(
            title=row.get("title", ""),
            file_path=row["file_path"],
            heading=row.get("heading") or None,
            snippet=row["content"][:300],
            score=round(score, 4),
            tags=split_tokens(row.get("tags", "")),
            projects=split_tokens(row.get("projects", "")),
            area=row.get("area") or None,
        )
        for score, row in ranked[:top_k]
    ]


def exact_search(
    query: str,
    vault_path: str,
    top_k: int = 10,
    folder: str | None = None,
    case_sensitive: bool = False,
    tags: list[str] | None = None,
) -> list[TextSearchResult]:
    """Literal substring match across vault files."""
    from some_vault_some_mcp.core.paths import walk_vault
    from some_vault_some_mcp.core.frontmatter import extract_all_tags

    all_files = walk_vault(vault_path)
    if folder:
        all_files = [f for f in all_files if f.startswith(folder.rstrip("/") + "/")]

    search_str = query if case_sensitive else query.lower()
    results: list[TextSearchResult] = []

    for rel_path in all_files:
        if len(results) >= top_k:
            break
        full_path = Path(vault_path) / rel_path
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Post-filter by tags if requested
        if tags:
            note_tags = set(extract_all_tags(content))
            if not any(t.lower() in note_tags for t in tags):
                continue

        lines = content.split("\n")
        matches = []
        for i, line in enumerate(lines):
            compare = line if case_sensitive else line.lower()
            if search_str in compare:
                matches.append(TextSearchMatch(line=i + 1, content=line.strip()))

        if matches:
            results.append(TextSearchResult(relative_path=rel_path, matches=matches))

    return results
