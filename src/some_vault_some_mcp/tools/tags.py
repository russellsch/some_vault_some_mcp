"""Tag tools: get_tags."""

import logging

from some_vault_some_mcp.core.frontmatter import extract_all_tags

logger = logging.getLogger(__name__)


def get_tags(vault_path: str, sort_by: str = "count") -> list[dict]:
    """Enumerate all unique tags with usage counts.

    Returns list of {tag: str, count: int} sorted by sort_by.
    count = number of notes (not occurrences) that use the tag.
    """
    from some_vault_some_mcp.core.vault_cache import read_all
    all_notes, contents = read_all(vault_path)
    tag_map: dict[str, set[str]] = {}  # tag -> set of file paths

    for rel_path in all_notes:
        content = contents.get(rel_path, "")
        tags = extract_all_tags(content)
        for tag in tags:
            normalized = tag.lower()
            if normalized not in tag_map:
                tag_map[normalized] = set()
            tag_map[normalized].add(rel_path)

    result = [{"tag": tag, "count": len(files)} for tag, files in tag_map.items()]

    if sort_by == "name":
        result.sort(key=lambda x: x["tag"])
    else:
        result.sort(key=lambda x: (-x["count"], x["tag"]))

    return result
