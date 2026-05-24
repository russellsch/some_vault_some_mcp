"""Link/graph tools: backlinks, outlinks, orphans, broken links, graph neighbors."""

import logging
from pathlib import Path

from some_vault_some_mcp.core.frontmatter import extract_aliases
from some_vault_some_mcp.core.paths import resolve_note_path, walk_vault
from some_vault_some_mcp.core.wikilinks import extract_wikilinks, resolve_wikilink

logger = logging.getLogger(__name__)


def _load_vault(vault_path: str) -> tuple[list[str], dict[str, str]]:
    """Load all notes and their content."""
    all_notes = walk_vault(vault_path)
    contents: dict[str, str] = {}
    for rel in all_notes:
        try:
            contents[rel] = (Path(vault_path) / rel).read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return all_notes, contents


def _build_alias_map(all_notes: list[str], contents: dict[str, str]) -> dict[str, str]:
    """Build {alias_lower: note_path}."""
    amap: dict[str, str] = {}
    for rel in all_notes:
        content = contents.get(rel, "")
        for alias in extract_aliases(content):
            amap[alias.lower()] = rel
    return amap


def _find_link_line(lines: list[str], link_target: str) -> tuple[int, str]:
    """Find the line number (1-indexed) and content of a wikilink."""
    target_lower = link_target.lower()
    for i, line in enumerate(lines):
        ll = line.lower()
        if f"[[{target_lower}" in ll or f"[[{target_lower}|" in ll:
            return i + 1, line.strip()
    # Basename fallback
    basename = target_lower.split("/")[-1]
    for i, line in enumerate(lines):
        if f"[[{basename}" in line.lower():
            return i + 1, line.strip()
    return 0, ""


def get_backlinks(vault_path: str, path: str) -> list[dict]:
    """Return list of {source, line, context} for notes linking to path."""
    all_notes, contents = _load_vault(vault_path)
    alias_map = _build_alias_map(all_notes, contents)
    target = resolve_note_path(path, all_notes)
    if target is None:
        return []

    results = []
    for note in all_notes:
        if note == target:
            continue
        content = contents.get(note, "")
        links = extract_wikilinks(content)
        lines = content.split("\n")
        for link in links:
            target_base = link["target"].split("#")[0].strip()
            resolved = resolve_wikilink(target_base, note, all_notes, alias_map)
            if resolved == target:
                line_num, ctx = _find_link_line(lines, link["target"])
                results.append({"source": note, "line": line_num, "context": ctx})

    # Deduplicate by source+line
    seen: set[str] = set()
    deduped = []
    for r in results:
        key = f"{r['source']}:{r['line']}"
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped


def get_outlinks(vault_path: str, path: str) -> dict:
    """Return {valid, broken, embeds} lists for outgoing links from a note."""
    all_notes, contents = _load_vault(vault_path)
    alias_map = _build_alias_map(all_notes, contents)

    target = resolve_note_path(path, all_notes)
    if target is None:
        raise FileNotFoundError(f"Note not found: {path}")

    content = contents.get(target, "")
    links = extract_wikilinks(content)

    valid = []
    broken = []
    embeds = []

    for link in links:
        target_base = link["target"].split("#")[0].strip()
        if not target_base:
            continue
        resolved = resolve_wikilink(target_base, target, all_notes, alias_map)
        entry = {
            "target": link["target"],
            "resolved_path": resolved,
            "is_valid": resolved is not None,
            "is_embed": link["is_embed"],
        }
        if link["is_embed"]:
            embeds.append(entry)
        elif resolved:
            valid.append(entry)
        else:
            broken.append(entry)

    return {"valid": valid, "broken": broken, "embeds": embeds}


def find_orphans(
    vault_path: str,
    include_outlinks_check: bool = True,
    max_results: int = 200,
) -> dict:
    """Find notes with no connections."""
    all_notes, contents = _load_vault(vault_path)
    alias_map = _build_alias_map(all_notes, contents)

    outlinks_map: dict[str, set[str]] = {n: set() for n in all_notes}
    backlinks_map: dict[str, set[str]] = {n: set() for n in all_notes}

    for note in all_notes:
        content = contents.get(note, "")
        for link in extract_wikilinks(content):
            target_base = link["target"].split("#")[0].strip()
            if not target_base:
                continue
            resolved = resolve_wikilink(target_base, note, all_notes, alias_map)
            if resolved:
                outlinks_map[note].add(resolved)
                backlinks_map.setdefault(resolved, set()).add(note)

    fully_isolated = []
    no_backlinks = []
    no_outlinks = []

    for note in all_notes:
        has_back = bool(backlinks_map.get(note))
        has_out = bool(outlinks_map.get(note))
        if not has_back and not has_out:
            fully_isolated.append(note)
        elif not has_back:
            no_backlinks.append(note)
        elif not has_out and include_outlinks_check:
            no_outlinks.append(note)

    remaining = max_results
    capped_iso = fully_isolated[:remaining]
    remaining -= len(capped_iso)
    capped_nob = no_backlinks[:max(0, remaining)]
    remaining -= len(capped_nob)
    capped_noo = (no_outlinks[:max(0, remaining)] if include_outlinks_check else [])

    return {
        "fully_isolated": capped_iso,
        "fully_isolated_total": len(fully_isolated),
        "no_backlinks": capped_nob,
        "no_backlinks_total": len(no_backlinks),
        "no_outlinks": capped_noo if include_outlinks_check else [],
        "no_outlinks_total": len(no_outlinks) if include_outlinks_check else 0,
        "total_notes": len(all_notes),
    }


def find_broken_links(
    vault_path: str,
    folder: str | None = None,
    max_results: int = 200,
) -> list[dict]:
    """Find wikilinks pointing at non-existent notes."""
    all_notes, contents = _load_vault(vault_path)
    alias_map = _build_alias_map(all_notes, contents)

    scan_notes = all_notes
    if folder:
        scan_notes = [n for n in all_notes if n.startswith(folder.rstrip("/") + "/") or n == folder]

    broken: list[dict] = []
    for note in scan_notes:
        if len(broken) >= max_results:
            break
        content = contents.get(note, "")
        lines = content.split("\n")
        for link in extract_wikilinks(content):
            target_base = link["target"].split("#")[0].strip()
            if not target_base:
                continue
            resolved = resolve_wikilink(target_base, note, all_notes, alias_map)
            if not resolved:
                line_num, _ = _find_link_line(lines, link["target"])
                broken.append({
                    "source_path": note,
                    "target_link": link["target"],
                    "line": line_num,
                })

    return broken


def get_graph_neighbors(
    vault_path: str,
    path: str,
    depth: int = 1,
    direction: str = "both",
) -> list[dict]:
    """BFS traversal of the wikilink graph from a starting note."""
    all_notes, contents = _load_vault(vault_path)
    alias_map = _build_alias_map(all_notes, contents)

    start = resolve_note_path(path, all_notes)
    if start is None:
        raise FileNotFoundError(f"Note not found: {path}")

    # Build outlinks and backlinks maps
    outlinks: dict[str, set[str]] = {n: set() for n in all_notes}
    backlinks: dict[str, set[str]] = {n: set() for n in all_notes}
    for note in all_notes:
        content = contents.get(note, "")
        for link in extract_wikilinks(content):
            target_base = link["target"].split("#")[0].strip()
            if not target_base:
                continue
            resolved = resolve_wikilink(target_base, note, all_notes, alias_map)
            if resolved:
                outlinks[note].add(resolved)
                backlinks.setdefault(resolved, set()).add(note)

    # BFS
    visited: dict[str, dict] = {start: {"path": start, "depth": 0, "direction": "both"}}
    queue = [(start, 0)]

    while queue:
        current, cur_depth = queue.pop(0)
        if cur_depth >= depth:
            continue
        neighbors = []
        if direction in ("outbound", "both"):
            for t in outlinks.get(current, set()):
                neighbors.append((t, "outbound"))
        if direction in ("inbound", "both"):
            for s in backlinks.get(current, set()):
                neighbors.append((s, "inbound"))
        for neighbor_path, dir_ in neighbors:
            if neighbor_path not in visited:
                visited[neighbor_path] = {
                    "path": neighbor_path,
                    "depth": cur_depth + 1,
                    "direction": dir_,
                }
                queue.append((neighbor_path, cur_depth + 1))

    # Remove start from results
    del visited[start]

    return sorted(visited.values(), key=lambda x: (x["depth"], x["path"]))
