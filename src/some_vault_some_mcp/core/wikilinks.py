"""Obsidian-compatible wikilink resolver and extractor.

Resolution order matches Obsidian:
  1. Exact relative-path match (case-insensitive, with or without .md)
  2. Path-suffix match (case-insensitive)
  3. Basename match — proximity tie-break (deepest shared prefix with source note)
  4. Alias match (frontmatter aliases on any note)

Returns None if no match found.
"""

import re
from pathlib import PurePosixPath


# Matches [[target]] and ![[target]], handling display text (|) and heading refs (#)
_WIKILINK_RE = re.compile(r"(!)?(?<!\[)\[\[([^\]\n]+?)\]\](?!\])")


def extract_wikilinks(content: str) -> list[dict]:
    """Extract all wikilinks from markdown content.

    Returns list of dicts: {target, display_text, is_embed, source}.
    Skips links inside fenced code blocks and inline code.
    source is left empty — caller fills in.
    """
    from some_vault_some_mcp.core.markdown import iter_lines_with_fence_state
    links = []

    for line, in_code in iter_lines_with_fence_state(content):
        if in_code:
            continue

        # Strip inline code spans
        cleaned = re.sub(r"`[^`\n]*`", " " * 2, line)

        for m in _WIKILINK_RE.finditer(cleaned):
            is_embed = m.group(1) == "!"
            inner = m.group(2)
            pipe = inner.find("|")
            if pipe >= 0:
                target_raw = inner[:pipe]
                display = inner[pipe + 1:].strip()
            else:
                target_raw = inner
                display = None
            target = target_raw.strip()
            links.append({
                "target": target,
                "display_text": display,
                "is_embed": is_embed,
                "source": "",
            })
    return links


def _shared_depth(a: str, b: str) -> int:
    """Count leading path segments shared between two paths."""
    parts_a = a.lower().split("/")
    parts_b = b.lower().split("/")
    depth = 0
    for x, y in zip(parts_a, parts_b):
        if x == y:
            depth += 1
        else:
            break
    return depth


def resolve_wikilink(
    link: str,
    current_note_path: str,
    all_note_paths: list[str],
    alias_map: dict[str, str] | None = None,
) -> str | None:
    """Resolve a wikilink target to an actual vault-relative file path.

    link: raw target text (may include heading ref after #, display after |
          — callers should strip those before calling this).
    current_note_path: vault-relative path of the note containing the link.
    all_note_paths: list of all vault-relative .md paths.
    alias_map: optional {alias_lower: note_path} dict for alias resolution.
    """
    # Strip heading anchors and block refs
    clean = link.split("#")[0].split("^")[0].strip()
    if not clean:
        return None

    # Strip .md extension for comparison
    norm = clean
    if norm.lower().endswith(".md"):
        norm = norm[:-3]
    norm_lower = norm.lower()

    # 1. Exact relative-path match
    for p in all_note_paths:
        p_norm = p[:-3] if p.lower().endswith(".md") else p
        if p_norm.lower() == norm_lower:
            return p

    # 2. Path-suffix match (only when link contains /)
    if "/" in norm:
        for p in all_note_paths:
            p_norm = p[:-3] if p.lower().endswith(".md") else p
            if p_norm.lower().endswith("/" + norm_lower):
                return p

    # 3. Basename match with proximity tie-break
    link_basename = PurePosixPath(norm).name.lower()
    candidates = []
    for p in all_note_paths:
        p_stem = PurePosixPath(p).stem.lower()
        if p_stem == link_basename:
            candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        src_dir = str(PurePosixPath(current_note_path).parent)
        candidates.sort(key=lambda c: (
            -_shared_depth(src_dir, str(PurePosixPath(c).parent)),
            len(c),
        ))
        return candidates[0]

    # 4. Alias match
    if alias_map:
        hit = alias_map.get(norm_lower)
        if hit:
            return hit

    return None


def build_alias_map(
    note_paths: list[str],
    note_contents: dict[str, str],
) -> dict[str, str]:
    """Build {alias_lower: note_path} from all notes' frontmatter aliases."""
    from some_vault_some_mcp.core.frontmatter import extract_aliases
    alias_map: dict[str, str] = {}
    for path in note_paths:
        content = note_contents.get(path, "")
        for alias in extract_aliases(content):
            key = alias.lower()
            if key:
                alias_map[key] = path
    return alias_map
