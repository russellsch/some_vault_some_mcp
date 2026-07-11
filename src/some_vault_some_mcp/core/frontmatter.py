"""YAML frontmatter reader/writer — gray-matter equivalent for Python.

Parses the leading ---/--- YAML block and returns (metadata dict, body str).
Round-trips cleanly: parsed metadata can be merged and re-serialized without
losing the body.
"""

import io
import re
from typing import Any

import yaml
from ruamel.yaml import YAML


_FENCE_RE = re.compile(r"^---\r?\n", re.MULTILINE)
MAX_FRONTMATTER_LINES = 500
MAX_FRONTMATTER_BYTES = 64 * 1024

# Round-trip YAML for the WRITE path only — preserves comments, key order, and
# style. The read path (parse_frontmatter and its consumers) stays on PyYAML and
# returns plain dicts; ruamel's CommentedMap must not leak out of this module.
_rt_yaml = YAML()
_rt_yaml.preserve_quotes = True
_rt_yaml.width = 4096  # avoid line-wrapping long scalars


def _rt_dump(data: Any) -> str:
    buf = io.StringIO()
    _rt_yaml.dump(data, buf)
    return buf.getvalue()


def _scan_frontmatter(content: str) -> tuple[str | None, str]:
    """Locate the frontmatter block without parsing its YAML.

    Returns (raw_yaml_text, body). raw_yaml_text is None when there is no valid
    frontmatter, in which case body is the original content unchanged.
    """
    if not content.startswith("---"):
        return None, content
    first_newline = content.find("\n")
    if first_newline == -1:
        return None, content
    if content[:first_newline].rstrip("\r") != "---":
        return None, content
    offset = first_newline + 1
    lines = 0
    while offset < len(content) and offset < MAX_FRONTMATTER_BYTES:
        if lines >= MAX_FRONTMATTER_LINES:
            return None, content
        next_newline = content.find("\n", offset)
        line_end = next_newline if next_newline != -1 else len(content)
        line = content[offset:line_end].rstrip("\r")
        if line == "---":
            raw = content[first_newline + 1:offset]
            body_start = line_end + 1 if next_newline != -1 else line_end
            body = content[body_start:].lstrip("\n")
            return raw, body
        if next_newline == -1:
            return None, content
        offset = next_newline + 1
        lines += 1
    return None, content


def split_frontmatter(content: str) -> tuple[str | None, str]:
    """Public accessor for the raw frontmatter text + body (no YAML parsing).

    Used by prepend_to_note to keep the existing frontmatter block byte-for-byte.
    """
    return _scan_frontmatter(content)


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Return (metadata, body) from markdown content.

    Body is the content after the closing --- delimiter (stripped of leading
    whitespace). If no valid frontmatter, returns ({}, content).
    Malformed YAML returns ({}, body) — a single broken note must not abort
    vault-wide loops. Always returns a plain dict (PyYAML), never ruamel types.
    """
    raw, body = _scan_frontmatter(content)
    if raw is None:
        return {}, body
    try:
        metadata = yaml.safe_load(raw.strip()) or {}
        if not isinstance(metadata, dict):
            metadata = {}
    except yaml.YAMLError:
        metadata = {}
    return metadata, body


def serialize_frontmatter(metadata: dict[str, Any], body: str) -> str:
    """Serialize metadata as YAML frontmatter prepended to body.

    Uses ruamel dump so insertion order is preserved (no alphabetizing). For a
    fresh dict there are no comments to keep; update_frontmatter handles the
    comment-preserving round-trip of existing blocks.
    """
    return f"---\n{_rt_dump(dict(metadata))}---\n{body}"


def update_frontmatter(content: str, updates: dict[str, Any]) -> str:
    """Merge updates into existing frontmatter (or create one) and return new content.

    Round-trips the *raw* frontmatter text through ruamel so comments, key order,
    and flow/quote style survive. Keys in updates overwrite existing values; all
    other keys and their comments are preserved. Body content is unchanged.
    """
    raw, body = _scan_frontmatter(content)
    if raw is None:
        # No existing frontmatter — create a fresh block from updates.
        return f"---\n{_rt_dump(dict(updates))}---\n{body}"
    try:
        data = _rt_yaml.load(raw)
    except Exception:
        data = None
    if not isinstance(data, dict):  # CommentedMap is a dict subclass
        data = {}
    for key, value in updates.items():
        data[key] = value
    return f"---\n{_rt_dump(data)}---\n{body}"


def extract_inline_tags(content: str) -> list[str]:
    """Extract #hashtags from the body (not frontmatter) of a note.

    Skips lines inside code blocks. Returns list without # prefix.
    """
    from some_vault_some_mcp.core.markdown import iter_lines_with_fence_state
    _, body = parse_frontmatter(content)
    tags: list[str] = []
    tag_re = re.compile(
        r"(?:^|\s)#([a-zA-ZÀ-ɏЀ-ӿ_][a-zA-Z0-9À-ɏЀ-ӿ_/-]*)"
    )
    for line, in_code in iter_lines_with_fence_state(body):
        if in_code:
            continue
        # Skip headings
        if re.match(r"^\s*#{1,6}\s", line):
            continue
        for m in tag_re.finditer(line):
            tags.append(m.group(1))
    return tags


def extract_all_tags(content: str) -> list[str]:
    """Extract tags from both frontmatter and inline body text.

    Returns deduplicated list, lowercase, no # prefix.
    """
    tag_set: set[str] = set()
    metadata, _ = parse_frontmatter(content)
    # Frontmatter tags — check common casings
    fm_tags = (
        metadata.get("tags")
        or metadata.get("Tags")
        or metadata.get("TAGS")
        or metadata.get("tag")
        or metadata.get("Tag")
        or []
    )
    if isinstance(fm_tags, list):
        for t in fm_tags:
            s = str(t).strip()
            if s:
                tag_set.add(s.lower())
    elif isinstance(fm_tags, str):
        for t in fm_tags.split(","):
            s = t.strip()
            if s:
                tag_set.add(s.lower())

    for t in extract_inline_tags(content):
        tag_set.add(t.lower())

    return list(tag_set)


def extract_aliases(content: str) -> list[str]:
    """Extract frontmatter aliases field."""
    metadata, _ = parse_frontmatter(content)
    aliases_raw = (
        metadata.get("aliases")
        or metadata.get("Aliases")
        or metadata.get("ALIASES")
        or metadata.get("alias")
        or metadata.get("Alias")
        or []
    )
    if isinstance(aliases_raw, list):
        return [str(a).strip() for a in aliases_raw if str(a).strip()]
    if isinstance(aliases_raw, str):
        return [a.strip() for a in aliases_raw.split(",") if a.strip()]
    return []
