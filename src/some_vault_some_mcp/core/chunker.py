"""Markdown-aware chunker with heading breadcrumb support.

Port of willfanguy's chunker.py, enhanced with heading depth tracking for
breadcrumb paths (e.g. "Design > Architecture > Implementation") stored in
the heading column and used in the Section: field of the embedding metadata
header.

Chunking strategy:
- split by headings first (preserving hierarchy)
- sections > TARGET_CHUNK_SIZE split by paragraphs with OVERLAP_SIZE overlap
- heading breadcrumb replaces flat heading in Section: metadata header
"""

import re
from datetime import datetime
from html.parser import HTMLParser
import logging

logger = logging.getLogger(__name__)

TARGET_CHUNK_SIZE = 2000  # chars, ~512 tokens
OVERLAP_SIZE = 200         # chars overlap between adjacent paragraph chunks
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def strip_html(text: str) -> str:
    if "<" not in text:
        return text
    ex = _HTMLTextExtractor()
    try:
        ex.feed(text)
        return ex.get_text()
    except Exception:
        return re.sub(r"<[^>]+>", "", text)


def _extract_metadata_fields(fm: dict) -> dict:
    def clean_wikilinks(val):
        if isinstance(val, str):
            return re.sub(r"\[\[([^\]]+)\]\]", r"\1", val)
        if isinstance(val, list):
            return [clean_wikilinks(v) for v in val]
        return val

    def safe_str(val):
        return "" if val is None else str(val)

    return {
        "title": safe_str(fm.get("title", "")),
        "tags": clean_wikilinks(fm.get("tags", [])) or [],
        "projects": clean_wikilinks(fm.get("projects", [])) or [],
        "status": safe_str(fm.get("status")),
        "area": safe_str(clean_wikilinks(fm.get("area", ""))),
        "source": safe_str(clean_wikilinks(fm.get("source", ""))),
    }


def _build_metadata_header(metadata: dict, heading: str | None) -> str:
    """Build bracketed metadata header prepended to chunk text for embedding."""
    parts = []
    if metadata.get("title"):
        parts.append(f"Title: {metadata['title']}")
    if heading:
        parts.append(f"Section: {heading}")
    tags = metadata.get("tags", [])
    if tags and isinstance(tags, list):
        parts.append(f"Tags: {', '.join(str(t) for t in tags if t)}")
    projects = metadata.get("projects", [])
    if projects and isinstance(projects, list):
        parts.append(f"Projects: {', '.join(str(p) for p in projects if p)}")
    if metadata.get("area"):
        parts.append(f"Area: {metadata['area']}")
    if metadata.get("source"):
        parts.append(f"Source: {metadata['source']}")
    if metadata.get("date"):
        parts.append(f"Date: {metadata['date']}")
    return f"[{' | '.join(parts)}]" if parts else ""


def _split_by_headings(body: str) -> list[tuple[str | None, str]]:
    """Split body into (breadcrumb, content) pairs.

    Tracks heading nesting to build breadcrumb path:
    # A / ## B / ### C → "A > B > C"
    Depth resets when a higher-level heading appears.
    No-heading chunks get None breadcrumb.
    """
    matches = list(HEADING_PATTERN.finditer(body))
    if not matches:
        return [(None, body)]

    sections: list[tuple[str | None, str]] = []

    # Content before first heading
    if matches[0].start() > 0:
        pre = body[:matches[0].start()]
        if pre.strip():
            sections.append((None, pre))

    # Track breadcrumb stack: list of (level, heading_text)
    stack: list[tuple[int, str]] = []

    for i, match in enumerate(matches):
        level = len(match.group(1))  # number of # chars
        heading_text = match.group(2).strip()

        # Pop stack entries with level >= current (reset deeper / same-level)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading_text))

        # Build breadcrumb from stack
        breadcrumb = " > ".join(h for _, h in stack)

        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append((breadcrumb, body[content_start:content_end]))

    return sections


def _split_paragraph(text: str, target_size: int) -> list[str]:
    """Split an oversized paragraph into sub-chunks.

    Tries sentence boundaries first, falls back to character windows.
    """
    if len(text) <= target_size:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        groups: list[str] = []
        current_parts: list[str] = []
        current_len = 0
        for sentence in sentences:
            if current_len + len(sentence) > target_size and current_parts:
                groups.append(" ".join(current_parts))
                current_parts = [sentence]
                current_len = len(sentence)
            else:
                current_parts.append(sentence)
                current_len += len(sentence) + 1
        if current_parts:
            groups.append(" ".join(current_parts))
        if all(len(g) <= target_size * 1.5 for g in groups):
            return groups

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + target_size, len(text))
        if end < len(text):
            word_break = text.rfind(" ", max(start, end - 200), end)
            if word_break > start:
                end = word_break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - OVERLAP_SIZE if end < len(text) else end
    return chunks


def _split_section(text: str, target_size: int = TARGET_CHUNK_SIZE) -> list[str]:
    """Split a large section by paragraphs with OVERLAP_SIZE overlap."""
    paragraphs = re.split(r"\n\s*\n", text)

    expanded: list[str] = []
    for para in paragraphs:
        if len(para) > target_size:
            expanded.extend(_split_paragraph(para, target_size))
        else:
            expanded.append(para)
    paragraphs = expanded

    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > target_size and current:
            groups.append(current)
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 2

    if current:
        groups.append(current)

    if len(groups) <= 1:
        return ["\n\n".join(g) for g in groups]

    chunks = []
    for i, group in enumerate(groups):
        if i == 0:
            chunks.append("\n\n".join(group))
        else:
            prev = groups[i - 1]
            overlap_paras = []
            overlap_len = 0
            for para in reversed(prev):
                if overlap_len + len(para) > OVERLAP_SIZE:
                    break
                overlap_paras.insert(0, para)
                overlap_len += len(para) + 2
            chunks.append("\n\n".join(overlap_paras + group))

    return chunks


def _make_chunk(
    file_path: str,
    chunk_index: int,
    heading: str | None,
    content: str,
    metadata: dict,
) -> dict:
    header = _build_metadata_header(metadata, heading)
    return {
        "file_path": file_path,
        "chunk_index": chunk_index,
        "heading": heading or "",
        "content": content,
        "text_to_embed": f"{header}\n\n{content}" if header else content,
        "title": metadata.get("title", ""),
        "tags": metadata.get("tags", []),
        "projects": metadata.get("projects", []),
        "area": metadata.get("area", ""),
        "status": metadata.get("status", ""),
        "source": metadata.get("source", ""),
        # file_mtime is injected by indexer after chunking
    }


def chunk_markdown(file_path: str, content: str, file_mtime: float | None = None) -> list[dict]:
    """Split a markdown file into indexable chunks.

    Returns list of chunk dicts. The indexer injects file_mtime after calling
    this function.
    """
    from some_vault_some_mcp.core.frontmatter import parse_frontmatter
    metadata_raw, body = parse_frontmatter(content)
    fields = _extract_metadata_fields(metadata_raw)

    # Extract date for embedding context
    date_str = ""
    for key in ("date created", "dateCreated", "date", "created"):
        val = metadata_raw.get(key)
        if val:
            if isinstance(val, list):  # list-valued date field → take first entry
                val = val[0] if val else ""
            date_str = str(val).split(",")[0].split("T")[0].strip()
            break
    if not date_str and file_mtime:
        date_str = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")
    fields["date"] = date_str

    if not body.strip():
        return []

    # Strip HTML only outside fenced code — otherwise `List<int>` and other
    # `<word` sequences in code get eaten and become unsearchable (F17).
    from some_vault_some_mcp.core.markdown import iter_lines_with_fence_state
    body = "\n".join(
        line if in_code else strip_html(line)
        for line, in_code in iter_lines_with_fence_state(body)
    )
    sections = _split_by_headings(body)
    chunks: list[dict] = []

    for heading, section_text in sections:
        if not section_text.strip():
            continue
        header = _build_metadata_header(fields, heading)
        header_len = len(header) + 2 if header else 0
        effective_budget = max(TARGET_CHUNK_SIZE - header_len, TARGET_CHUNK_SIZE // 2)
        if len(section_text) <= effective_budget:
            chunks.append(_make_chunk(
                file_path=file_path,
                chunk_index=len(chunks),
                heading=heading,
                content=section_text.strip(),
                metadata=fields,
            ))
        else:
            for sub in _split_section(section_text, target_size=effective_budget):
                if sub.strip():
                    chunks.append(_make_chunk(
                        file_path=file_path,
                        chunk_index=len(chunks),
                        heading=heading,
                        content=sub.strip(),
                        metadata=fields,
                    ))

    return chunks
