"""Unit tests for the markdown chunker with breadcrumb support."""

import pytest

from some_vault_some_mcp.core.chunker import (
    chunk_markdown, _split_by_headings, _split_section, _split_paragraph,
    _build_metadata_header, TARGET_CHUNK_SIZE, OVERLAP_SIZE,
)


def test_basic_chunk():
    content = "---\ntitle: Test\n---\n\nSome body text."
    chunks = chunk_markdown("test.md", content, file_mtime=0.0)
    assert len(chunks) >= 1
    assert chunks[0]["file_path"] == "test.md"
    assert "Some body text." in chunks[0]["content"]


def test_no_frontmatter():
    content = "# Heading\n\nBody text."
    chunks = chunk_markdown("note.md", content)
    assert len(chunks) >= 1


def test_empty_body():
    content = "---\ntitle: Empty\n---\n"
    chunks = chunk_markdown("empty.md", content)
    assert chunks == []


def test_breadcrumb_nested_headings():
    content = (
        "# Design\n\nDesign intro.\n\n"
        "## Architecture\n\nArch content.\n\n"
        "### Implementation\n\nImpl content.\n"
    )
    sections = _split_by_headings(content)
    # Find the Implementation section
    impl = next((h for h, _ in sections if h and "Implementation" in h), None)
    assert impl is not None
    assert "Design" in impl
    assert "Architecture" in impl
    assert "Implementation" in impl
    # Verify breadcrumb format
    assert " > " in impl


def test_breadcrumb_in_chunk_heading():
    content = (
        "---\ntitle: BreadcrumbTest\n---\n\n"
        "# A\n\nA content.\n\n"
        "## B\n\nB content.\n\n"
        "### C\n\nC content.\n"
    )
    chunks = chunk_markdown("bc.md", content)
    # Find C's chunk
    c_chunk = next((c for c in chunks if c["heading"] and "C" in c["heading"]), None)
    assert c_chunk is not None
    assert "A" in c_chunk["heading"]
    assert "B" in c_chunk["heading"]
    assert "C" in c_chunk["heading"]


def test_breadcrumb_resets_on_higher_level():
    content = (
        "# A\n\nA content.\n\n"
        "## B\n\nB content.\n\n"
        "# C\n\nC content (new top-level, breadcrumb resets).\n"
    )
    sections = _split_by_headings(content)
    # C should not include A or B in its breadcrumb
    c_section = next((h for h, _ in sections if h and h == "C"), None)
    assert c_section is not None
    assert "A" not in c_section
    assert "B" not in c_section


def test_no_heading_chunk_empty_breadcrumb():
    content = "Body before any heading.\n"
    sections = _split_by_headings(content)
    assert sections[0][0] is None


def test_breadcrumb_in_section_field():
    content = (
        "---\ntitle: Meta\n---\n\n"
        "# Parent\n\nParent content.\n\n"
        "## Child\n\nChild content here.\n"
    )
    chunks = chunk_markdown("meta.md", content)
    child = next((c for c in chunks if "Child" in c.get("heading", "")), None)
    assert child is not None
    assert "Section: Parent > Child" in child["text_to_embed"]


def test_metadata_header_in_text_to_embed():
    content = (
        "---\ntitle: MyNote\ntags:\n  - foo\n---\n\n"
        "# Section\n\nContent.\n"
    )
    chunks = chunk_markdown("meta.md", content)
    assert chunks
    assert "Title: MyNote" in chunks[0]["text_to_embed"]
    assert "Tags: foo" in chunks[0]["text_to_embed"]


def test_chunk_index_sequential():
    content = (
        "---\ntitle: Seq\n---\n\n"
        "# A\n\nA.\n\n"
        "# B\n\nB.\n\n"
        "# C\n\nC.\n"
    )
    chunks = chunk_markdown("seq.md", content)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_large_section_splits():
    # Create a section with multiple paragraphs totalling > TARGET_CHUNK_SIZE (2000 chars)
    # Paragraph-based splitting requires double-newlines between paragraphs
    para = "word " * 200 + "\n"  # ~1000 chars per para
    long_body = (para + "\n") * 4  # 4 paragraphs, ~4000 chars total
    content = f"---\ntitle: Large\n---\n\n# Big Section\n\n{long_body}"
    chunks = chunk_markdown("large.md", content, file_mtime=0.0)
    assert len(chunks) > 1


def test_html_stripped():
    content = "---\ntitle: T\n---\n\n# S\n\n<b>Bold</b> text.\n"
    chunks = chunk_markdown("html.md", content)
    assert "<b>" not in chunks[0]["content"]
    assert "Bold" in chunks[0]["content"]


# --- Fix 1: paragraph sub-splitter ---

def test_split_paragraph_by_sentences():
    text = ". ".join(f"Sentence number {i} with some padding words" for i in range(80))
    assert len(text) > TARGET_CHUNK_SIZE
    parts = _split_paragraph(text, TARGET_CHUNK_SIZE)
    assert len(parts) > 1
    for part in parts:
        assert len(part) <= TARGET_CHUNK_SIZE


def test_split_paragraph_no_sentences():
    text = "x" * 5000
    parts = _split_paragraph(text, TARGET_CHUNK_SIZE)
    assert len(parts) > 1
    for part in parts:
        assert len(part) <= TARGET_CHUNK_SIZE


def test_split_paragraph_passthrough():
    text = "Short paragraph."
    assert _split_paragraph(text, TARGET_CHUNK_SIZE) == [text]


def test_oversized_single_paragraph_in_section():
    para = "word " * 1000  # ~5000 chars, single paragraph (no blank lines)
    content = f"---\ntitle: T\n---\n\n# H\n\n{para}"
    chunks = chunk_markdown("big.md", content, file_mtime=0.0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c["text_to_embed"]) <= TARGET_CHUNK_SIZE


# --- Fix 2: overlap bounded ---

def test_overlap_bounded():
    big_para = "A" * 1500
    small_para = "B" * 100
    text = f"{big_para}\n\n{small_para}\n\n{small_para}"
    parts = _split_section(text, target_size=TARGET_CHUNK_SIZE)
    for part in parts:
        assert big_para not in part or part == parts[0]


def test_overlap_small_paragraphs():
    paras = [f"Para {i}. " + "x" * 80 for i in range(30)]
    text = "\n\n".join(paras)
    parts = _split_section(text, target_size=TARGET_CHUNK_SIZE)
    assert len(parts) > 1
    for i in range(1, len(parts)):
        overlap = set(parts[i - 1].split("\n\n")) & set(parts[i].split("\n\n"))
        assert len(overlap) > 0


# --- Fix 3: header counted in budget ---

def test_header_counted_in_budget():
    tags = [f"tag{i}" for i in range(20)]
    tags_yaml = "\n".join(f"  - {t}" for t in tags)
    section_text = "w" * 1900
    content = (
        f"---\ntitle: A Very Long Title For Testing Purposes\n"
        f"tags:\n{tags_yaml}\n"
        f"projects:\n  - proj1\n  - proj2\n"
        f"area: '[[SomeArea]]'\nsource: '[[SomeSource]]'\n---\n\n"
        f"# Section\n\n{section_text}"
    )
    chunks = chunk_markdown("hdr.md", content, file_mtime=0.0)
    for c in chunks:
        assert len(c["text_to_embed"]) <= TARGET_CHUNK_SIZE


def test_text_to_embed_bounded():
    for size in [1800, 1900, 1950, 2000, 2500, 5000]:
        body = "w" * size
        content = (
            "---\ntitle: Note\ntags:\n  - a\n  - b\n  - c\n"
            "projects:\n  - p1\narea: Area\nsource: Source\n---\n\n"
            f"# Heading\n\n{body}"
        )
        chunks = chunk_markdown("bounded.md", content, file_mtime=0.0)
        for c in chunks:
            assert len(c["text_to_embed"]) <= TARGET_CHUNK_SIZE, (
                f"text_to_embed is {len(c['text_to_embed'])} chars for input size {size}"
            )


def test_strip_html_preserves_generics_in_fenced_code():
    """`List<int>` inside a code fence must survive (F17) — the HTML stripper
    used to eat `<int>`, making code unsearchable."""
    content = (
        "---\ntitle: Code\n---\n\n"
        "Prose with <b>bold</b> tag.\n\n"
        "```rust\nlet v: List<int> = default();\n```\n"
    )
    chunks = chunk_markdown("code.md", content)
    joined = "\n".join(c["content"] for c in chunks)
    assert "List<int>" in joined          # generics in code preserved
    assert "<b>" not in joined            # real HTML outside code still stripped
