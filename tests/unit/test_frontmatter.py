"""Unit tests for frontmatter parser."""

import pytest

from some_vault_some_mcp.core.frontmatter import (
    extract_all_tags,
    parse_frontmatter,
    serialize_frontmatter,
    update_frontmatter,
)


def test_parse_with_frontmatter():
    content = "---\ntitle: Test\ntags:\n  - foo\n---\n\nBody here."
    fm, body = parse_frontmatter(content)
    assert fm["title"] == "Test"
    assert "foo" in fm["tags"]
    assert "Body here." in body


def test_parse_no_frontmatter():
    content = "# Just a heading\n\nBody text."
    fm, body = parse_frontmatter(content)
    assert fm == {}
    assert "Just a heading" in body


def test_parse_malformed_yaml_returns_empty():
    content = "---\n: bad: yaml: [\n---\nBody."
    fm, body = parse_frontmatter(content)
    # Malformed YAML → empty dict, original content preserved
    assert isinstance(fm, dict)


def test_parse_missing_closing_delimiter():
    content = "---\ntitle: Unclosed\n"
    fm, body = parse_frontmatter(content)
    assert fm == {}


def test_roundtrip():
    original = "---\ntitle: Test\nstatus: done\n---\nBody content here."
    fm, body = parse_frontmatter(original)
    reassembled = serialize_frontmatter(fm, body)
    fm2, body2 = parse_frontmatter(reassembled)
    assert fm2["title"] == "Test"
    assert fm2["status"] == "done"
    assert body2 == body


def test_update_frontmatter_merges():
    content = "---\ntitle: Old Title\nstatus: draft\n---\nBody."
    updated = update_frontmatter(content, {"status": "done", "priority": 1})
    fm, _ = parse_frontmatter(updated)
    assert fm["title"] == "Old Title"  # preserved
    assert fm["status"] == "done"      # overwritten
    assert fm["priority"] == 1        # new key


def test_update_frontmatter_creates_block():
    content = "No frontmatter here."
    updated = update_frontmatter(content, {"status": "new"})
    fm, _ = parse_frontmatter(updated)
    assert fm["status"] == "new"


def test_extract_all_tags_frontmatter_and_inline():
    content = "---\ntags:\n  - reference\n  - project\n---\n\nBody #inline-tag #another"
    tags = extract_all_tags(content)
    assert "reference" in tags
    assert "project" in tags
    assert "inline-tag" in tags
    assert "another" in tags


def test_extract_tags_deduplicates():
    content = "---\ntags:\n  - foo\n---\n\n#foo"
    tags = extract_all_tags(content)
    assert tags.count("foo") == 1


def test_tags_case_normalized():
    content = "---\ntags:\n  - FOO\n  - Bar\n---\n"
    tags = extract_all_tags(content)
    assert "foo" in tags
    assert "bar" in tags


def test_update_frontmatter_preserves_comments_order_and_flow_style():
    content = "---\ntitle: Z\ntags: [b, a]\n# keep me\ncreated: 2024-05-01\n---\nBody."
    out = update_frontmatter(content, {"status": "done"})
    assert "# keep me" in out          # comment preserved
    assert "[b, a]" in out             # flow style preserved
    assert out.index("title") < out.index("created") < out.index("status")  # order + append
