"""Unit tests for frontmatter parser."""

from datetime import date, datetime

import pytest

from some_vault_some_mcp.core.frontmatter import (
    extract_all_tags,
    parse_frontmatter,
    serialize_frontmatter,
    update_frontmatter,
)
from some_vault_some_mcp.core.chunker import chunk_markdown


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


def test_parse_invalid_timestamp_keeps_body_indexable():
    content = "---\ndate: 2001-13-01\n---\nRetained body."

    fm, body = parse_frontmatter(content)
    chunks = chunk_markdown("bad-date.md", content)

    assert fm == {}
    assert body == "Retained body."
    assert [chunk["content"] for chunk in chunks] == ["Retained body."]


@pytest.mark.parametrize(
    "yaml_value",
    [
        "tags: &loop [*loop]",
        "a: &a\n  - *a",
        "a: &a\n  - &b\n    - *a",
    ],
)
def test_parse_recursive_yaml_alias_discards_metadata_but_keeps_body(yaml_value):
    fm, body = parse_frontmatter(f"---\n{yaml_value}\n---\n\nRetained body.")
    assert fm == {}
    assert body == "Retained body."


def test_parse_repeated_acyclic_alias_is_independently_copied():
    fm, _ = parse_frontmatter("---\na: &shared\n  - one\nb: *shared\n---\nBody.")
    assert fm == {"a": ["one"], "b": ["one"]}
    assert fm["a"] is not fm["b"]


def test_parse_rejects_acyclic_alias_scalar_amplification_but_indexes_body():
    scalar = "x" * 4096
    aliases = ", ".join(["*large"] * 20)
    content = f"---\nvalue: &large {scalar}\ntags: [{aliases}]\n---\nRetained body."

    fm, body = parse_frontmatter(content)
    chunks = chunk_markdown("amplified.md", content)

    assert fm == {}
    assert body == "Retained body."
    assert [chunk["text_to_embed"] for chunk in chunks] == ["Retained body."]


def _nested_list_frontmatter(depth: int) -> str:
    """Produce a document whose scalar leaf is at the requested graph depth."""
    lines = ["---", "value:"]
    for level in range(1, depth):
        lines.append("  " * level + "-")
    lines.append("  " * depth + "leaf")
    lines.extend(["---", "Body."])
    return "\n".join(lines)


def test_parse_accepts_frontmatter_at_exact_depth_limit():
    fm, body = parse_frontmatter(_nested_list_frontmatter(32))
    assert fm
    assert body == "Body."


def test_parse_rejects_frontmatter_past_depth_limit_and_keeps_body():
    fm, body = parse_frontmatter(_nested_list_frontmatter(33))
    assert fm == {}
    assert body == "Body."


@pytest.mark.parametrize("extra_scalars, expected", [(497, True), (498, False)])
def test_parse_enforces_exact_node_limit(extra_scalars, expected):
    # Root mapping + key + outer list + 500 expansions of a 19-node list,
    # followed by 497 (or 498) scalar list elements. Flow YAML stays below the
    # independent 500-line scanner limit.
    shared = "&shared [" + ", ".join("x" for _ in range(18)) + "]"
    items = [shared] + ["*shared"] * 499 + ["x"] * extra_scalars
    raw = "---\nitems: [" + ", ".join(items) + "]\n---\nBody."
    fm, body = parse_frontmatter(raw)
    assert bool(fm) is expected
    assert body == "Body."


@pytest.mark.parametrize(
    "yaml_value",
    [
        "1: non-string-key",
        "value: !!set {a: null}",
        "value: !!binary YQ==",
    ],
)
def test_parse_rejects_non_string_keys_and_unsupported_safe_yaml_types(yaml_value):
    fm, body = parse_frontmatter(f"---\n{yaml_value}\n---\nBody.")
    assert fm == {}
    assert body == "Body."


def test_parse_accepts_all_supported_scalar_types():
    fm, _ = parse_frontmatter(
        "---\nnull_value: null\nbool_value: true\nint_value: 42\n"
        "float_value: 1.5\nstr_value: hello\ndate_value: 2024-05-01\n"
        "datetime_value: 2024-05-01T12:34:56Z\n---\nBody."
    )
    assert fm["null_value"] is None
    assert fm["bool_value"] is True
    assert fm["int_value"] == 42
    assert fm["float_value"] == 1.5
    assert fm["str_value"] == "hello"
    assert isinstance(fm["date_value"], date)
    assert not isinstance(fm["date_value"], datetime)
    assert isinstance(fm["datetime_value"], datetime)


def test_parse_recursion_error_discards_metadata_but_keeps_body(monkeypatch):
    monkeypatch.setattr(
        "some_vault_some_mcp.core.frontmatter.yaml.safe_load",
        lambda _: (_ for _ in ()).throw(RecursionError("parser recursion")),
    )
    fm, body = parse_frontmatter("---\ntitle: Test\n---\nBody.")
    assert fm == {}
    assert body == "Body."


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
