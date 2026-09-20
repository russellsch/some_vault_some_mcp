"""Unit tests for LanceDB filter helpers (plan Phase 0.1 / Phase 1 F3)."""

from some_vault_some_mcp.core.filters import (
    escape_like,
    escape_string,
    like_token,
    split_tokens,
    store_tokens,
)


def test_escape_string_doubles_single_quotes():
    assert escape_string("it's") == "it''s"
    # A double quote is not special inside a single-quoted literal.
    assert escape_string('a"b') == 'a"b'
    # A backslash is a plain character in an `=` literal.
    assert escape_string("a\\b") == "a\\b"


def test_escape_like_escapes_wildcards_and_quotes():
    assert escape_like("a_b") == "a\\_b"
    assert escape_like("10%") == "10\\%"
    assert escape_like("it's") == "it''s"
    assert escape_like('q"x') == 'q"x'
    assert escape_like("a\\b") == "a\\\\b"
    assert escape_like("100%'s") == "100\\%''s"


def test_store_tokens_lowercases_wraps_sentinels():
    assert store_tokens(["Art", "Ideas"]) == ",art,ideas,"
    assert store_tokens([]) == ""
    assert store_tokens(["  Spaced  "]) == ",spaced,"
    assert store_tokens("single") == ",single,"


def test_split_tokens_roundtrips_store():
    assert split_tokens(store_tokens(["a", "b"])) == ["a", "b"]
    assert split_tokens("") == []
    assert split_tokens(",art,") == ["art"]


def test_like_token_matches_whole_token_only():
    # ",art," can never appear inside ",smart," or ",cart," — that's the point.
    cond = like_token("tags", "Art")
    assert cond == "tags LIKE '%,art,%'"


def test_no_double_quoted_sql_literals_in_source():
    """DataFusion treats a double-quoted token as a column identifier. The locked
    LanceDB still accepts it through a legacy fallback, so this static check is
    the only guard on the locked version against a regression at any filter
    site. See filters.py module docstring."""
    import re
    from pathlib import Path

    src = Path(__file__).parent.parent.parent / "src" / "some_vault_some_mcp"
    files = [
        src / "core" / "filters.py",
        src / "core" / "indexer.py",
        src / "tools" / "read.py",
        src / "tools" / "search.py",
    ]
    pattern = re.compile(r'\b(file_path|status|area|tags|projects)\s*=\s*"|LIKE\s*"')
    fstring = re.compile(r"""\bf["']""")
    for f in files:
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            # Filters are built in f-strings; docstrings may quote the bad form.
            if not fstring.search(line):
                continue
            assert not pattern.search(line), f"double-quoted SQL literal at {f.name}:{lineno}: {line.strip()}"
