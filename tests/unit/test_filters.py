"""Unit tests for LanceDB filter helpers (plan Phase 0.1 / Phase 1 F3)."""

from some_vault_some_mcp.core.filters import (
    escape_like,
    escape_string,
    like_token,
    split_tokens,
    store_tokens,
)


def test_escape_string_doubles_quotes():
    assert escape_string('a"b') == 'a""b'


def test_escape_like_escapes_wildcards_and_quotes():
    assert escape_like("a_b") == "a\\_b"
    assert escape_like("10%") == "10\\%"
    assert escape_like('q"x') == 'q""x'


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
    assert cond == 'tags LIKE "%,art,%"'
