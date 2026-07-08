"""Regression tests for LanceDB filter escaping (plan fable-fixes, Phase 0.1 / F1/O2).

Two failure modes the old backslash-escaper had:
  - `=` / delete path: underscore/percent filenames matched 0 rows, so old chunks
    were never deleted on reindex → duplicate/stale chunks accumulated forever.
    A double-quote filename crashed the whole delete batch (tokenizer error).
  - LIKE path: `_`/`%` in a filter value acted as wildcards (wrong-token matches).
"""

import os

import lancedb
import pytest

from some_vault_some_mcp.core.embeddings import MockProvider
from some_vault_some_mcp.core.filters import escape_like, escape_string
from some_vault_some_mcp.core.indexer import (
    _get_db,
    _get_table,
    full_index,
    incremental_index,
)

pytestmark = pytest.mark.integration

HOSTILE = ["my_note.md", "pct%.md", 'has"quote.md', "plain.md"]


def _write(vault, name, body, mtime):
    p = vault / name
    p.write_text(body, encoding="utf-8")
    os.utime(str(p), (mtime, mtime))


def test_reindex_with_hostile_filenames_no_stale_chunks(tmp_path):
    """Editing underscore/percent/quote-named notes must delete their old chunks
    (the `=` delete path), leaving exactly one chunk-set of the latest content."""
    vault = tmp_path / "vault"
    vault.mkdir()
    for i, name in enumerate(HOSTILE):
        _write(vault, name, "version1", 1_000_000 + i)

    db_path = str(tmp_path / "db.lance")
    provider = MockProvider()
    full_index(str(vault), db_path, provider)

    # Edit every file twice — exercises the delete filter twice per file.
    for rev, tag in ((2, "version2"), (3, "version3")):
        for i, name in enumerate(HOSTILE):
            _write(vault, name, tag, 1_000_000 + i + rev * 1000)
        incremental_index(str(vault), db_path, provider)

    table = _get_table(_get_db(db_path))
    df = table.to_pandas()

    # No stale content survives.
    assert not df["content"].str.contains("version1").any()
    assert not df["content"].str.contains("version2").any()
    # Every file present, exactly one chunk each (no accumulation).
    for name in HOSTILE:
        rows = df[df["file_path"] == name]
        assert len(rows) == 1, f"{name}: expected 1 chunk, got {len(rows)} (stale duplicates)"
        assert "version3" in rows.iloc[0]["content"]


def test_like_escaping_matches_literal_wildcards(tmp_path):
    """A filter value containing literal `_`/`%` must match only itself, not act
    as a SQL wildcard (the LIKE path the delete test does not exercise)."""
    db = lancedb.connect(str(tmp_path / "t.lance"))
    rows = [{"tag": t, "v": [0.0]} for t in ["a_b", "axb", "10%", "10x", 'q"x']]
    t = db.create_table("t", data=rows)

    def like(val):
        w = f'tag LIKE "%{escape_like(val)}%"'
        return set(t.search().where(w).select(["tag"]).to_pandas()["tag"])

    assert like("a_b") == {"a_b"}      # underscore is literal, not wildcard
    assert like("10%") == {"10%"}      # percent is literal
    assert like('q"x') == {'q"x'}      # double-quote survives

    # exact `=` with escape_string
    w = f'tag = "{escape_string(chr(34).join(["q", "x"]))}"'  # tag == 'q"x'
    assert set(t.search().where(w).select(["tag"]).to_pandas()["tag"]) == {'q"x'}
