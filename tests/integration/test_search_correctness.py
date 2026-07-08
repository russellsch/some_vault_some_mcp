"""Search-correctness regressions (plan Phase 1: F2 cosine/score-bounds, F3 tokens).

Drives the REAL search/list functions against a real MockProvider index — the
old test_search_scoring.py only exercised a local _score() copy.
"""

import pytest

from some_vault_some_mcp.core.embeddings import MockProvider
from some_vault_some_mcp.core.indexer import full_index
from some_vault_some_mcp.tools.read import list_notes
from some_vault_some_mcp.tools.search import hybrid_search, semantic_search

pytestmark = pytest.mark.integration

NOTES = {
    "art.md": "---\ntags: [art]\n---\n\nPainting and sculpture and drawing.",
    "smart.md": "---\ntags: [smart]\n---\n\nClever intelligent reasoning here.",
    "ideas.md": "---\ntags: [Ideas]\n---\n\nBrainstorm of new concepts.",
    "work.md": "---\narea: Work\n---\n\nOffice tasks and meetings.",
    "homework.md": "---\narea: Homework\n---\n\nSchool assignments due.",
}


@pytest.fixture()
def indexed(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    for name, body in NOTES.items():
        (vault / name).write_text(body, encoding="utf-8")
    db_path = str(tmp_path / "db.lance")
    full_index(str(vault), db_path, MockProvider())
    return str(vault), db_path


def test_tag_filter_matches_whole_token_not_substring(indexed):
    vault, db_path = indexed
    results, total = list_notes(vault, db_path=db_path, tags=["art"])
    paths = {r.file_path for r in results}
    assert "art.md" in paths
    assert "smart.md" not in paths  # substring-bleed regression (F3)


def test_tag_filter_is_case_insensitive(indexed):
    vault, db_path = indexed
    results, _ = list_notes(vault, db_path=db_path, tags=["ideas"])  # stored as "Ideas"
    assert "ideas.md" in {r.file_path for r in results}


def test_area_filter_exact_no_substring_bleed(indexed):
    vault, db_path = indexed
    results, _ = list_notes(vault, db_path=db_path, area="Work")
    paths = {r.file_path for r in results}
    assert "work.md" in paths
    assert "homework.md" not in paths  # "Work" must not match "Homework"


def test_semantic_scores_bounded_0_1(indexed):
    _, db_path = indexed
    results = semantic_search("assignments and school", db_path, MockProvider(), top_k=10)
    assert results
    assert all(0.0 <= r.score <= 1.0 for r in results), [r.score for r in results]


def test_hybrid_scores_bounded_0_1(indexed):
    _, db_path = indexed
    results = hybrid_search("clever painting ideas", db_path, MockProvider(), top_k=10)
    assert results
    assert all(0.0 <= r.score <= 1.0 for r in results), [r.score for r in results]
