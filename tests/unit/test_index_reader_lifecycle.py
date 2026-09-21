"""Index consumers retain their generation lease through materialization."""

from contextlib import contextmanager

import pandas as pd

from some_vault_some_mcp.tools import read, search


class _Query:
    def __init__(self, frame, lease_state):
        self._frame = frame
        self._lease_state = lease_state

    def metric(self, _name):
        return self

    def limit(self, _count):
        return self

    def where(self, _expression):
        return self

    def select(self, _columns):
        return self

    def to_pandas(self):
        assert self._lease_state["held"]
        return self._frame


class _Table:
    def __init__(self, frame, lease_state):
        self._frame = frame
        self._lease_state = lease_state

    def count_rows(self):
        assert self._lease_state["held"]
        return 1

    def search(self, *_args, **_kwargs):
        assert self._lease_state["held"]
        return _Query(self._frame, self._lease_state)


def _reader_for(table, lease_state):
    @contextmanager
    def reader(_db_path, _provider_dims=None):
        lease_state["held"] = True
        try:
            yield object(), table, "generation", False
        finally:
            lease_state["held"] = False

    return reader


def test_semantic_search_materializes_under_generation_lease(monkeypatch):
    state = {"held": False}
    frame = pd.DataFrame(
        [{
            "title": "Note",
            "file_path": "note.md",
            "heading": "",
            "content": "body",
            "_distance": 0.1,
            "tags": "",
            "projects": "",
            "area": "",
        }]
    )
    table = _Table(frame, state)
    monkeypatch.setattr(
        "some_vault_some_mcp.core.indexer.active_table_reader",
        _reader_for(table, state),
    )
    def embed_query(_self, _query):
        assert not state["held"]
        return [0.0]

    provider = type("Provider", (), {"embed_query": embed_query})()

    results = search.semantic_search("body", "db", provider)

    assert [result.file_path for result in results] == ["note.md"]
    assert not state["held"]


def test_hybrid_search_embeds_outside_and_materializes_under_lease(monkeypatch):
    state = {"held": False}
    frame = pd.DataFrame(
        [{
            "title": "Note",
            "file_path": "note.md",
            "chunk_index": 0,
            "heading": "",
            "content": "body",
            "_distance": 0.1,
            "_score": 1.0,
            "tags": "",
            "projects": "",
            "area": "",
        }]
    )
    table = _Table(frame, state)
    monkeypatch.setattr(
        "some_vault_some_mcp.core.indexer.active_table_reader",
        _reader_for(table, state),
    )

    def embed_query(_self, _query):
        assert not state["held"]
        return [0.0]

    provider = type("Provider", (), {"embed_query": embed_query})()

    results = search.hybrid_search("body", "db", provider)

    assert [result.file_path for result in results] == ["note.md"]
    assert not state["held"]


def test_index_filter_materializes_under_generation_lease(monkeypatch):
    state = {"held": False}
    table = _Table(pd.DataFrame([{"file_path": "note.md"}]), state)
    monkeypatch.setattr(
        "some_vault_some_mcp.core.indexer.active_table_reader",
        _reader_for(table, state),
    )

    paths = read._list_from_index("db", ["tag"], None, None, None)

    assert paths == ["note.md"]
    assert not state["held"]
