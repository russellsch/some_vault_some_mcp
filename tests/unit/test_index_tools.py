"""Unit tests for index tool orchestration."""

from some_vault_some_mcp.tools import index as index_tools
from some_vault_some_mcp.core import indexer


def test_targeted_reindex_promotes_to_full_rebuild_for_buffering_watcher(monkeypatch):
    watcher = type("Watcher", (), {"is_buffering": True})()
    provider = object()
    calls = []

    monkeypatch.setattr(
        "some_vault_some_mcp.core.watcher.get_watcher",
        lambda _db_path: watcher,
    )
    monkeypatch.setattr(
        index_tools,
        "full_index",
        lambda **kwargs: calls.append(kwargs) or {
            "files_indexed": 2,
            "chunks_created": 2,
            "files_removed": 0,
            "files_skipped": 0,
            "duration_seconds": 0.0,
        },
    )
    monkeypatch.setattr(
        index_tools,
        "incremental_index",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("buffering watcher must not use targeted incremental")
        ),
    )

    result = index_tools.vault_reindex("vault", "db", provider, "one.md")

    assert result.files_indexed == 2
    assert calls == [
        {
            "vault_path": "vault",
            "db_path": "db",
            "provider": provider,
            "watcher": watcher,
        }
    ]


def test_targeted_reindex_stays_incremental_for_normal_watcher(monkeypatch):
    watcher = type("Watcher", (), {"is_buffering": False})()
    provider = object()
    calls = []
    monkeypatch.setattr(
        "some_vault_some_mcp.core.watcher.get_watcher",
        lambda _db_path: watcher,
    )
    monkeypatch.setattr(
        index_tools,
        "full_index",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("normal watcher must not force a full rebuild")
        ),
    )
    monkeypatch.setattr(
        index_tools,
        "incremental_index",
        lambda **kwargs: calls.append(kwargs) or {
            "files_indexed": 1,
            "chunks_created": 1,
            "files_removed": 0,
            "files_skipped": 0,
            "duration_seconds": 0.0,
        },
    )

    result = index_tools.vault_reindex("vault", "db", provider, "one.md")

    assert result.files_indexed == 1
    assert calls == [
        {
            "vault_path": "vault",
            "db_path": "db",
            "provider": provider,
            "single_file": "one.md",
        }
    ]


def test_core_targeted_incremental_promotes_when_watcher_is_buffering(monkeypatch):
    watcher = type("Watcher", (), {"is_buffering": True})()
    provider = object()
    calls = []
    monkeypatch.setattr(
        "some_vault_some_mcp.core.watcher.get_watcher",
        lambda _db_path: watcher,
    )
    monkeypatch.setattr(
        indexer,
        "full_index",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"recovered": True},
    )

    result = indexer.incremental_index(
        "vault",
        "db",
        provider,
        single_file="one.md",
    )

    assert result == {"recovered": True}
    assert calls == [(('vault', 'db', provider, indexer.BATCH_SIZE), {"watcher": watcher})]
