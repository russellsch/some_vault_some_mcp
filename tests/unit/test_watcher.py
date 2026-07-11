"""Unit tests for watcher debounce logic."""

import threading
import time

import pytest


def test_debounce_collapses_rapid_events(tmp_path):
    """Multiple rapid events for the same file collapse to one reindex call."""
    from unittest.mock import MagicMock, patch

    vault_path = str(tmp_path / "vault")
    (tmp_path / "vault").mkdir()

    call_log = []

    def fake_incremental(vault, db, provider, only_files=None, single_file=None):
        call_log.append(only_files)
        return {"files_indexed": 1, "chunks_created": 1, "files_removed": 0, "duration_seconds": 0.0}

    from some_vault_some_mcp.core.watcher import _VaultEventHandler, DEBOUNCE_SECS

    with patch("some_vault_some_mcp.core.watcher.incremental_index", fake_incremental):
        handler = _VaultEventHandler(vault_path, "fake_db", None)
        note_path = str(tmp_path / "vault" / "note.md")

        # Fire 5 rapid events
        for _ in range(5):
            handler._on_event(note_path)

        # Wait for debounce to fire
        time.sleep(DEBOUNCE_SECS + 0.5)

    # Should have collapsed to exactly 1 batched call
    assert len(call_log) == 1
    assert call_log[0] == {"note.md"}


def test_non_md_events_ignored(tmp_path):
    """Non-.md file events are silently dropped."""
    from unittest.mock import patch

    vault_path = str(tmp_path / "vault")
    (tmp_path / "vault").mkdir()
    call_log = []

    def fake_incremental(*args, **kwargs):
        call_log.append(kwargs.get("single_file"))
        return {"files_indexed": 0, "chunks_created": 0, "files_removed": 0, "duration_seconds": 0.0}

    from some_vault_some_mcp.core.watcher import _VaultEventHandler, DEBOUNCE_SECS

    with patch("some_vault_some_mcp.core.watcher.incremental_index", fake_incremental):
        handler = _VaultEventHandler(vault_path, "fake_db", None)
        handler._on_event(str(tmp_path / "vault" / "image.png"))
        handler._on_event(str(tmp_path / "vault" / "data.json"))
        time.sleep(DEBOUNCE_SECS + 0.5)

    assert call_log == []


def test_error_recovery_continues(tmp_path):
    """Single-file indexer failure doesn't prevent next event from being processed."""
    from unittest.mock import patch

    vault_path = str(tmp_path / "vault")
    (tmp_path / "vault").mkdir()
    call_log = []

    call_count = [0]

    def fake_incremental(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("Simulated indexer failure")
        call_log.append(kwargs.get("only_files"))
        return {"files_indexed": 1, "chunks_created": 1, "files_removed": 0, "duration_seconds": 0.0}

    from some_vault_some_mcp.core.watcher import _VaultEventHandler, DEBOUNCE_SECS

    with patch("some_vault_some_mcp.core.watcher.incremental_index", fake_incremental):
        handler = _VaultEventHandler(vault_path, "fake_db", None)
        # First event fails
        handler._on_event(str(tmp_path / "vault" / "note1.md"))
        time.sleep(DEBOUNCE_SECS + 0.5)
        # Second event should still work
        handler._on_event(str(tmp_path / "vault" / "note2.md"))
        time.sleep(DEBOUNCE_SECS + 0.5)

    # Second call succeeded
    assert {"note2.md"} in call_log


def test_excluded_dir_events_dropped(tmp_path):
    """Events under .trash/.git/.obsidian are dropped before any reindex (F8)."""
    from unittest.mock import patch

    vault_path = str(tmp_path / "vault")
    (tmp_path / "vault").mkdir()
    call_log = []

    def fake_incremental(*args, **kwargs):
        call_log.append(kwargs.get("only_files"))
        return {"files_indexed": 0, "chunks_created": 0, "files_removed": 0, "duration_seconds": 0.0}

    from some_vault_some_mcp.core.watcher import _VaultEventHandler, DEBOUNCE_SECS

    with patch("some_vault_some_mcp.core.watcher.incremental_index", fake_incremental):
        handler = _VaultEventHandler(vault_path, "fake_db", None)
        handler._on_event(str(tmp_path / "vault" / ".trash" / "gone.md"))
        handler._on_event(str(tmp_path / "vault" / ".git" / "x.md"))
        time.sleep(DEBOUNCE_SECS + 0.5)

    assert call_log == []
