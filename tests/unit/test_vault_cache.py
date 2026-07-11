"""Unit tests for the mtime-validated vault cache (plan Phase 4 / O11/F11)."""

import os

from some_vault_some_mcp.core.vault_cache import read_all


def test_read_all_returns_content_and_reflects_changes(tmp_path):
    vault = tmp_path / "v"
    vault.mkdir()
    (vault / "a.md").write_text("v1", encoding="utf-8")

    notes, contents = read_all(str(vault))
    assert notes == ["a.md"]
    assert contents["a.md"] == "v1"

    # Change content + bump mtime → cache must re-read.
    (vault / "a.md").write_text("v2", encoding="utf-8")
    os.utime(vault / "a.md", (2_000_000_000, 2_000_000_000))
    _, contents2 = read_all(str(vault))
    assert contents2["a.md"] == "v2"


def test_read_all_drops_deleted_file(tmp_path):
    vault = tmp_path / "v"
    vault.mkdir()
    (vault / "a.md").write_text("x", encoding="utf-8")
    read_all(str(vault))
    (vault / "a.md").unlink()
    notes, contents = read_all(str(vault))
    assert "a.md" not in notes
    assert "a.md" not in contents
