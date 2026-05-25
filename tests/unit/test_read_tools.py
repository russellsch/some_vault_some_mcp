"""Unit tests for get_note path resolution (extension-agnostic, Obsidian-style)."""

from pathlib import Path

import pytest

from some_vault_some_mcp.tools.read import get_note


@pytest.fixture()
def vault(tmp_path):
    """Small vault: a root note and a note nested in a folder."""
    (tmp_path / "todo.md").write_text("---\ntitle: Todo\n---\n\nThings.", encoding="utf-8")
    sub = tmp_path / "projects" / "alpha"
    sub.mkdir(parents=True)
    (sub / "deep-note.md").write_text("---\ntitle: Deep\n---\n\nNested.", encoding="utf-8")
    return str(tmp_path)


def test_get_note_with_extension(vault):
    note = get_note(vault, "todo.md")
    assert note is not None
    assert note.file_path == "todo.md"
    assert "Things." in note.content


def test_get_note_without_extension(vault):
    note = get_note(vault, "todo")
    assert note is not None
    assert note.file_path == "todo.md"
    assert "Things." in note.content


def test_get_note_basename_across_folders(vault):
    # A bare basename resolves to the nested note (Obsidian-style)
    note = get_note(vault, "deep-note")
    assert note is not None
    assert note.file_path == "projects/alpha/deep-note.md"


def test_get_note_relative_path(vault):
    note = get_note(vault, "projects/alpha/deep-note")
    assert note is not None
    assert note.file_path == "projects/alpha/deep-note.md"


def test_get_note_missing_returns_none(vault):
    assert get_note(vault, "phantom") is None


def test_get_note_traversal_rejected(vault):
    assert get_note(vault, "../secret") is None
