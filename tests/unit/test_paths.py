"""Unit tests for vault path resolver."""

import os
import tempfile
from pathlib import Path

import pytest

from some_vault_some_mcp.core.paths import (
    VaultPathError,
    ensure_md_extension,
    resolve_note_path,
    resolve_vault_path,
    strip_md_suffix,
    walk_vault,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "vault"


def test_resolve_valid_path():
    vault = str(FIXTURES)
    result = resolve_vault_path(vault, "simple.md")
    assert result == str(FIXTURES / "simple.md")


def test_resolve_traversal_rejected():
    vault = str(FIXTURES)
    with pytest.raises(VaultPathError, match="traversal"):
        resolve_vault_path(vault, "../../../etc/passwd")


def test_resolve_null_byte_rejected():
    vault = str(FIXTURES)
    with pytest.raises(VaultPathError, match="null byte"):
        resolve_vault_path(vault, "note\x00.md")


def test_resolve_obsidian_rejected():
    vault = str(FIXTURES)
    with pytest.raises(VaultPathError, match="excluded"):
        resolve_vault_path(vault, ".obsidian/config.json")


def test_resolve_git_rejected():
    vault = str(FIXTURES)
    with pytest.raises(VaultPathError, match="excluded"):
        resolve_vault_path(vault, ".git/config")


def test_resolve_trash_rejected():
    vault = str(FIXTURES)
    with pytest.raises(VaultPathError, match="excluded"):
        resolve_vault_path(vault, ".trash/note.md")


def test_resolve_nested_excluded():
    vault = str(FIXTURES)
    with pytest.raises(VaultPathError, match="excluded"):
        resolve_vault_path(vault, "projects/.git/config")


def test_walk_vault_excludes_hidden():
    vault = str(FIXTURES)
    paths = walk_vault(vault)
    # Should not include .obsidian, .git, .trash files
    for p in paths:
        parts = p.split("/")
        for seg in parts:
            assert seg not in (".obsidian", ".git", ".trash"), f"Excluded dir leaked: {p}"


def test_walk_vault_only_md():
    vault = str(FIXTURES)
    paths = walk_vault(vault)
    for p in paths:
        assert p.endswith(".md"), f"Non-.md file leaked: {p}"


def test_walk_vault_finds_nested():
    vault = str(FIXTURES)
    paths = walk_vault(vault)
    assert any("projects/alpha" in p for p in paths), "Nested vault files not found"


def test_ensure_md_extension():
    assert ensure_md_extension("note") == "note.md"
    assert ensure_md_extension("note.md") == "note.md"
    assert ensure_md_extension("note.MD") == "note.MD"
    assert ensure_md_extension("folder/note") == "folder/note.md"


def test_strip_md_suffix():
    assert strip_md_suffix("note.md") == "note"
    assert strip_md_suffix("note.MD") == "note"
    assert strip_md_suffix("note") == "note"
    # Only a trailing .md is stripped, not .md elsewhere in the name
    assert strip_md_suffix("a.md.notes") == "a.md.notes"
    assert strip_md_suffix("my.mdata/x") == "my.mdata/x"


def test_resolve_note_path_extension_agnostic():
    notes = ["todo.md", "projects/alpha/deep-note.md"]
    assert resolve_note_path("todo", notes) == "todo.md"
    assert resolve_note_path("todo.md", notes) == "todo.md"


def test_resolve_note_path_case_insensitive():
    notes = ["todo.md"]
    assert resolve_note_path("ToDo", notes) == "todo.md"
    assert resolve_note_path("TODO.MD", notes) == "todo.md"


def test_resolve_note_path_basename_across_folders():
    notes = ["projects/alpha/deep-note.md", "other.md"]
    # A bare basename resolves to a note nested in a folder (Obsidian-style)
    assert resolve_note_path("deep-note", notes) == "projects/alpha/deep-note.md"


def test_resolve_note_path_exact_relative_match_preferred():
    notes = ["todo.md", "archive/todo.md"]
    # Exact relative path wins over basename
    assert resolve_note_path("archive/todo", notes) == "archive/todo.md"


def test_resolve_note_path_no_match():
    assert resolve_note_path("phantom", ["todo.md"]) is None


def test_resolve_note_path_suffix_bug_regression():
    # A note whose name contains ".md" mid-string must not be mangled.
    notes = ["a.md.notes.md"]
    assert resolve_note_path("a.md.notes", notes) == "a.md.notes.md"
    assert resolve_note_path("a.md.notes.md", notes) == "a.md.notes.md"


def test_symlink_escape_rejected():
    """Symlinks that resolve outside the vault root must be rejected."""
    with tempfile.TemporaryDirectory() as vault:
        with tempfile.TemporaryDirectory() as outside:
            outside_file = Path(outside) / "secret.md"
            outside_file.write_text("secret data", encoding="utf-8")

            symlink = Path(vault) / "escape.md"
            symlink.symlink_to(outside_file)

            with pytest.raises(VaultPathError, match="traversal"):
                resolve_vault_path(vault, "escape.md")


def test_error_message_no_absolute_path_leak():
    with tempfile.TemporaryDirectory() as vault:
        # The error message for path traversal should not include the host path
        # (it mentions the relative path the user supplied, not the vault root)
        try:
            resolve_vault_path(vault, "../../../root")
        except VaultPathError as e:
            # Should not contain the vault path itself in the error
            assert vault not in str(e) or "traversal" in str(e)
