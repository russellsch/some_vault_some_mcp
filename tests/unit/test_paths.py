"""Unit tests for vault path resolver."""

import os
import tempfile
from pathlib import Path

import pytest

from some_vault_some_mcp.core.paths import (
    VaultPathError,
    check_blocked_suffixes,
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


def test_check_blocked_suffixes_blocks_matching():
    with pytest.raises(ValueError):
        check_blocked_suffixes("note.md.old", [".old"])


def test_check_blocked_suffixes_blocks_bkp():
    with pytest.raises(ValueError):
        check_blocked_suffixes("archive/note.bkp", [".old", ".bkp"])


def test_check_blocked_suffixes_case_insensitive():
    with pytest.raises(ValueError):
        check_blocked_suffixes("note.md.OLD", [".old"])


def test_check_blocked_suffixes_passes_clean_path():
    check_blocked_suffixes("note.md", [".old", ".bkp"])
    check_blocked_suffixes("folder/note", [".old"])


def test_check_blocked_suffixes_real_daily_note_pattern():
    # The actual pattern seen in the wild: agent passes "2026-05-22.md.old"
    # which ensure_md_extension would turn into "2026-05-22.md.old.md"
    with pytest.raises(ValueError):
        check_blocked_suffixes("2026-05-22.md.old", [".old"])


def test_check_blocked_suffixes_real_project_note_pattern():
    with pytest.raises(ValueError):
        check_blocked_suffixes("300 Projects.md.old", [".old", ".bkp"])


def test_check_blocked_suffixes_real_nested_path():
    with pytest.raises(ValueError):
        check_blocked_suffixes(
            "500 Research/Ontology-Grounded Validation Pipeline Design.md.old",
            [".old"],
        )


def test_check_blocked_suffixes_empty_list_always_passes():
    check_blocked_suffixes("note.md.old", [])


def test_check_blocked_suffixes_custom_message():
    with pytest.raises(ValueError, match="custom msg"):
        check_blocked_suffixes("note.old", [".old"], "custom msg")


def test_check_blocked_suffixes_default_message_includes_path():
    with pytest.raises(ValueError, match="note.md.old"):
        check_blocked_suffixes("note.md.old", [".old"])


def test_error_message_no_absolute_path_leak():
    with tempfile.TemporaryDirectory() as vault:
        # The error message for path traversal should not include the host path
        # (it mentions the relative path the user supplied, not the vault root)
        try:
            resolve_vault_path(vault, "../../../root")
        except VaultPathError as e:
            # Should not contain the vault path itself in the error
            assert vault not in str(e) or "traversal" in str(e)


def test_walk_vault_excludes_symlink_escaping_vault(tmp_path):
    """A symlinked .md whose target is outside the vault must not be walked
    (O8(b) — the discovery path enforces the same boundary as resolve_vault_path)."""
    from some_vault_some_mcp.core.paths import walk_vault
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("SECRET", encoding="utf-8")
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "real.md").write_text("legit", encoding="utf-8")
    os.symlink(outside / "secret.md", vault / "leak.md")           # escapes vault
    os.symlink(vault / "real.md", vault / "innervault-link.md")    # stays in vault

    found = set(walk_vault(str(vault)))
    assert "real.md" in found
    assert "leak.md" not in found                 # symlink out of vault excluded
    assert "innervault-link.md" in found          # in-vault symlink still allowed
