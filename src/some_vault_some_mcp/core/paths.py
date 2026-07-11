"""Vault-boundary path resolver.

All user-supplied paths from MCP tool parameters go through resolve_vault_path()
before any filesystem operation. The resolver:
- rejects null bytes
- rejects path traversal (../)
- rejects access to .obsidian, .git, .trash at any depth (MCP boundary)
- resolves symlinks to catch symlink escapes (realpath check)

Internal server code that needs to read .obsidian/ (e.g. daily-notes.json)
calls resolve_internal() which skips the excluded-dir check.
"""

import os
from pathlib import Path


EXCLUDED_DIRS = frozenset([".obsidian", ".git", ".trash"])


class VaultPathError(Exception):
    """Raised when a user-supplied path violates vault boundary rules."""


def _check_excluded(rel: str) -> None:
    """Raise VaultPathError if any segment of rel is an excluded dir."""
    parts = rel.replace("\\", "/").split("/")
    for seg in parts:
        if seg.lower() in EXCLUDED_DIRS:
            raise VaultPathError(f"Access to excluded directory denied: {rel}")


def resolve_vault_path(vault_path: str, relative_path: str) -> str:
    """Resolve a user-supplied relative path to an absolute path within vault.

    Raises VaultPathError on traversal, null bytes, or excluded dir access.
    Returns the resolved absolute path as str.
    """
    if not vault_path:
        raise VaultPathError("Vault path is not configured")
    if "\0" in relative_path:
        raise VaultPathError("Invalid path: contains null byte")

    # Check excluded segments on the user-supplied path too (before resolve), so a
    # symlink can't smuggle access to .obsidian/.git/.trash past the post-resolve
    # check (O8a). The post-resolve check below still guards realpath containment.
    _check_excluded(relative_path)

    vault = Path(vault_path).resolve()
    candidate = (vault / relative_path).resolve()

    # Must start with vault root
    try:
        candidate.relative_to(vault)
    except ValueError:
        raise VaultPathError("Path traversal detected")

    # Check excluded dirs in the relative portion
    rel = str(candidate.relative_to(vault))
    if rel and rel != ".":
        _check_excluded(rel)

    return str(candidate)


def resolve_internal(vault_path: str, relative_path: str) -> str:
    """Resolve a server-internal path (e.g. .obsidian/daily-notes.json).

    Checks traversal boundary but NOT the excluded-dir list — the server
    itself must be able to read .obsidian/ config files.
    """
    if not vault_path:
        raise VaultPathError("Vault path is not configured")
    if "\0" in relative_path:
        raise VaultPathError("Invalid path: contains null byte")

    vault = Path(vault_path).resolve()
    candidate = (vault / relative_path).resolve()

    try:
        candidate.relative_to(vault)
    except ValueError:
        raise VaultPathError("Path traversal detected")

    return str(candidate)


def _within_vault(path: Path, vault_root: Path) -> bool:
    """True if path's real target is inside the vault (blocks symlink escapes).

    The index-discovery walkers enforce the same boundary as resolve_vault_path,
    so a symlink pointing outside the vault is never read into the index.
    """
    try:
        return path.resolve().is_relative_to(vault_root)
    except OSError:
        return False  # broken symlink / cycle


def walk_vault(vault_path: str) -> list[str]:
    """Return vault-relative paths of all .md files, excluding EXCLUDED_DIRS and
    any file whose real path escapes the vault (symlink boundary)."""
    vault = Path(vault_path)
    vault_root = vault.resolve()
    results: list[str] = []
    for path in vault.rglob("*.md"):
        rel = str(path.relative_to(vault)).replace("\\", "/")
        parts = rel.split("/")
        if any(seg.lower() in EXCLUDED_DIRS for seg in parts):
            continue
        if not _within_vault(path, vault_root):
            continue
        results.append(rel)
    return sorted(results)


def check_blocked_suffixes(path: str, blocked: list[str], message: str = "") -> None:
    """Raise ValueError if path ends with any blocked suffix (case-insensitive).

    Checked against the raw user-supplied path before ensure_md_extension runs,
    so "note.md.old" is caught by suffix ".old" before it becomes "note.md.old.md".
    """
    lower = path.lower()
    for suffix in blocked:
        if lower.endswith(suffix.lower()):
            raise ValueError(
                message or f"Path '{path}' uses blocked suffix '{suffix}'."
            )


def ensure_md_extension(path: str) -> str:
    """Append .md if the path doesn't already end with .md (case-insensitive)."""
    if not path.lower().endswith(".md"):
        return path + ".md"
    return path


def strip_md_suffix(path: str) -> str:
    """Remove a single trailing .md (case-insensitive). Leaves other dots alone."""
    return path[:-3] if path.lower().endswith(".md") else path


def resolve_note_path(target: str, all_notes: list[str]) -> str | None:
    """Resolve a user-supplied note reference to a vault-relative .md path.

    Extension-agnostic ("todo" and "todo.md" both match todo.md) and
    Obsidian-style: a bare name resolves by basename anywhere in the vault.

    Resolution order:
      1. exact relative-path match (case-insensitive, .md optional)
      2. basename match (case-insensitive), first by sorted vault order

    all_notes must be vault-relative .md paths (e.g. from walk_vault).
    Returns None if nothing matches.
    """
    target_norm = strip_md_suffix(target).replace("\\", "/").lower()
    target_basename = target_norm.split("/")[-1]

    for note in all_notes:
        if strip_md_suffix(note).lower() == target_norm:
            return note

    for note in all_notes:
        note_basename = strip_md_suffix(note).split("/")[-1].lower()
        if note_basename == target_basename:
            return note

    return None


def ensure_canvas_extension(path: str) -> str:
    """Append .canvas if the path doesn't already end with .canvas (case-insensitive)."""
    if not path.lower().endswith(".canvas"):
        return path + ".canvas"
    return path


def walk_canvas(vault_path: str) -> list[str]:
    """Return vault-relative paths of all .canvas files, excluding EXCLUDED_DIRS
    and any file whose real path escapes the vault (symlink boundary)."""
    vault = Path(vault_path)
    vault_root = vault.resolve()
    results: list[str] = []
    for path in vault.rglob("*.canvas"):
        rel = str(path.relative_to(vault)).replace("\\", "/")
        parts = rel.split("/")
        if any(seg.lower() in EXCLUDED_DIRS for seg in parts):
            continue
        if not _within_vault(path, vault_root):
            continue
        results.append(rel)
    return sorted(results)
