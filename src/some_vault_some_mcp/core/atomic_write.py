"""Atomic file write utilities with per-path locking.

Uses temp-file + rename (POSIX rename(2) is atomic on same filesystem) to
ensure readers see either old content or new content, never partial writes.
Per-path asyncio.Lock serialises concurrent MCP calls to the same file.
"""

import asyncio
import os
import secrets
from pathlib import Path

# Per-path locks. Lock key is the resolved absolute path (lower-cased on
# case-insensitive filesystems would be needed, but we target Linux only).
_locks: dict[str, asyncio.Lock] = {}
_locks_mutex = asyncio.Lock()


async def _get_lock(full_path: str) -> asyncio.Lock:
    async with _locks_mutex:
        if full_path not in _locks:
            _locks[full_path] = asyncio.Lock()
        return _locks[full_path]


async def atomic_write(full_path: str, content: str) -> None:
    """Write content to full_path atomically via a sibling temp file + rename.

    Caller must hold the per-path lock (use with_file_lock).
    """
    p = Path(full_path)
    tmp = p.parent / f".{p.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    try:
        tmp.write_text(content, encoding="utf-8")
        # Preserve the destination's permission bits (the temp file is created
        # with default umask, which would otherwise reset mode on every write).
        try:
            os.chmod(str(tmp), os.stat(full_path).st_mode)
        except FileNotFoundError:
            pass  # new file — keep default mode
        os.replace(str(tmp), full_path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


async def atomic_create(full_path: str, content: str) -> None:
    """Create a NEW file atomically; raise FileExistsError if it already exists.

    Writes a sibling temp file then hard-links it into place. ``os.link`` fails
    with FileExistsError if the target exists, so the create is exclusive without
    a separate check-then-write window (closes the TOCTOU in create_note/create_canvas).
    Caller must hold the per-path lock (use with_file_lock).
    """
    p = Path(full_path)
    tmp = p.parent / f".{p.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    try:
        tmp.write_text(content, encoding="utf-8")
        os.link(str(tmp), full_path)  # atomic; raises FileExistsError if target exists
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


async def with_file_lock(full_path: str, fn) -> None:
    """Acquire the per-path lock and call fn() inside it."""
    lock = await _get_lock(full_path)
    async with lock:
        await fn()
