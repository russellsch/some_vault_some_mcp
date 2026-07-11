"""Unit tests for atomic write utilities."""

import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from some_vault_some_mcp.core.atomic_write import atomic_write, with_file_lock


@pytest.mark.asyncio
async def test_concurrent_atomic_write_same_path(tmp_path):
    """Multiple concurrent writes to the same path produce no corruption.

    Final content must be one of the expected values (serialised by per-path lock).
    """
    target = str(tmp_path / "concurrent.md")
    values = [f"writer-{i}" for i in range(10)]

    async def write_value(val: str):
        await with_file_lock(target, lambda: atomic_write(target, val))

    await asyncio.gather(*(write_value(v) for v in values))

    content = Path(target).read_text(encoding="utf-8")
    assert content in values, f"Final content '{content}' is not one of the expected values"


@pytest.mark.asyncio
async def test_temp_file_cleanup_on_error(tmp_path):
    """If atomic_write fails mid-write, no temp files remain."""
    target = str(tmp_path / "fail-target.md")

    with patch("some_vault_some_mcp.core.atomic_write.os.replace", side_effect=OSError("injected")):
        with pytest.raises(OSError, match="injected"):
            await atomic_write(target, "should not persist")

    remaining = list(tmp_path.iterdir())
    assert remaining == [], f"Temp files left behind: {[f.name for f in remaining]}"


@pytest.mark.asyncio
async def test_atomic_write_basic(tmp_path):
    """Basic atomic write creates file with correct content."""
    target = str(tmp_path / "basic.md")
    await atomic_write(target, "hello world")
    assert Path(target).read_text(encoding="utf-8") == "hello world"


@pytest.mark.asyncio
async def test_atomic_create_is_exclusive(tmp_path):
    from some_vault_some_mcp.core.atomic_write import atomic_create
    target = str(tmp_path / "a.txt")
    await atomic_create(target, "first")
    assert Path(target).read_text() == "first"
    with pytest.raises(FileExistsError):
        await atomic_create(target, "second")
    assert Path(target).read_text() == "first"  # existing content not clobbered
