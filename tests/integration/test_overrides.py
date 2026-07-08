"""Integration tests for MCP tool registration with overrides.

Tests that tool names and descriptions are correctly applied at registration
time, and that disabled tools are excluded.
"""

import asyncio
import os
import tempfile

import pytest
import yaml

from some_vault_some_mcp.config import VaultMcpConfig, ToolOverride, load_overrides
from some_vault_some_mcp.core.embeddings import MockProvider


def _make_config(override_path=None, **kwargs):
    overrides, disabled, _ = load_overrides(override_path)
    cfg = VaultMcpConfig(
        vault_path=kwargs.get("vault_path", "/tmp"),
        db_path=kwargs.get("db_path", "/tmp/vault.lance"),
        tool_overrides=overrides,
        disabled_tools=disabled,
    )
    return cfg


def _write_overrides(data: dict) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(data, f)
        return f.name


def _tool_names(mcp) -> set[str]:
    """Return set of registered tool names via FastMCP async list_tools()."""
    tools = asyncio.run(mcp.list_tools())
    return {t.name for t in tools}


def _tool_count(mcp) -> int:
    tools = asyncio.run(mcp.list_tools())
    return len(tools)


def test_default_registration_uses_default_names(tmp_path):
    """No override file → tools registered with default names."""
    from some_vault_some_mcp.server import build_server
    cfg = _make_config(vault_path=str(tmp_path), db_path=str(tmp_path / "db.lance"))
    provider = MockProvider()
    mcp = build_server(cfg, provider)
    names = _tool_names(mcp)
    assert "get_note" in names
    assert "search" in names
    assert "list_notes" in names


def test_override_renames_tool(tmp_path):
    """Override file renames tools correctly."""
    from some_vault_some_mcp.server import build_server
    data = {"tools": {"get_note": {"name": "recall", "description": "Custom desc"}}}
    override_path = _write_overrides(data)
    cfg = _make_config(
        override_path=override_path,
        vault_path=str(tmp_path),
        db_path=str(tmp_path / "db.lance"),
    )
    provider = MockProvider()
    mcp = build_server(cfg, provider)
    names = _tool_names(mcp)
    assert "recall" in names
    assert "get_note" not in names


def test_disabled_tool_excluded(tmp_path):
    """Disabled tools are not registered."""
    from some_vault_some_mcp.server import build_server
    data = {"disabled": ["vault_reindex"]}
    override_path = _write_overrides(data)
    cfg = _make_config(
        override_path=override_path,
        vault_path=str(tmp_path),
        db_path=str(tmp_path / "db.lance"),
    )
    provider = MockProvider()
    mcp = build_server(cfg, provider)
    names = _tool_names(mcp)
    assert "vault_reindex" not in names


def test_tool_count_with_no_overrides(tmp_path):
    """28 tools registered by default (19 original + 9 canvas tools)."""
    from some_vault_some_mcp.server import build_server
    cfg = _make_config(vault_path=str(tmp_path), db_path=str(tmp_path / "db.lance"))
    provider = MockProvider()
    mcp = build_server(cfg, provider)
    assert _tool_count(mcp) == 28
