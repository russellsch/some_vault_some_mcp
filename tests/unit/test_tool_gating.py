"""Unit tests for IndexGate and tool gating during background indexing."""

import asyncio
import shutil
from pathlib import Path

import pytest

from some_vault_some_mcp.server import IndexGate, _check_index_gate, build_server
from some_vault_some_mcp.config import VaultMcpConfig, load_overrides
from some_vault_some_mcp.core.embeddings import MockProvider

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "vault"


# ── IndexGate unit tests ────────────────────────────────────────────────


class TestIndexGate:
    def test_not_ready_initially(self):
        g = IndexGate()
        assert g.is_ready is False
        assert g.error is None

    def test_ready_after_set(self):
        g = IndexGate()
        g.set_ready()
        assert g.is_ready is True
        assert g.error is None

    def test_failed_sets_error(self):
        g = IndexGate()
        g.set_failed("boom")
        assert g.is_ready is False
        assert g.error == "boom"

    def test_double_set_ready_is_idempotent(self):
        g = IndexGate()
        g.set_ready()
        g.set_ready()
        assert g.is_ready is True

    def test_set_ready_after_failed_stays_failed(self):
        g = IndexGate()
        g.set_failed("boom")
        g.set_ready()
        assert g.is_ready is False
        assert g.error == "boom"


# ── _check_index_gate unit tests ────────────────────────────────────────


class TestCheckIndexGate:
    def test_none_gate_returns_none(self):
        assert _check_index_gate(None, "search") is None

    def test_ready_gate_returns_none(self):
        g = IndexGate()
        g.set_ready()
        assert _check_index_gate(g, "search") is None

    def test_indexing_returns_message_with_tool_name(self):
        g = IndexGate()
        msg = _check_index_gate(g, "search")
        assert msg is not None
        assert "search" in msg
        assert "being built" in msg

    def test_failed_returns_error_message(self):
        g = IndexGate()
        g.set_failed("connection refused")
        msg = _check_index_gate(g, "search")
        assert msg is not None
        assert "connection refused" in msg
        assert "failed" in msg


# ── Tool-level gating tests ─────────────────────────────────────────────


def _make_config(tmp_path):
    overrides, disabled, _ = load_overrides(None)
    return VaultMcpConfig(
        vault_path=str(tmp_path / "vault"),
        db_path=str(tmp_path / "db.lance"),
        tool_overrides=overrides,
        disabled_tools=disabled,
    )


def _call_tool(mcp, name, args=None):
    """Call an MCP tool and return the text result."""
    result = asyncio.run(mcp.call_tool(name, arguments=args or {}))
    return result.content[0].text


@pytest.fixture()
def vault(tmp_path):
    vault_dir = tmp_path / "vault"
    shutil.copytree(str(FIXTURES), str(vault_dir), ignore=shutil.ignore_patterns(".git"))
    return tmp_path


class TestToolGatingDuringIndexing:
    """Tools gated while gate is not ready."""

    def test_search_hybrid_gated(self, vault):
        gate = IndexGate()
        mcp = build_server(_make_config(vault), MockProvider(), gate)
        result = _call_tool(mcp, "search", {"query": "test", "mode": "hybrid"})
        assert "being built" in result
        assert "mode='exact'" in result

    def test_search_semantic_gated(self, vault):
        gate = IndexGate()
        mcp = build_server(_make_config(vault), MockProvider(), gate)
        result = _call_tool(mcp, "search", {"query": "test", "mode": "semantic"})
        assert "being built" in result

    def test_search_exact_not_gated(self, vault):
        gate = IndexGate()
        mcp = build_server(_make_config(vault), MockProvider(), gate)
        result = _call_tool(mcp, "search", {"query": "simple", "mode": "exact"})
        assert "being built" not in result

    def test_list_notes_with_tags_gated(self, vault):
        gate = IndexGate()
        mcp = build_server(_make_config(vault), MockProvider(), gate)
        result = _call_tool(mcp, "list_notes", {"tags": ["test"]})
        assert "being built" in result
        assert "list_notes" in result

    def test_list_notes_with_folder_not_gated(self, vault):
        gate = IndexGate()
        mcp = build_server(_make_config(vault), MockProvider(), gate)
        result = _call_tool(mcp, "list_notes", {"folder": "projects"})
        assert "being built" not in result

    def test_list_notes_no_filters_not_gated(self, vault):
        gate = IndexGate()
        mcp = build_server(_make_config(vault), MockProvider(), gate)
        result = _call_tool(mcp, "list_notes", {})
        assert "being built" not in result

    def test_vault_reindex_gated(self, vault):
        gate = IndexGate()
        mcp = build_server(_make_config(vault), MockProvider(), gate)
        result = _call_tool(mcp, "vault_reindex", {})
        assert "already being built" in result

    def test_vault_index_status_shows_indexing(self, vault):
        gate = IndexGate()
        mcp = build_server(_make_config(vault), MockProvider(), gate)
        result = _call_tool(mcp, "vault_index_status", {})
        assert "INDEXING" in result
        assert "Files indexed" in result

    def test_get_note_not_gated(self, vault):
        gate = IndexGate()
        mcp = build_server(_make_config(vault), MockProvider(), gate)
        result = _call_tool(mcp, "get_note", {"path": "simple.md"})
        assert "being built" not in result


class TestToolsAfterGateReady:
    """All tools work normally once gate is ready."""

    def test_search_hybrid_works_after_ready(self, vault):
        gate = IndexGate()
        gate.set_ready()
        mcp = build_server(_make_config(vault), MockProvider(), gate)
        result = _call_tool(mcp, "search", {"query": "test", "mode": "exact"})
        assert "being built" not in result

    def test_list_notes_with_tags_works_after_ready(self, vault):
        gate = IndexGate()
        gate.set_ready()
        mcp = build_server(_make_config(vault), MockProvider(), gate)
        result = _call_tool(mcp, "list_notes", {"tags": ["test"]})
        assert "being built" not in result

    def test_vault_index_status_shows_ready(self, vault):
        gate = IndexGate()
        gate.set_ready()
        mcp = build_server(_make_config(vault), MockProvider(), gate)
        result = _call_tool(mcp, "vault_index_status", {})
        assert "READY" in result

    def test_no_gate_all_tools_work(self, vault):
        """Backward compat: gate=None means no gating."""
        mcp = build_server(_make_config(vault), MockProvider())
        result = _call_tool(mcp, "search", {"query": "test", "mode": "exact"})
        assert "being built" not in result
