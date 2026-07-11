"""Unit tests for IndexGate recovery (plan Phase 3 / F15)."""

from some_vault_some_mcp.server import IndexGate


def test_reset_to_ready_clears_failure():
    gate = IndexGate()
    gate.set_failed("boom")
    assert gate.is_ready is False
    assert gate.error == "boom"
    gate.reset_to_ready()
    assert gate.is_ready is True
    assert gate.error is None
