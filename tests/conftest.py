"""Shared fixtures for the whole test tree."""

import pytest


@pytest.fixture(autouse=True)
def _reset_excluded_dirs():
    """Clear the configured extra excluded folders after every test so the
    module-level set in core.paths cannot leak between tests."""
    from some_vault_some_mcp.core.paths import configure_excluded_dirs
    yield
    configure_excluded_dirs(())
