"""Config loader for tool overrides and server settings.

Reads some_vault_some_mcp_OVERRIDES env var for the path to the per-agent YAML
override file. Override file structure:

    tools:
      <default_tool_name>:
        name: <custom_name>
        description: <custom_desc>
    disabled:
      - <tool_name>

Override application happens at registration time (not per-call).
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ToolOverride:
    name: str | None = None
    description: str | None = None


@dataclass
class VaultMcpConfig:
    vault_path: str = ""
    db_path: str = ""
    transport: str = "sse"
    host: str = "127.0.0.1"
    port: int = 3789
    api_key: str = ""
    allow_unauth_sse: bool = False
    soft_delete_is_permanent: bool = False
    tool_overrides: dict[str, ToolOverride] = field(default_factory=dict)
    disabled_tools: set[str] = field(default_factory=set)
    blocked_path_suffixes: list[str] = field(default_factory=list)
    blocked_suffix_message: str = ""


def load_overrides(
    override_path: str | None = None,
) -> tuple[dict[str, ToolOverride], set[str], dict]:
    """Load tool overrides from YAML file.

    Returns (overrides_dict, disabled_set, path_validation_dict).
    All empty on missing/empty file.

    path_validation_dict keys:
      blocked_suffixes: list[str]
      message: str
    """
    path = (
        override_path
        or os.getenv("VAULT_MCP_OVERRIDES")
        or os.getenv("some_vault_some_mcp_OVERRIDES", "")  # legacy name, kept for compat
    )
    if not path:
        return {}, set(), {}

    p = Path(path)
    if not p.exists():
        logger.info(f"Override file not found at {path} — using defaults")
        return {}, set(), {}

    try:
        raw = p.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
    except Exception as e:
        logger.warning(f"Failed to parse override file {path}: {e} — using defaults")
        return {}, set(), {}

    overrides: dict[str, ToolOverride] = {}
    tools_data = data.get("tools") or {}
    for default_name, spec in tools_data.items():
        if not isinstance(spec, dict):
            continue
        overrides[str(default_name)] = ToolOverride(
            name=spec.get("name"),
            description=spec.get("description"),
        )

    disabled_raw = data.get("disabled") or []
    disabled = {str(t) for t in disabled_raw if t}

    pv = data.get("path_validation") or {}
    path_validation = {
        "blocked_suffixes": [str(s) for s in (pv.get("blocked_suffixes") or []) if s],
        "message": str(pv.get("message") or ""),
    }

    return overrides, disabled, path_validation


def load_config() -> VaultMcpConfig:
    """Build VaultMcpConfig from environment variables."""
    overrides, disabled, path_validation = load_overrides()
    raw = os.getenv("VAULT_SOFT_DELETE_IS_PERMANENT", "").strip().lower()

    port_raw = os.getenv("MCP_PORT", "3789")
    try:
        port = int(port_raw)
    except ValueError:
        logger.warning(f"Invalid MCP_PORT={port_raw!r} — falling back to 3789")
        port = 3789

    # YAML path_validation takes precedence; env vars are the fallback.
    blocked = path_validation.get("blocked_suffixes") or [
        s.strip()
        for s in os.getenv("VAULT_BLOCKED_PATH_SUFFIXES", "").split(",")
        if s.strip()
    ]
    blocked_message = path_validation.get("message") or os.getenv(
        "VAULT_BLOCKED_SUFFIX_MESSAGE", ""
    )

    return VaultMcpConfig(
        vault_path=os.getenv("VAULT_PATH", ""),
        db_path=os.getenv("LANCE_DB_PATH", "./data/vault.lance"),
        transport=os.getenv("MCP_TRANSPORT", "sse"),
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=port,
        api_key=os.getenv("VAULT_API_KEY", ""),
        allow_unauth_sse=os.getenv("VAULT_ALLOW_UNAUTH_SSE", "").strip().lower()
        in ("1", "true", "yes"),
        soft_delete_is_permanent=raw in ("1", "true", "yes"),
        tool_overrides=overrides,
        disabled_tools=disabled,
        blocked_path_suffixes=blocked,
        blocked_suffix_message=blocked_message,
    )


def apply_override(
    default_name: str,
    default_desc: str,
    overrides: dict[str, ToolOverride],
) -> tuple[str, str]:
    """Return (name, description) with any override applied."""
    override = overrides.get(default_name)
    if override is None:
        return default_name, default_desc
    name = override.name if override.name else default_name
    desc = override.description if override.description else default_desc
    return name, desc
