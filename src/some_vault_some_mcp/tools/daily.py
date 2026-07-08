"""Daily note tools: get_daily_note, create_daily_note."""

import json
import logging
from datetime import datetime
from pathlib import Path

from some_vault_some_mcp.core.dates import format_moment_date, parse_date_str
from some_vault_some_mcp.core.frontmatter import parse_frontmatter
from some_vault_some_mcp.core.paths import resolve_internal, resolve_vault_path, VaultPathError, ensure_md_extension
from some_vault_some_mcp.tools.write import create_note

logger = logging.getLogger(__name__)


def _get_daily_note_config(vault_path: str) -> dict:
    """Read .obsidian/daily-notes.json. Falls back to defaults if not found."""
    defaults = {"folder": "", "format": "YYYY-MM-DD"}
    try:
        config_full_path = resolve_internal(vault_path, ".obsidian/daily-notes.json")
        config_path = Path(config_full_path)
    except VaultPathError as e:
        logger.warning(f"Could not resolve daily notes config path: {e}")
        return defaults
    if not config_path.exists():
        return defaults
    try:
        raw = config_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        return {
            "folder": str(parsed.get("folder", "")),
            "format": str(parsed.get("format", "YYYY-MM-DD")),
            "template": str(parsed.get("template", "")) if parsed.get("template") else None,
        }
    except Exception as e:
        logger.warning(f"Failed to read daily notes config: {e}")
        return defaults


def _resolve_daily_note_path(vault_path: str, date_str: str | None = None) -> tuple[str, str]:
    """Return (vault_relative_path, formatted_date_str) for a daily note."""
    config = _get_daily_note_config(vault_path)
    if date_str:
        dt = parse_date_str(date_str)
    else:
        dt = datetime.now()
    formatted = format_moment_date(dt, config["format"])
    filename = ensure_md_extension(formatted)
    folder = config.get("folder", "").strip()
    if folder:
        rel_path = f"{folder}/{filename}"
    else:
        rel_path = filename
    return rel_path, formatted


def get_daily_note(vault_path: str, date: str | None = None) -> dict | None:
    """Read the daily note for a date. Returns dict or None if not found."""
    rel_path, formatted = _resolve_daily_note_path(vault_path, date)
    full_path = Path(vault_path) / rel_path

    if not full_path.exists():
        return None

    content = full_path.read_text(encoding="utf-8", errors="replace")
    fm, body = parse_frontmatter(content)

    return {
        "path": rel_path,
        "date": formatted,
        "frontmatter": fm,
        "content": body,
    }


async def create_daily_note(
    vault_path: str,
    date: str | None = None,
    content: str | None = None,
    template_path: str | None = None,
) -> str:
    """Create a daily note. Returns the vault-relative path created.

    Raises FileExistsError if note already exists.
    """
    rel_path, formatted = _resolve_daily_note_path(vault_path, date)

    final_content = content or ""
    if template_path:
        tmpl_path = ensure_md_extension(template_path)
        # Route through the vault boundary — templates live in-vault, and this is
        # the one filesystem read that previously skipped resolve_vault_path
        # (allowed `../secret` to escape). Obsidian templates are in-vault anyway.
        try:
            full_tmpl = Path(resolve_vault_path(vault_path, tmpl_path))
        except VaultPathError as e:
            raise ValueError(f"Invalid template path: {e}")
        try:
            template_content = full_tmpl.read_text(encoding="utf-8", errors="replace")
            final_content = template_content.replace("{{date}}", formatted)
        except Exception as e:
            raise ValueError(f"Error reading template: {e}")

    await create_note(vault_path, rel_path, final_content)
    return rel_path
