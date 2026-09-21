"""Daily note tools: get_daily_note, create_daily_note."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from some_vault_some_mcp.core.dates import format_moment_date, parse_date_str
from some_vault_some_mcp.core.frontmatter import parse_frontmatter
from some_vault_some_mcp.core.paths import resolve_internal, resolve_vault_path, VaultPathError, ensure_md_extension
from some_vault_some_mcp.tools.write import create_note

logger = logging.getLogger(__name__)


def _daily_path_error() -> ValueError:
    """Return the intentionally content-free error used for bad config paths."""
    return ValueError("Invalid daily note path")


def _validated_daily_relpath(vault_path: str, rel_path: str) -> str:
    """Validate a computed daily path and return its canonical vault-relative form.

    Daily-note settings are configuration, rather than a direct tool argument, so
    they need their own lexical check before the normal vault resolver.  In
    particular, Windows absolute paths would otherwise be ordinary filenames on
    POSIX hosts.
    """
    if not isinstance(rel_path, str) or "\0" in rel_path:
        raise _daily_path_error()

    normalized = rel_path.replace("\\", "/")
    if (
        normalized.startswith("/")
        or normalized.startswith("//")
        or re.match(r"^[A-Za-z]:", normalized)
        or any(part == ".." for part in normalized.split("/"))
        or any(part.lower() in {".obsidian", ".git", ".trash"} for part in normalized.split("/"))
    ):
        raise _daily_path_error()

    try:
        resolved = Path(resolve_vault_path(vault_path, normalized))
        return resolved.relative_to(Path(vault_path).resolve()).as_posix()
    except (VaultPathError, ValueError, OSError):
        raise _daily_path_error() from None


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
        if not isinstance(parsed, dict):
            return defaults
        folder = parsed.get("folder", "")
        date_format = parsed.get("format", "YYYY-MM-DD")
        if not isinstance(folder, str) or not isinstance(date_format, str) or not date_format:
            return defaults
        return {
            "folder": folder,
            "format": date_format,
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
    return _validated_daily_relpath(vault_path, rel_path), formatted


def get_daily_note(vault_path: str, date: str | None = None) -> dict | None:
    """Read the daily note for a date. Returns dict or None if not found."""
    rel_path, formatted = _resolve_daily_note_path(vault_path, date)
    try:
        full_path = Path(resolve_vault_path(vault_path, rel_path))
    except VaultPathError:
        raise _daily_path_error() from None

    if not full_path.exists():
        return None

    # Resolve immediately before the file read as well, so a swapped symlink
    # cannot turn the existence check into an out-of-vault read.
    try:
        full_path = Path(resolve_vault_path(vault_path, rel_path))
    except VaultPathError:
        raise _daily_path_error() from None
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
    # Keep validation adjacent to creation even though _resolve_daily_note_path
    # also validates: config may be changed between computation and write.
    rel_path = _validated_daily_relpath(vault_path, rel_path)

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
