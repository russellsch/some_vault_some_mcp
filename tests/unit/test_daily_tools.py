"""Unit tests for daily note tools: get_daily_note, create_daily_note."""

import json
import shutil
from pathlib import Path

import pytest

from some_vault_some_mcp.tools.daily import get_daily_note, create_daily_note


FIXTURES = Path(__file__).parent.parent / "fixtures" / "vault"


@pytest.fixture()
def vault(tmp_path):
    """Temporary vault with daily-notes config and an existing daily note."""
    vault_dir = tmp_path / "vault"
    shutil.copytree(
        str(FIXTURES),
        str(vault_dir),
        ignore=shutil.ignore_patterns(".git", ".trash"),
    )
    (vault_dir / ".trash").mkdir(exist_ok=True)
    return str(vault_dir)


@pytest.fixture()
def minimal_vault(tmp_path):
    """Minimal vault with a custom daily-notes.json config."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    obsidian_dir = vault_dir / ".obsidian"
    obsidian_dir.mkdir()
    # Custom config: folder=journal, format=YYYY-MM-DD
    config = {"folder": "journal", "format": "YYYY-MM-DD"}
    (obsidian_dir / "daily-notes.json").write_text(json.dumps(config), encoding="utf-8")
    (vault_dir / "journal").mkdir()
    (vault_dir / ".trash").mkdir()
    return str(vault_dir)


# --- get_daily_note ---

def test_get_daily_note_existing(vault):
    """get_daily_note reads an existing daily note from configured folder."""
    # Fixture vault has daily/2026-05-08.md with daily-notes.json pointing to "daily" folder
    result = get_daily_note(vault, "2026-05-08")
    assert result is not None
    assert result["path"] == "daily/2026-05-08.md"
    assert result["date"] == "2026-05-08"
    assert "Today's daily note" in result["content"]


def test_get_daily_note_missing_returns_none(vault):
    """get_daily_note returns None when the note doesn't exist."""
    result = get_daily_note(vault, "1999-01-01")
    assert result is None


def test_get_daily_note_no_config_defaults(tmp_path):
    """Without a daily-notes.json, falls back to YYYY-MM-DD format in vault root."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "2026-05-08.md").write_text("# Entry\nNo config fallback.", encoding="utf-8")

    result = get_daily_note(str(vault_dir), "2026-05-08")
    assert result is not None
    assert result["path"] == "2026-05-08.md"


def test_get_daily_note_parses_frontmatter(vault):
    """get_daily_note returns parsed frontmatter dict."""
    result = get_daily_note(vault, "2026-05-08")
    assert result is not None
    # Fixture daily note has frontmatter with date field
    assert "date" in result["frontmatter"]


@pytest.mark.parametrize(
    "folder",
    [
        "/absolute-inside-vault",
        "C:\\absolute-inside-vault",
        "C:drive-relative",
        "\\\\server\\share",
        "journal/../outside",
        "journal\\..\\outside",
        ".obsidian/daily",
        ".git/daily",
        ".trash/daily",
    ],
)
def test_get_daily_note_rejects_unsafe_configured_folder(minimal_vault, folder):
    config_path = Path(minimal_vault) / ".obsidian" / "daily-notes.json"
    config_path.write_text(json.dumps({"folder": folder, "format": "YYYY-MM-DD"}), encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        get_daily_note(minimal_vault, "2026-05-08")
    assert str(exc.value) == "Invalid daily note path"


def test_get_daily_note_rejects_absolute_path_that_points_inside_vault(minimal_vault):
    config_path = Path(minimal_vault) / ".obsidian" / "daily-notes.json"
    config_path.write_text(
        json.dumps({"folder": str(Path(minimal_vault) / "journal"), "format": "YYYY-MM-DD"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="^Invalid daily note path$"):
        get_daily_note(minimal_vault, "2026-05-08")


def test_get_daily_note_allows_non_builtin_hidden_folder(minimal_vault):
    hidden = Path(minimal_vault) / ".config"
    hidden.mkdir()
    (hidden / "2026-05-08.md").write_text("# permitted", encoding="utf-8")
    config_path = Path(minimal_vault) / ".obsidian" / "daily-notes.json"
    config_path.write_text(json.dumps({"folder": ".config", "format": "YYYY-MM-DD"}), encoding="utf-8")

    result = get_daily_note(minimal_vault, "2026-05-08")
    assert result is not None
    assert result["path"] == ".config/2026-05-08.md"


def test_get_daily_note_symlink_cycle_has_content_free_error(minimal_vault):
    loop = Path(minimal_vault) / "loop"
    try:
        loop.symlink_to(loop, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    try:
        loop.resolve()
    except RuntimeError:
        pass
    else:
        pytest.skip("this Python version does not raise RuntimeError for symlink cycles")
    config_path = Path(minimal_vault) / ".obsidian" / "daily-notes.json"
    config_path.write_text(
        json.dumps({"folder": "loop", "format": "YYYY-MM-DD"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as caught:
        get_daily_note(minimal_vault, "2026-05-08")

    assert str(caught.value) == "Invalid daily note path"
    assert minimal_vault not in str(caught.value)


def test_daily_config_symlink_cycle_falls_back_without_path_leak(tmp_path):
    vault_dir = tmp_path / "vault"
    config_dir = vault_dir / ".obsidian"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "daily-notes.json"
    try:
        config_path.symlink_to(config_path)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    assert get_daily_note(str(vault_dir), "2026-05-08") is None


def test_malformed_daily_config_falls_back_to_defaults(tmp_path):
    vault_dir = tmp_path / "vault"
    (vault_dir / ".obsidian").mkdir(parents=True)
    (vault_dir / ".obsidian" / "daily-notes.json").write_text("[]", encoding="utf-8")
    (vault_dir / "2026-05-08.md").write_text("# fallback", encoding="utf-8")

    result = get_daily_note(str(vault_dir), "2026-05-08")
    assert result is not None
    assert result["path"] == "2026-05-08.md"


# --- create_daily_note ---

@pytest.mark.asyncio
async def test_create_daily_note_creates_at_config_path(minimal_vault):
    """create_daily_note creates a note at the configured daily folder."""
    path = await create_daily_note(minimal_vault, "2026-06-15", content="Today's thoughts")
    assert path == "journal/2026-06-15.md"
    note_file = Path(minimal_vault) / "journal" / "2026-06-15.md"
    assert note_file.exists()
    content = note_file.read_text(encoding="utf-8")
    assert "Today's thoughts" in content


@pytest.mark.asyncio
async def test_create_daily_note_fails_if_exists(vault):
    """create_daily_note raises FileExistsError if note already exists."""
    # Fixture already has daily/2026-05-08.md
    with pytest.raises(FileExistsError):
        await create_daily_note(vault, "2026-05-08", content="duplicate")


@pytest.mark.asyncio
async def test_create_daily_note_validates_configured_path_before_write(minimal_vault):
    config_path = Path(minimal_vault) / ".obsidian" / "daily-notes.json"
    config_path.write_text(json.dumps({"folder": "journal/../outside", "format": "YYYY-MM-DD"}), encoding="utf-8")

    with pytest.raises(ValueError, match="^Invalid daily note path$"):
        await create_daily_note(minimal_vault, "2026-06-16", content="must not write")
    assert not (Path(minimal_vault) / "outside" / "2026-06-16.md").exists()


@pytest.mark.asyncio
async def test_create_daily_note_with_template(minimal_vault):
    """create_daily_note uses template file and substitutes {{date}}."""
    template_path = Path(minimal_vault) / "template.md"
    template_path.write_text("# {{date}}\n\nTemplate content.", encoding="utf-8")

    path = await create_daily_note(
        minimal_vault,
        "2026-07-04",
        template_path="template.md",
    )
    note_file = Path(minimal_vault) / path
    content = note_file.read_text(encoding="utf-8")
    assert "2026-07-04" in content
    assert "Template content." in content
    # {{date}} placeholder should be replaced
    assert "{{date}}" not in content


@pytest.mark.asyncio
async def test_create_daily_note_no_date_uses_today(minimal_vault):
    """create_daily_note with no date arg creates a note for today."""
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    path = await create_daily_note(minimal_vault, content="today entry")
    assert today in path


@pytest.mark.asyncio
async def test_create_daily_note_template_traversal_blocked(vault, tmp_path):
    """template_path must not escape the vault boundary (plan Phase 0.3 / F4)."""
    secret = Path(tmp_path) / "secret.md"
    secret.write_text("TOP SECRET OUTSIDE VAULT", encoding="utf-8")
    # vault is tmp_path/vault, so ../secret escapes to tmp_path/secret
    with pytest.raises(ValueError) as exc:
        await create_daily_note(vault, date="2025-01-02", template_path="../secret")
    assert "template" in str(exc.value).lower()
    # And the outside content must not have leaked into a created note.
    for p in Path(vault).rglob("*.md"):
        assert "TOP SECRET" not in p.read_text(encoding="utf-8")
