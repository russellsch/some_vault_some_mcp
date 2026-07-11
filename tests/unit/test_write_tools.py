"""Unit tests for write tools: create, append, prepend, move, delete."""

import os
import shutil
from pathlib import Path

import pytest

from some_vault_some_mcp.tools.write import (
    append_to_note,
    create_note,
    delete_note,
    move_note,
    prepend_to_note,
    update_note_frontmatter,
)
from some_vault_some_mcp.tools.read import get_note


FIXTURES = Path(__file__).parent.parent / "fixtures" / "vault"


@pytest.fixture()
def vault(tmp_path):
    """Temporary vault copy for write tests."""
    vault_dir = tmp_path / "vault"
    shutil.copytree(
        str(FIXTURES),
        str(vault_dir),
        ignore=shutil.ignore_patterns(".git", ".trash", ".obsidian"),
    )
    # Ensure .trash dir exists (delete_note uses it)
    (vault_dir / ".trash").mkdir(exist_ok=True)
    return str(vault_dir)


# --- create_note ---

@pytest.mark.asyncio
async def test_create_note_creates_file(vault):
    path = await create_note(vault, "new-note.md", "Hello world")
    assert path == "new-note.md"
    assert (Path(vault) / "new-note.md").exists()
    content = (Path(vault) / "new-note.md").read_text(encoding="utf-8")
    assert "Hello world" in content


@pytest.mark.asyncio
async def test_create_note_adds_md_extension(vault):
    path = await create_note(vault, "auto-ext", "content")
    assert path == "auto-ext.md"
    assert (Path(vault) / "auto-ext.md").exists()


@pytest.mark.asyncio
async def test_create_note_with_frontmatter(vault):
    fm = {"title": "Test", "tags": ["a", "b"]}
    await create_note(vault, "with-fm.md", "Body text", frontmatter=fm)
    content = (Path(vault) / "with-fm.md").read_text(encoding="utf-8")
    assert "title:" in content
    assert "Body text" in content


@pytest.mark.asyncio
async def test_create_note_round_trip(vault):
    """create_note + get_note round-trip: content reads back correctly."""
    body = "# My Note\n\nSome content here."
    await create_note(vault, "round-trip.md", body)
    note = get_note(vault, "round-trip.md")
    assert note is not None
    assert "Some content here." in note.content


@pytest.mark.asyncio
async def test_create_note_existing_path_fails(vault):
    """create_note on an existing path raises FileExistsError."""
    await create_note(vault, "once.md", "first")
    with pytest.raises(FileExistsError):
        await create_note(vault, "once.md", "second")


@pytest.mark.asyncio
async def test_create_note_in_subdir(vault):
    """create_note creates parent directories as needed."""
    path = await create_note(vault, "subdir/nested/note.md", "nested content")
    assert (Path(vault) / "subdir" / "nested" / "note.md").exists()


# --- append_to_note ---

@pytest.mark.asyncio
async def test_append_to_note(vault):
    await create_note(vault, "append-target.md", "Original line")
    await append_to_note(vault, "append-target.md", "Appended line")
    content = (Path(vault) / "append-target.md").read_text(encoding="utf-8")
    assert "Original line" in content
    assert "Appended line" in content
    # Appended content comes after original
    assert content.index("Original line") < content.index("Appended line")


@pytest.mark.asyncio
async def test_append_to_note_no_double_newline(vault):
    """If file ends with newline, no extra blank line is inserted."""
    await create_note(vault, "append-nl.md", "Line one\n")
    await append_to_note(vault, "append-nl.md", "Line two")
    content = (Path(vault) / "append-nl.md").read_text(encoding="utf-8")
    assert content == "Line one\nLine two"


@pytest.mark.asyncio
async def test_append_to_missing_note_fails(vault):
    with pytest.raises(FileNotFoundError):
        await append_to_note(vault, "does-not-exist.md", "content")


# --- prepend_to_note ---

@pytest.mark.asyncio
async def test_prepend_to_note_no_frontmatter(vault):
    """prepend inserts before body when no frontmatter exists."""
    await create_note(vault, "prepend-target.md", "Body line")
    await prepend_to_note(vault, "prepend-target.md", "Prepended line")
    content = (Path(vault) / "prepend-target.md").read_text(encoding="utf-8")
    assert content.index("Prepended line") < content.index("Body line")


@pytest.mark.asyncio
async def test_prepend_to_note_after_frontmatter(vault):
    """prepend inserts after frontmatter block."""
    fm = {"title": "FM Note"}
    await create_note(vault, "prepend-fm.md", "Original body", frontmatter=fm)
    await prepend_to_note(vault, "prepend-fm.md", "Inserted text")
    content = (Path(vault) / "prepend-fm.md").read_text(encoding="utf-8")
    # Frontmatter block must come before the inserted text
    assert "---" in content
    fm_end = content.index("---", 3) + 3  # second ---
    inserted_pos = content.index("Inserted text")
    body_pos = content.index("Original body")
    assert inserted_pos > fm_end
    assert inserted_pos < body_pos


# --- delete_note ---

@pytest.mark.asyncio
async def test_delete_note_soft(vault):
    """Soft delete moves file to .trash."""
    await create_note(vault, "to-delete.md", "going away")
    assert (Path(vault) / "to-delete.md").exists()
    await delete_note(vault, "to-delete.md", permanent=False)
    assert not (Path(vault) / "to-delete.md").exists()
    assert (Path(vault) / ".trash" / "to-delete.md").exists()


@pytest.mark.asyncio
async def test_delete_note_permanent(vault):
    """Permanent delete removes file entirely."""
    await create_note(vault, "perm-delete.md", "gone")
    await delete_note(vault, "perm-delete.md", permanent=True)
    assert not (Path(vault) / "perm-delete.md").exists()
    assert not (Path(vault) / ".trash" / "perm-delete.md").exists()


@pytest.mark.asyncio
async def test_delete_note_soft_delete_is_permanent(vault, monkeypatch):
    """When config forces permanent, soft delete hard-deletes instead of trashing."""
    monkeypatch.setenv("VAULT_SOFT_DELETE_IS_PERMANENT", "true")
    from some_vault_some_mcp.config import load_config
    cfg = load_config()
    assert cfg.soft_delete_is_permanent is True

    await create_note(vault, "cfg-delete.md", "should be hard deleted")
    # Simulate what server.py does: override permanent based on config
    effective = False or cfg.soft_delete_is_permanent
    await delete_note(vault, "cfg-delete.md", permanent=effective)
    assert not (Path(vault) / "cfg-delete.md").exists()
    assert not (Path(vault) / ".trash" / "cfg-delete.md").exists()


@pytest.mark.asyncio
async def test_delete_note_missing_fails(vault):
    with pytest.raises(FileNotFoundError):
        await delete_note(vault, "phantom.md")


# --- move_note ---

@pytest.mark.asyncio
async def test_move_note_basic(vault):
    await create_note(vault, "old-name.md", "move me")
    result = await move_note(vault, "old-name.md", "new-name.md", update_links=False)
    assert not (Path(vault) / "old-name.md").exists()
    assert (Path(vault) / "new-name.md").exists()
    assert "updated_referrers" in result


@pytest.mark.asyncio
async def test_move_note_update_links(vault):
    """move_note with update_links=True rewrites wikilinks in referencing notes."""
    # Create a note that links to the one we'll move
    await create_note(vault, "source-note.md", "See [[target-note]] for details.")
    await create_note(vault, "target-note.md", "I am the target.")

    result = await move_note(vault, "target-note.md", "renamed-target.md", update_links=True)

    # Source note should have its link updated
    source_content = (Path(vault) / "source-note.md").read_text(encoding="utf-8")
    assert "[[renamed-target]]" in source_content or "[[renamed-target.md]]" in source_content
    assert result["updated_referrers"] == ["source-note.md"]


@pytest.mark.asyncio
async def test_move_note_rewrites_case_insensitive_links(vault):
    """Links with different casing than the filename should still be rewritten."""
    await create_note(vault, "MyNote.md", "Target note content.")
    await create_note(vault, "ref.md", "See [[mynote]] for info.")

    result = await move_note(vault, "MyNote.md", "renamed.md", update_links=True)

    ref_content = (Path(vault) / "ref.md").read_text(encoding="utf-8")
    assert "[[renamed]]" in ref_content
    assert "ref.md" in result["updated_referrers"]


@pytest.mark.asyncio
async def test_move_note_missing_source_fails(vault):
    with pytest.raises(FileNotFoundError):
        await move_note(vault, "phantom.md", "dest.md")


@pytest.mark.asyncio
async def test_move_note_existing_dest_fails(vault):
    await create_note(vault, "src.md", "source")
    await create_note(vault, "dst.md", "existing dest")
    with pytest.raises(FileExistsError):
        await move_note(vault, "src.md", "dst.md", update_links=False)


@pytest.mark.asyncio
async def test_delete_note_soft_collision_keeps_both(vault):
    """A repeat soft-delete of the same name must not clobber the prior trash copy
    (plan Phase 0.5 / O7/F7)."""
    from some_vault_some_mcp.tools.write import create_note
    await create_note(vault, "dup.md", "FIRST")
    await delete_note(vault, "dup.md", permanent=False)
    await create_note(vault, "dup.md", "SECOND")
    await delete_note(vault, "dup.md", permanent=False)

    trash = Path(vault) / ".trash"
    contents = sorted(p.read_text(encoding="utf-8") for p in trash.glob("dup*.md"))
    assert contents == ["FIRST", "SECOND"], f"trash lost a copy: {contents}"


@pytest.mark.asyncio
async def test_prepend_preserves_frontmatter_block(vault):
    from some_vault_some_mcp.tools.write import prepend_to_note
    full = Path(vault) / "cm.md"
    full.write_text("---\ntitle: T\n# keepcomment\n---\nOriginal body.", encoding="utf-8")
    await prepend_to_note(vault, "cm.md", "INSERTED")
    text = full.read_text()
    assert "# keepcomment" in text                       # comment survived prepend
    assert text.index("INSERTED") < text.index("Original body.")


@pytest.mark.asyncio
async def test_create_note_blocked_suffix_rejected(vault):
    from some_vault_some_mcp.tools.write import create_note
    with pytest.raises(ValueError):
        await create_note(vault, "x.backup", "data",
                          blocked_suffixes=[".backup"], blocked_message="nope")
    assert not (Path(vault) / "x.backup").exists()


@pytest.mark.asyncio
async def test_move_note_rewrites_all_referrer_forms(tmp_path):
    from some_vault_some_mcp.tools.write import create_note, move_note
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Archive" / "Projects").mkdir(parents=True)
    (vault / "Archive" / "Projects" / "Note.md").write_text(
        "---\naliases: [mynickname]\n---\n## Section\nHi", encoding="utf-8")
    (vault / "ref_exact.md").write_text("[[Archive/Projects/Note]] and [[Note]]", encoding="utf-8")
    (vault / "ref_suffix.md").write_text("see [[Projects/Note]]", encoding="utf-8")
    (vault / "ref_heading.md").write_text("go [[Note#Section]]", encoding="utf-8")
    (vault / "ref_embed.md").write_text("![[Note]]", encoding="utf-8")
    (vault / "ref_alias.md").write_text("[[mynickname]]", encoding="utf-8")

    result = await move_note(str(vault), "Archive/Projects/Note.md", "Moved/Renamed.md")

    def read(n): return (vault / n).read_text()
    assert read("ref_exact.md") == "[[Moved/Renamed]] and [[Moved/Renamed]]"
    assert read("ref_suffix.md") == "see [[Moved/Renamed]]"
    assert read("ref_heading.md") == "go [[Moved/Renamed#Section]]"
    assert read("ref_embed.md") == "![[Moved/Renamed]]"
    assert read("ref_alias.md") == "[[mynickname]]"          # alias untouched
    assert "ref_alias.md" in result["skipped_alias_referrers"]
    assert result["failed_referrers"] == []
    assert set(result["updated_referrers"]) == {
        "ref_exact.md", "ref_suffix.md", "ref_heading.md", "ref_embed.md"}
