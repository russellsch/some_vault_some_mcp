"""Write tools: create, append, prepend, update_frontmatter, move, delete."""

import logging
import os
import re
from datetime import datetime
from pathlib import Path

from some_vault_some_mcp.core.atomic_write import atomic_write, with_file_lock
from some_vault_some_mcp.core.frontmatter import parse_frontmatter, update_frontmatter, serialize_frontmatter
from some_vault_some_mcp.core.paths import resolve_vault_path, VaultPathError, ensure_md_extension
from some_vault_some_mcp.core.wikilinks import extract_wikilinks, resolve_wikilink, build_alias_map
from some_vault_some_mcp.core.paths import walk_vault

logger = logging.getLogger(__name__)


async def create_note(vault_path: str, path: str, content: str, frontmatter: dict | None = None) -> str:
    """Create a new note. Raises FileExistsError if note already exists."""
    resolved_path = ensure_md_extension(path)
    try:
        full_path = resolve_vault_path(vault_path, resolved_path)
    except VaultPathError as e:
        raise ValueError(str(e))

    final_content = content
    if frontmatter:
        final_content = serialize_frontmatter(frontmatter, content)

    async def _create():
        p = Path(full_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            err = FileExistsError(f"Note already exists at '{resolved_path}'")
            err.errno = 17  # EEXIST
            raise err
        await atomic_write(full_path, final_content)

    await with_file_lock(full_path, _create)
    return resolved_path


async def append_to_note(vault_path: str, path: str, content: str) -> None:
    """Append content to end of existing note."""
    resolved_path = ensure_md_extension(path)
    try:
        full_path = resolve_vault_path(vault_path, resolved_path)
    except VaultPathError as e:
        raise ValueError(str(e))

    async def _append():
        p = Path(full_path)
        if not p.exists():
            raise FileNotFoundError(f"Note not found: {resolved_path}")
        existing = p.read_text(encoding="utf-8", errors="replace")
        separator = "" if existing.endswith("\n") else "\n"
        await atomic_write(full_path, existing + separator + content)

    await with_file_lock(full_path, _append)


async def prepend_to_note(vault_path: str, path: str, content: str) -> None:
    """Insert content after frontmatter, before body."""
    resolved_path = ensure_md_extension(path)
    try:
        full_path = resolve_vault_path(vault_path, resolved_path)
    except VaultPathError as e:
        raise ValueError(str(e))

    async def _prepend():
        p = Path(full_path)
        if not p.exists():
            raise FileNotFoundError(f"Note not found: {resolved_path}")
        existing = p.read_text(encoding="utf-8", errors="replace")
        fm, body = parse_frontmatter(existing)
        if fm:
            # Reconstruct with frontmatter + prepended content + body
            new_content = serialize_frontmatter(fm, content + "\n" + body)
        else:
            new_content = content + "\n" + existing
        await atomic_write(full_path, new_content)

    await with_file_lock(full_path, _prepend)


async def update_note_frontmatter(vault_path: str, path: str, properties: dict) -> int:
    """Merge properties into existing frontmatter. Returns count of properties written."""
    resolved_path = ensure_md_extension(path)
    try:
        full_path = resolve_vault_path(vault_path, resolved_path)
    except VaultPathError as e:
        raise ValueError(str(e))

    async def _update():
        p = Path(full_path)
        if not p.exists():
            raise FileNotFoundError(f"Note not found: {resolved_path}")
        existing = p.read_text(encoding="utf-8", errors="replace")
        new_content = update_frontmatter(existing, properties)
        await atomic_write(full_path, new_content)

    await with_file_lock(full_path, _update)
    return len(properties)


async def delete_note(vault_path: str, path: str, permanent: bool = False) -> None:
    """Delete note. Default: move to .trash. permanent=True: unlink from disk."""
    resolved_path = ensure_md_extension(path)
    try:
        full_path = resolve_vault_path(vault_path, resolved_path)
    except VaultPathError as e:
        raise ValueError(str(e))

    async def _delete():
        p = Path(full_path)
        if not p.exists():
            raise FileNotFoundError(f"Note not found: {resolved_path}")
        if permanent:
            p.unlink()
        else:
            trash_dir = Path(vault_path) / ".trash"
            trash_target = trash_dir / resolved_path
            trash_target.parent.mkdir(parents=True, exist_ok=True)
            # os.rename overwrites — disambiguate on collision so a repeat delete
            # of the same name doesn't destroy the previously trashed copy.
            if trash_target.exists():
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                trash_target = trash_target.with_name(
                    f"{trash_target.stem}.{stamp}{trash_target.suffix}"
                )
            os.rename(full_path, str(trash_target))

    await with_file_lock(full_path, _delete)


async def move_note(
    vault_path: str,
    old_path: str,
    new_path: str,
    update_links: bool = True,
) -> dict:
    """Move/rename a note. Optionally rewrites all referencing wikilinks."""
    resolved_old = ensure_md_extension(old_path)
    resolved_new = ensure_md_extension(new_path)

    try:
        full_old = resolve_vault_path(vault_path, resolved_old)
        full_new = resolve_vault_path(vault_path, resolved_new)
    except VaultPathError as e:
        raise ValueError(str(e))

    if not Path(full_old).exists():
        raise FileNotFoundError(f"Note not found: {resolved_old}")

    if Path(full_new).exists() and full_old.lower() != full_new.lower():
        raise FileExistsError(f"Destination already exists: {resolved_new}")

    # Gather all notes for link rewriting before the move
    all_notes = walk_vault(vault_path) if update_links else []
    note_contents: dict[str, str] = {}
    if update_links:
        for rel in all_notes:
            try:
                note_contents[rel] = (Path(vault_path) / rel).read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
        alias_map = build_alias_map(all_notes, note_contents)

    # Perform the move
    Path(full_new).parent.mkdir(parents=True, exist_ok=True)
    os.rename(full_old, full_new)

    updated_referrers = []
    failed_referrers = []

    if update_links:
        for rel in all_notes:
            if rel == resolved_old:
                continue
            content = note_contents.get(rel, "")
            links = extract_wikilinks(content)
            needs_rewrite = False
            for link in links:
                target_base = link["target"].split("#")[0].strip()
                resolved = resolve_wikilink(target_base, rel, all_notes, alias_map)
                if resolved == resolved_old:
                    needs_rewrite = True
                    break

            if not needs_rewrite:
                continue

            # Simple rewrite: replace [[old_stem]] or [[old_path]] references
            old_stem = Path(resolved_old).stem
            new_stem = Path(resolved_new).stem
            old_no_ext = resolved_old[:-3] if resolved_old.lower().endswith(".md") else resolved_old
            new_no_ext = resolved_new[:-3] if resolved_new.lower().endswith(".md") else resolved_new

            new_content = content
            # Replace path-form links first, then basename-form (case-insensitive)
            new_content = re.sub(re.escape(f"[[{old_no_ext}]]"), f"[[{new_no_ext}]]",
                                 new_content, flags=re.IGNORECASE)
            new_content = re.sub(re.escape(f"[[{old_no_ext}|"), f"[[{new_no_ext}|",
                                 new_content, flags=re.IGNORECASE)
            if old_stem != new_stem:
                new_content = re.sub(re.escape(f"[[{old_stem}]]"), f"[[{new_stem}]]",
                                     new_content, flags=re.IGNORECASE)
                new_content = re.sub(re.escape(f"[[{old_stem}|"), f"[[{new_stem}|",
                                     new_content, flags=re.IGNORECASE)

            if new_content != content:
                try:
                    full_referrer = resolve_vault_path(vault_path, rel)

                    async def _rewrite(full=full_referrer, nc=new_content):
                        await atomic_write(full, nc)

                    await with_file_lock(full_referrer, _rewrite)
                    updated_referrers.append(rel)
                except Exception as e:
                    logger.warning(f"Failed to rewrite links in {rel}: {e}")
                    failed_referrers.append({"path": rel, "error": str(e)})

    return {
        "updated_referrers": updated_referrers,
        "failed_referrers": failed_referrers,
    }
