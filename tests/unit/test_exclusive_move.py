"""Focused coverage for native no-replace note moves."""

import errno
import os
from pathlib import Path

import pytest

from some_vault_some_mcp.core import exclusive_move as moves
from some_vault_some_mcp.core.exclusive_move import (
    ExclusiveMoveUnavailableError,
    MoveRecoveryRequiredError,
    exclusive_move,
)
from some_vault_some_mcp.tools import write


def test_native_move_works_on_supported_host(tmp_path):
    source = tmp_path / "source.md"
    destination = tmp_path / "destination.md"
    source.write_text("source", encoding="utf-8")

    exclusive_move(source, destination, tmp_path)

    assert not source.exists()
    assert destination.read_text(encoding="utf-8") == "source"


def test_native_adapter_rejects_existing_destination(tmp_path):
    source = tmp_path / "native-source.md"
    destination = tmp_path / "native-destination.md"
    source.write_text("source", encoding="utf-8")
    destination.write_text("destination", encoding="utf-8")

    with pytest.raises(FileExistsError):
        moves._native_exclusive_move(source, destination)

    assert source.read_text(encoding="utf-8") == "source"
    assert destination.read_text(encoding="utf-8") == "destination"


def test_real_case_only_move_on_case_insensitive_filesystem(tmp_path):
    source = tmp_path / "NativeCaseProbe.md"
    destination = tmp_path / "nativecaseprobe.md"
    source.write_text("content", encoding="utf-8")
    if not destination.exists():
        pytest.skip("filesystem is case-sensitive")

    exclusive_move(source, destination, tmp_path)

    names = {entry.name for entry in tmp_path.iterdir()}
    assert destination.read_text(encoding="utf-8") == "content"
    assert destination.name in names
    assert source.name not in names


def test_existing_distinct_destination_preserves_both(tmp_path):
    source = tmp_path / "source.md"
    destination = tmp_path / "destination.md"
    source.write_text("source", encoding="utf-8")
    destination.write_text("destination", encoding="utf-8")

    with pytest.raises(FileExistsError):
        exclusive_move(source, destination, tmp_path)

    assert source.read_text(encoding="utf-8") == "source"
    assert destination.read_text(encoding="utf-8") == "destination"


def test_distinct_case_collision_is_not_mistaken_for_alias(tmp_path, monkeypatch):
    source = tmp_path / "Note.md"
    destination = tmp_path / "note.md"
    source.write_text("source", encoding="utf-8")
    monkeypatch.setattr(
        moves,
        "_identity",
        lambda path: (1, 1) if path.name == "Note.md" else (1, 2),
    )

    with pytest.raises(FileExistsError):
        exclusive_move(source, destination, tmp_path)

    assert source.read_text(encoding="utf-8") == "source"


def test_destination_race_is_reported_and_source_is_preserved(tmp_path, monkeypatch):
    source = tmp_path / "source.md"
    destination = tmp_path / "destination.md"
    source.write_text("source", encoding="utf-8")

    def race(_source, raced_destination):
        raced_destination.write_text("racer", encoding="utf-8")
        raise FileExistsError(errno.EEXIST, "raced")

    monkeypatch.setattr(moves, "_native_exclusive_move", race)

    with pytest.raises(FileExistsError):
        exclusive_move(source, destination, tmp_path)

    assert source.read_text(encoding="utf-8") == "source"
    assert destination.read_text(encoding="utf-8") == "racer"


def test_existing_hard_link_fails_closed_and_preserves_both(tmp_path):
    source = tmp_path / "source.md"
    destination = tmp_path / "destination.md"
    source.write_text("content", encoding="utf-8")
    os.link(source, destination)

    with pytest.raises(ExclusiveMoveUnavailableError, match="hard link"):
        exclusive_move(source, destination, tmp_path)

    assert source.read_text(encoding="utf-8") == "content"
    assert destination.read_text(encoding="utf-8") == "content"


def test_hard_link_destination_replacement_race_preserves_source(tmp_path, monkeypatch):
    source = tmp_path / "source.md"
    destination = tmp_path / "destination.md"
    source.write_text("source", encoding="utf-8")
    os.link(source, destination)

    def replace_destination(_path):
        destination.unlink()
        destination.write_text("racer", encoding="utf-8")
        return True

    monkeypatch.setattr(moves, "_has_exact_leaf", replace_destination)

    with pytest.raises(ExclusiveMoveUnavailableError, match="hard link"):
        exclusive_move(source, destination, tmp_path)

    assert source.read_text(encoding="utf-8") == "source"
    assert destination.read_text(encoding="utf-8") == "racer"


def test_exact_existing_path_is_noop(tmp_path, monkeypatch):
    source = tmp_path / "source.md"
    source.write_text("content", encoding="utf-8")
    monkeypatch.setattr(
        moves,
        "_native_exclusive_move",
        lambda *_args: pytest.fail("exact no-op must not invoke native move"),
    )

    exclusive_move(source, source, tmp_path)

    assert source.read_text(encoding="utf-8") == "content"


def test_case_only_alias_uses_two_exclusive_legs(tmp_path, monkeypatch):
    source = tmp_path / "Note.md"
    destination = tmp_path / "note.md"
    source.write_text("content", encoding="utf-8")
    calls = []

    monkeypatch.setattr(moves, "_identity", lambda _path: (1, 2))
    monkeypatch.setattr(moves, "_same_parent", lambda *_paths: True)
    monkeypatch.setattr(moves, "_has_exact_leaf", lambda _path: False)

    def native(old, new):
        calls.append((old, new))
        old.rename(new)

    monkeypatch.setattr(moves, "_native_exclusive_move", native)

    exclusive_move(source, destination, tmp_path)

    assert len(calls) == 2
    assert calls[0][0] == source
    assert calls[0][1].parent == source.parent
    assert calls[1] == (calls[0][1], destination)
    assert destination.read_text(encoding="utf-8") == "content"


def test_second_leg_failure_rolls_back_to_source(tmp_path, monkeypatch):
    source = tmp_path / "Note.md"
    destination = tmp_path / "note.md"
    source.write_text("content", encoding="utf-8")
    calls = []

    monkeypatch.setattr(moves, "_identity", lambda _path: (1, 2))
    monkeypatch.setattr(moves, "_same_parent", lambda *_paths: True)
    monkeypatch.setattr(moves, "_has_exact_leaf", lambda _path: False)

    def native(old, new):
        calls.append((old, new))
        if len(calls) == 2:
            raise FileExistsError(errno.EEXIST, "raced")
        old.rename(new)

    monkeypatch.setattr(moves, "_native_exclusive_move", native)

    with pytest.raises(FileExistsError):
        exclusive_move(source, destination, tmp_path)

    assert len(calls) == 3
    assert calls[2] == (calls[0][1], source)
    assert source.read_text(encoding="utf-8") == "content"


def test_failed_rollback_preserves_temp_and_reports_recovery_path(tmp_path, monkeypatch):
    source = tmp_path / "folder" / "Note.md"
    destination = tmp_path / "folder" / "note.md"
    source.parent.mkdir()
    source.write_text("content", encoding="utf-8")
    calls = []

    monkeypatch.setattr(moves, "_identity", lambda _path: (1, 2))
    monkeypatch.setattr(moves, "_same_parent", lambda *_paths: True)
    monkeypatch.setattr(moves, "_has_exact_leaf", lambda _path: False)

    def native(old, new):
        calls.append((old, new))
        if len(calls) == 1:
            old.rename(new)
        elif len(calls) == 2:
            raise FileExistsError(errno.EEXIST, "second leg raced")
        else:
            raise FileExistsError(errno.EEXIST, "source path raced")

    monkeypatch.setattr(moves, "_native_exclusive_move", native)

    with pytest.raises(MoveRecoveryRequiredError) as caught:
        exclusive_move(source, destination, tmp_path)

    recovery = tmp_path / caught.value.recovery_path
    assert caught.value.recovery_path.startswith("folder/")
    assert recovery == calls[0][1]
    assert recovery.read_text(encoding="utf-8") == "content"


def test_unsupported_platform_fails_closed(tmp_path, monkeypatch):
    source = tmp_path / "source.md"
    destination = tmp_path / "destination.md"
    source.write_text("content", encoding="utf-8")
    monkeypatch.setattr(moves.sys, "platform", "unsupported-os")

    with pytest.raises(ExclusiveMoveUnavailableError, match="unsupported platform"):
        exclusive_move(source, destination, tmp_path)

    assert source.exists()
    assert not destination.exists()


def test_linux_missing_api_fails_closed(tmp_path, monkeypatch):
    class NoRenameAt2:
        pass

    source = tmp_path / "source.md"
    destination = tmp_path / "destination.md"
    source.write_text("content", encoding="utf-8")
    monkeypatch.setattr(moves.sys, "platform", "linux")
    monkeypatch.setattr(moves.ctypes, "CDLL", lambda *_args, **_kwargs: NoRenameAt2())

    with pytest.raises(ExclusiveMoveUnavailableError, match="renameat2"):
        exclusive_move(source, destination, tmp_path)

    assert source.exists()
    assert not destination.exists()


def test_unsupported_filesystem_errno_is_clear(monkeypatch, tmp_path):
    monkeypatch.setattr(moves.ctypes, "get_errno", lambda: errno.EOPNOTSUPP)

    with pytest.raises(ExclusiveMoveUnavailableError, match="unavailable"):
        moves._raise_posix_failure("test adapter", tmp_path / "a", tmp_path / "b")


@pytest.mark.parametrize(
    ("adapter", "symbol"),
    [
        (moves._linux_rename_noreplace, "renameat2"),
        (moves._macos_rename_exclusive, "renamex_np"),
    ],
)
def test_posix_adapters_map_native_collision(
    monkeypatch, tmp_path, adapter, symbol
):
    calls = []

    class NativeCall:
        def __call__(self, *args):
            calls.append(args)
            return -1

    native_call = NativeCall()
    library = type("Library", (), {symbol: native_call})()
    monkeypatch.setattr(moves.ctypes, "CDLL", lambda *_args, **_kwargs: library)
    monkeypatch.setattr(moves.ctypes, "get_errno", lambda: errno.EEXIST)

    with pytest.raises(FileExistsError):
        adapter(tmp_path / "source", tmp_path / "destination")

    assert len(calls) == 1


def test_windows_adapter_uses_zero_flags_and_maps_collision(monkeypatch, tmp_path):
    calls = []

    class NativeCall:
        def __call__(self, *args):
            calls.append(args)
            return 0

    native_call = NativeCall()
    library = type("Library", (), {"MoveFileExW": native_call})()
    monkeypatch.setattr(
        moves.ctypes, "WinDLL", lambda *_args, **_kwargs: library, raising=False
    )
    monkeypatch.setattr(moves.ctypes, "get_last_error", lambda: 183, raising=False)
    monkeypatch.setattr(
        moves.ctypes,
        "WinError",
        lambda code: OSError(code, "already exists"),
        raising=False,
    )

    with pytest.raises(FileExistsError):
        moves._windows_move_exclusive(tmp_path / "source", tmp_path / "destination")

    assert calls == [(str(tmp_path / "source"), str(tmp_path / "destination"), 0)]


@pytest.mark.parametrize(
    ("platform", "adapter_name"),
    [
        ("linux", "_linux_rename_noreplace"),
        ("darwin", "_macos_rename_exclusive"),
        ("win32", "_windows_move_exclusive"),
    ],
)
def test_platform_dispatch_uses_only_native_adapter(monkeypatch, tmp_path, platform, adapter_name):
    calls = []
    monkeypatch.setattr(moves.sys, "platform", platform)
    monkeypatch.setattr(moves, adapter_name, lambda old, new: calls.append((old, new)))

    moves._native_exclusive_move(tmp_path / "a", tmp_path / "b")

    assert calls == [(tmp_path / "a", tmp_path / "b")]


@pytest.mark.asyncio
async def test_link_writes_start_only_after_successful_move(tmp_path, monkeypatch):
    source = tmp_path / "source.md"
    destination = tmp_path / "destination.md"
    referrer = tmp_path / "referrer.md"
    source.write_text("target", encoding="utf-8")
    referrer.write_text("See [[source]]", encoding="utf-8")
    moved = False
    real_atomic_write = write.atomic_write

    def perform_move(old, new, _vault):
        nonlocal moved
        assert referrer.read_text(encoding="utf-8") == "See [[source]]"
        Path(old).rename(new)
        moved = True

    async def checked_write(path, content):
        assert moved
        await real_atomic_write(path, content)

    monkeypatch.setattr(write, "exclusive_move", perform_move)
    monkeypatch.setattr(write, "atomic_write", checked_write)

    result = await write.move_note(str(tmp_path), "source.md", "destination.md")

    assert result["updated_referrers"] == ["referrer.md"]
    assert referrer.read_text(encoding="utf-8") == "See [[destination]]"


@pytest.mark.asyncio
async def test_move_note_restores_requested_leaf_after_path_canonicalization(
    tmp_path, monkeypatch
):
    source = tmp_path / "Note.md"
    source.write_text("target", encoding="utf-8")
    seen = []

    def canonicalizing_resolver(_vault, _relative):
        return str(source)

    def capture_move(old, new, _vault):
        seen.append((Path(old), Path(new)))

    monkeypatch.setattr(write, "resolve_vault_path", canonicalizing_resolver)
    monkeypatch.setattr(write, "exclusive_move", capture_move)

    await write.move_note(str(tmp_path), "Note.md", "note.md", update_links=False)

    assert seen == [(source, tmp_path / "note.md")]


@pytest.mark.asyncio
async def test_failed_move_does_not_write_links(tmp_path, monkeypatch):
    source = tmp_path / "source.md"
    referrer = tmp_path / "referrer.md"
    source.write_text("target", encoding="utf-8")
    referrer.write_text("See [[source]]", encoding="utf-8")
    monkeypatch.setattr(
        write,
        "exclusive_move",
        lambda *_args: (_ for _ in ()).throw(FileExistsError(errno.EEXIST, "raced")),
    )
    monkeypatch.setattr(
        write,
        "atomic_write",
        lambda *_args: pytest.fail("link write happened before move success"),
    )

    with pytest.raises(FileExistsError):
        await write.move_note(str(tmp_path), "source.md", "destination.md")

    assert source.read_text(encoding="utf-8") == "target"
    assert referrer.read_text(encoding="utf-8") == "See [[source]]"
