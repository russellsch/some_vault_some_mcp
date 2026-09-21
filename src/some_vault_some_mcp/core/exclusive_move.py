"""Native, no-replace filesystem moves.

This module deliberately has no portable ``os.rename`` fallback.  A move that
cannot prove that it will not replace an existing destination fails closed.
"""

from __future__ import annotations

import ctypes
import errno
import os
import secrets
import sys
from pathlib import Path


class ExclusiveMoveUnavailableError(OSError):
    """The host cannot provide a native atomic no-replace move."""


class MoveRecoveryRequiredError(OSError):
    """A case-only move failed and its temporary file could not be restored."""

    def __init__(self, recovery_path: str, move_error: OSError, rollback_error: OSError):
        self.recovery_path = recovery_path
        self.move_error = move_error
        self.rollback_error = rollback_error
        super().__init__(
            "Case-only move failed and rollback also failed; "
            f"the note is preserved at '{recovery_path}'"
        )


_COLLISION_ERRNOS = {errno.EEXIST, errno.ENOTEMPTY}
_UNAVAILABLE_ERRNOS = {
    errno.ENOSYS,
    errno.EINVAL,
    errno.EXDEV,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}


def _collision(path: Path, error: OSError | None = None) -> FileExistsError:
    result = FileExistsError(errno.EEXIST, f"Destination already exists: {path}", str(path))
    if error is not None:
        result.__cause__ = error
    return result


def _unavailable(operation: str, error: OSError | None = None) -> ExclusiveMoveUnavailableError:
    detail = f": {error}" if error is not None else ""
    result = ExclusiveMoveUnavailableError(
        getattr(error, "errno", errno.ENOTSUP),
        f"Safe exclusive move is unavailable ({operation}){detail}",
    )
    if error is not None:
        result.__cause__ = error
    return result


def _raise_posix_failure(operation: str, source: Path, destination: Path) -> None:
    code = ctypes.get_errno()
    error = OSError(code, os.strerror(code), str(source), str(destination))
    if code in _COLLISION_ERRNOS:
        raise _collision(destination, error)
    if code in _UNAVAILABLE_ERRNOS:
        raise _unavailable(operation, error)
    raise error


def _linux_rename_noreplace(source: Path, destination: Path) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as error:
        raise _unavailable("Linux renameat2(RENAME_NOREPLACE)", error) from error

    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result != 0:
        _raise_posix_failure("Linux renameat2(RENAME_NOREPLACE)", source, destination)


def _macos_rename_exclusive(source: Path, destination: Path) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = libc.renamex_np
    except (AttributeError, OSError) as error:
        raise _unavailable("macOS renamex_np(RENAME_EXCL)", error) from error

    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    rename_excl = 0x00000004
    result = renamex_np(os.fsencode(source), os.fsencode(destination), rename_excl)
    if result != 0:
        _raise_posix_failure("macOS renamex_np(RENAME_EXCL)", source, destination)


def _windows_move_exclusive(source: Path, destination: Path) -> None:
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move_file_ex = kernel32.MoveFileExW
    except (AttributeError, OSError) as error:
        raise _unavailable("Windows MoveFileExW(no replace)", error) from error

    move_file_ex.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    move_file_ex.restype = ctypes.c_int
    if move_file_ex(str(source), str(destination), 0):
        return

    code = ctypes.get_last_error()
    error = ctypes.WinError(code)
    if code in {80, 183}:  # ERROR_FILE_EXISTS, ERROR_ALREADY_EXISTS
        raise _collision(destination, error)
    if code in {1, 17, 50, 120}:  # invalid function, cross-device, unsupported, not implemented
        raise _unavailable("Windows MoveFileExW(no replace)", error)
    raise error


def _native_exclusive_move(source: Path, destination: Path) -> None:
    if sys.platform.startswith("linux"):
        _linux_rename_noreplace(source, destination)
    elif sys.platform == "darwin":
        _macos_rename_exclusive(source, destination)
    elif sys.platform == "win32":
        _windows_move_exclusive(source, destination)
    else:
        raise _unavailable(f"unsupported platform {sys.platform!r}")


def _identity(path: Path) -> tuple[int, int]:
    stat = path.lstat()
    return stat.st_dev, stat.st_ino


def _same_parent(source: Path, destination: Path) -> bool:
    try:
        source_stat = source.parent.stat()
        destination_stat = destination.parent.stat()
    except OSError:
        return False
    return (source_stat.st_dev, source_stat.st_ino) == (
        destination_stat.st_dev,
        destination_stat.st_ino,
    )


def _has_exact_leaf(path: Path) -> bool:
    """Return whether the directory contains the requested spelling exactly."""
    try:
        with os.scandir(path.parent) as entries:
            return any(entry.name == path.name for entry in entries)
    except FileNotFoundError:
        return False


def _relative_recovery_path(path: Path, vault_root: Path) -> str:
    try:
        return path.relative_to(vault_root).as_posix()
    except ValueError:
        # Callers should always pass paths inside the vault.  Keep the exception
        # useful and avoid concealing the recovery location if that invariant is
        # broken.
        return str(path)


def _case_only_move(source: Path, destination: Path, vault_root: Path) -> None:
    for _ in range(16):
        temporary = source.parent / f".{source.name}.move-{secrets.token_hex(8)}.tmp"
        try:
            _native_exclusive_move(source, temporary)
            break
        except FileExistsError:
            continue
    else:
        raise ExclusiveMoveUnavailableError(
            errno.EEXIST, "Could not allocate a unique temporary path for case-only move"
        )

    try:
        _native_exclusive_move(temporary, destination)
    except OSError as move_error:
        try:
            _native_exclusive_move(temporary, source)
        except OSError as rollback_error:
            raise MoveRecoveryRequiredError(
                _relative_recovery_path(temporary, vault_root),
                move_error,
                rollback_error,
            ) from rollback_error
        raise


def exclusive_move(source: str | Path, destination: str | Path, vault_root: str | Path) -> None:
    """Move one directory entry without ever replacing another.

    Existing entries are classified by filesystem identity.  A hard-link at the
    destination already represents the requested file, so only the source entry
    is removed.  A case-insensitive alias needs a temporary spelling so the host
    can perform the case-only rename safely.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    vault_path = Path(vault_root).resolve()

    try:
        source_identity = _identity(source_path)
    except FileNotFoundError:
        raise FileNotFoundError(errno.ENOENT, f"Source does not exist: {source_path}", str(source_path))

    try:
        destination_identity = _identity(destination_path)
    except FileNotFoundError:
        _native_exclusive_move(source_path, destination_path)
        return

    if source_identity != destination_identity:
        raise _collision(destination_path)

    parents_match = _same_parent(source_path, destination_path)
    if parents_match and source_path.name == destination_path.name:
        return

    if _has_exact_leaf(destination_path):
        # A separately named directory entry with the same identity is a hard
        # link.  There is no portable identity-conditional unlink: another
        # participant could replace the destination after the lstat checks and
        # make unlinking source destroy the last link to the original note.
        raise ExclusiveMoveUnavailableError(
            errno.ENOTSUP,
            "Cannot safely move onto a separately named hard link",
        )


    if parents_match:
        _case_only_move(source_path, destination_path, vault_path)
        return

    raise ExclusiveMoveUnavailableError(
        errno.ENOTSUP,
        "Cannot safely classify equal source and destination identities across different directories",
    )
