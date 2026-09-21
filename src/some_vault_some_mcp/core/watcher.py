"""Watchdog-based incremental reindexer.

Watches the vault directory for .md file events. Debounces rapid bursts
(multiple saves within DEBOUNCE_SECS collapse into one reindex call).
Runs in a background thread. Errors on individual files are logged and
retained in a cumulative journal; the watcher enters degraded buffering mode
until a staged rebuild succeeds — watcher does not crash.
"""

import logging
import threading
import time
from pathlib import Path

from some_vault_some_mcp.core.indexer import incremental_index
from some_vault_some_mcp.core.paths import is_index_excluded

logger = logging.getLogger(__name__)

DEBOUNCE_SECS = 2.0
MAX_DELAY_SECS = 30.0  # cap on how long a sustained write burst can starve indexing

_watchers: dict[str, "_VaultEventHandler"] = {}
_watchers_lock = threading.Lock()


def _watcher_key(db_path: str) -> str:
    return str(Path(db_path).expanduser().resolve())


def get_watcher(db_path: str) -> "_VaultEventHandler | None":
    with _watchers_lock:
        return _watchers.get(_watcher_key(db_path))


class _VaultEventHandler:
    """Collects file events and debounces into a single batched reindex call."""

    def __init__(self, vault_path: str, db_path: str, provider, *, buffering: bool = False):
        self.vault_path = vault_path
        self.db_path = db_path
        self.provider = provider
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._first_pending: float | None = None
        self._mode = "buffering" if buffering else "normal"
        self._journal: set[str] = set()
        self._cutover_locked = False
        self._observer = None

    @property
    def is_buffering(self) -> bool:
        with self._lock:
            return self._mode == "buffering"

    @property
    def journal_size(self) -> int:
        with self._lock:
            return len(self._journal)

    def begin_buffering(self) -> None:
        """Retain every event until an unpublished generation is committed."""
        with self._lock:
            self._mode = "buffering"
            self._journal.update(self._pending)
            self._pending.clear()
            if self._timer is not None:
                self._timer.cancel()
            self._timer = None
            self._first_pending = None

    def acquire_cutover(self) -> None:
        """Freeze callback delivery across final reconciliation/publication."""
        self._lock.acquire()
        self._cutover_locked = True

    def commit_cutover(self) -> None:
        if not self._cutover_locked:
            raise RuntimeError("watcher cutover lock is not held")
        self._journal.clear()
        self._pending.clear()
        self._timer = None
        self._first_pending = None
        self._mode = "normal"

    def abort_cutover(self) -> None:
        """Keep the full journal and old index in degraded read-only mode."""
        with self._lock:
            self._mode = "buffering"

    def release_cutover(self) -> None:
        if self._cutover_locked:
            self._cutover_locked = False
            self._lock.release()

    def _on_event(self, path: str) -> None:
        if not path.lower().endswith(".md"):
            return
        try:
            rel = str(Path(path).relative_to(self.vault_path)).replace("\\", "/")
        except ValueError:
            return
        # Drop excluded-dir events at the door (e.g. the .trash events every
        # soft-delete emits) so they don't trigger a scan + reindex to no-op.
        if is_index_excluded(rel):
            return
        with self._lock:
            if self._mode == "buffering":
                self._journal.add(rel)
                return
            self._pending.add(rel)
            now = time.monotonic()
            if self._first_pending is None:
                self._first_pending = now
            if self._timer is not None:
                # Once MAX_DELAY has elapsed, stop postponing — let the running
                # timer fire so a sustained burst can't starve indexing forever.
                if now - self._first_pending >= MAX_DELAY_SECS:
                    return
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_SECS, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            paths = set(self._pending)
            self._pending.clear()
            self._timer = None
            self._first_pending = None
        if not paths:
            return
        try:
            # One batched call for the whole burst — one scan, one FTS rebuild.
            result = incremental_index(
                self.vault_path, self.db_path, self.provider,
                only_files=paths,
            )
            logger.info(f"Reindexed {len(paths)} file(s): {result}")
        except Exception as e:
            logger.error(f"Reindex failed for {sorted(paths)}: {e}")
            # Do not discard evidence of changes after a failed mutation. A
            # later full candidate rebuild consumes the cumulative journal.
            with self._lock:
                self._journal.update(paths)
                self._mode = "buffering"
            try:
                from some_vault_some_mcp.core.indexer import _update_runtime_state
                _update_runtime_state(
                    self.db_path,
                    last_rebuild_error=str(e),
                    serving_degraded=True,
                )
            except Exception:
                pass


def start_watcher(
    vault_path: str,
    db_path: str,
    provider,
    *,
    buffering: bool = False,
) -> _VaultEventHandler | None:
    """Start the watchdog filesystem watcher in a daemon thread.

    Returns immediately. Watcher runs until process exits.
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        logger.warning("watchdog not installed — filesystem watcher disabled")
        return None

    handler_obj = _VaultEventHandler(vault_path, db_path, provider, buffering=buffering)

    class _WDHandler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory:
                handler_obj._on_event(event.src_path)

        def on_modified(self, event):
            if not event.is_directory:
                handler_obj._on_event(event.src_path)

        def on_deleted(self, event):
            if not event.is_directory:
                handler_obj._on_event(event.src_path)

        def on_moved(self, event):
            if not event.is_directory:
                handler_obj._on_event(event.src_path)
                handler_obj._on_event(event.dest_path)

    observer = Observer()
    observer.schedule(_WDHandler(), vault_path, recursive=True)
    observer.daemon = True
    observer.start()
    handler_obj._observer = observer
    with _watchers_lock:
        _watchers[_watcher_key(db_path)] = handler_obj
    logger.info(f"Vault watcher started on {vault_path}")
    return handler_obj
