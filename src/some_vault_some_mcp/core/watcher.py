"""Watchdog-based incremental reindexer.

Watches the vault directory for .md file events. Debounces rapid bursts
(multiple saves within DEBOUNCE_SECS collapse into one reindex call).
Runs in a background thread. Errors on individual files are logged and
retried on the next event — watcher does not crash.
"""

import logging
import threading
import time
from pathlib import Path

from some_vault_some_mcp.core.indexer import EXCLUDED_DIRS, incremental_index

logger = logging.getLogger(__name__)

DEBOUNCE_SECS = 2.0
MAX_DELAY_SECS = 30.0  # cap on how long a sustained write burst can starve indexing


class _VaultEventHandler:
    """Collects file events and debounces into a single batched reindex call."""

    def __init__(self, vault_path: str, db_path: str, provider):
        self.vault_path = vault_path
        self.db_path = db_path
        self.provider = provider
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._first_pending: float | None = None

    def _on_event(self, path: str) -> None:
        if not path.lower().endswith(".md"):
            return
        try:
            rel = str(Path(path).relative_to(self.vault_path)).replace("\\", "/")
        except ValueError:
            return
        # Drop excluded-dir events at the door (e.g. the .trash events every
        # soft-delete emits) so they don't trigger a scan + reindex to no-op.
        if any(seg.lower() in EXCLUDED_DIRS for seg in rel.split("/")):
            return
        with self._lock:
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


def start_watcher(vault_path: str, db_path: str, provider) -> None:
    """Start the watchdog filesystem watcher in a daemon thread.

    Returns immediately. Watcher runs until process exits.
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        logger.warning("watchdog not installed — filesystem watcher disabled")
        return

    handler_obj = _VaultEventHandler(vault_path, db_path, provider)

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
    logger.info(f"Vault watcher started on {vault_path}")
