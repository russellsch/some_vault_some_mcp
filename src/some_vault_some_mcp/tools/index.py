"""Index tools: vault_index_status, vault_reindex."""

import logging

from some_vault_some_mcp.core.indexer import full_index, get_index_status, incremental_index
from some_vault_some_mcp.models import IndexStatus, ReindexResult

logger = logging.getLogger(__name__)


def vault_index_status(vault_path: str, db_path: str, provider_dims: int | None = None) -> IndexStatus:
    info = get_index_status(vault_path, db_path, provider_dims)
    return IndexStatus(**info)


def vault_reindex(
    vault_path: str,
    db_path: str,
    provider,
    single_file: str | None = None,
) -> ReindexResult:
    """Rebuild the vault generation, or incrementally update one file."""
    from some_vault_some_mcp.core.watcher import get_watcher

    watcher = get_watcher(db_path)
    # A failed watcher incremental journals every later event and stays in
    # buffering mode.  A targeted update cannot reconcile that journal, so it
    # must promote to the same staged full rebuild used for recovery.
    if single_file is None or (watcher is not None and watcher.is_buffering):
        result = full_index(
            vault_path=vault_path,
            db_path=db_path,
            provider=provider,
            watcher=watcher,
        )
    else:
        result = incremental_index(
            vault_path=vault_path,
            db_path=db_path,
            provider=provider,
            single_file=single_file,
        )
    return ReindexResult(**result)
