"""Index publication lock tests."""

import multiprocessing
import threading
from pathlib import Path

import pytest

from some_vault_some_mcp.core.embeddings import MockProvider
from some_vault_some_mcp.core.indexer import full_index, incremental_index, _get_db, _get_table, TABLE_NAME

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "vault"


def _hold_process_lock(db_path, acquired, release):
    from some_vault_some_mcp.core.indexer import _database_lock

    with _database_lock(db_path):
        acquired.set()
        release.wait(10)


def _report_process_lock(db_path, acquired):
    from some_vault_some_mcp.core.indexer import _database_lock

    with _database_lock(db_path):
        acquired.set()


def _hold_generation_reader(db_path, acquired, release):
    from some_vault_some_mcp.core.indexer import _generation_cutover_lock

    with _generation_cutover_lock(db_path, exclusive=False):
        acquired.set()
        release.wait(10)


def _report_generation_writer(db_path, acquired):
    from some_vault_some_mcp.core.indexer import _generation_cutover_lock

    with _generation_cutover_lock(db_path, exclusive=True):
        acquired.set()


def _crash_with_generation_reader(db_path, acquired):
    import os
    from some_vault_some_mcp.core.indexer import _generation_cutover_lock

    with _generation_cutover_lock(db_path, exclusive=False):
        acquired.set()
        os._exit(0)


def test_database_publication_lock_serializes_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    first_acquired = context.Event()
    release_first = context.Event()
    second_acquired = context.Event()
    db_path = str(tmp_path / "db.lance")

    first = context.Process(
        target=_hold_process_lock,
        args=(db_path, first_acquired, release_first),
    )
    second = context.Process(
        target=_report_process_lock,
        args=(db_path, second_acquired),
    )
    first.start()
    try:
        assert first_acquired.wait(5)
        second.start()
        assert not second_acquired.wait(0.3)
        release_first.set()
        assert second_acquired.wait(5)
    finally:
        release_first.set()
        first.join(timeout=5)
        if second.pid is not None:
            second.join(timeout=5)

    assert first.exitcode == 0
    assert second.exitcode == 0


def test_generation_writer_waits_for_cross_process_reader(tmp_path):
    context = multiprocessing.get_context("spawn")
    reader_acquired = context.Event()
    release_reader = context.Event()
    writer_acquired = context.Event()
    db_path = str(tmp_path / "db.lance")

    reader = context.Process(
        target=_hold_generation_reader,
        args=(db_path, reader_acquired, release_reader),
    )
    writer = context.Process(
        target=_report_generation_writer,
        args=(db_path, writer_acquired),
    )
    reader.start()
    try:
        assert reader_acquired.wait(5)
        writer.start()
        assert not writer_acquired.wait(0.3)
        release_reader.set()
        assert writer_acquired.wait(5)
    finally:
        release_reader.set()
        reader.join(timeout=5)
        if writer.pid is not None:
            writer.join(timeout=5)

    assert reader.exitcode == 0
    assert writer.exitcode == 0


def test_generation_reader_lock_is_released_when_process_exits(tmp_path):
    context = multiprocessing.get_context("spawn")
    reader_acquired = context.Event()
    writer_acquired = context.Event()
    db_path = str(tmp_path / "db.lance")

    reader = context.Process(
        target=_crash_with_generation_reader,
        args=(db_path, reader_acquired),
    )
    reader.start()
    assert reader_acquired.wait(5)
    reader.join(timeout=5)
    assert reader.exitcode == 0

    writer = context.Process(
        target=_report_generation_writer,
        args=(db_path, writer_acquired),
    )
    writer.start()
    writer.join(timeout=5)

    assert writer_acquired.is_set()
    assert writer.exitcode == 0


@pytest.mark.integration
def test_concurrent_reindex_no_duplicates(tmp_path):
    """Two threads calling incremental_index for the same file produce no duplicates."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    note = vault_dir / "note.md"
    note.write_text("# Test\n\nSome content here.", encoding="utf-8")

    db_path = str(tmp_path / "db.lance")
    provider = MockProvider()

    result = full_index(str(vault_dir), db_path, provider)
    assert result["chunks_created"] > 0

    # Touch the file to trigger re-embed
    note.write_text("# Test\n\nUpdated content here.", encoding="utf-8")

    errors = []

    def reindex():
        try:
            incremental_index(str(vault_dir), db_path, provider, single_file="note.md")
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=reindex)
    t2 = threading.Thread(target=reindex)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Reindex raised: {errors}"

    db = _get_db(db_path)
    table = _get_table(db)
    import pandas as pd
    df = table.to_pandas()
    note_chunks = df[df["file_path"] == "note.md"]
    unique_chunks = note_chunks.drop_duplicates(subset=["content"])
    assert len(note_chunks) == len(unique_chunks), (
        f"Found {len(note_chunks)} chunks but only {len(unique_chunks)} unique — duplicates exist"
    )
