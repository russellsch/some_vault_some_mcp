"""Generation publication, recovery, and content-hash regression tests."""

import json
import os
from pathlib import Path

import pytest

from some_vault_some_mcp.core.embeddings import MockProvider
from some_vault_some_mcp.core.indexer import (
    GENERATION_PREFIX,
    ScannedFile,
    VaultScan,
    _get_db,
    _read_manifest,
    full_index,
    incremental_index,
    resolve_active_table,
)


def _vault(tmp_path, **notes):
    vault = tmp_path / "vault"
    vault.mkdir()
    for name, content in notes.items():
        path = vault / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return vault


def _active_frame(db_path):
    _, table, name, _ = resolve_active_table(str(db_path), 768)
    assert table is not None
    return table.to_pandas(), name


def test_full_index_publishes_generation_manifest_and_empty_vault(tmp_path):
    vault = _vault(tmp_path)
    db_path = tmp_path / "db.lance"
    result = full_index(str(vault), str(db_path), MockProvider())

    manifest = _read_manifest(str(db_path))
    assert manifest is not None
    assert manifest["active"].startswith(GENERATION_PREFIX)
    assert manifest["previous"] is None
    assert manifest["completion"] == "complete"
    assert result["chunks_created"] == 0
    _, table, _, _ = resolve_active_table(str(db_path), 768)
    assert table is not None and table.count_rows() == 0


def test_failed_rebuild_leaves_manifest_and_active_untouched(tmp_path, monkeypatch):
    vault = _vault(tmp_path, **{"note.md": "# Original\n\nold body"})
    db_path = tmp_path / "db.lance"
    full_index(str(vault), str(db_path), MockProvider())
    before_manifest = _read_manifest(str(db_path))
    before, before_name = _active_frame(db_path)
    (vault / "note.md").write_text("# Changed\n\nnew body", encoding="utf-8")

    import some_vault_some_mcp.core.indexer as indexer
    monkeypatch.setattr(indexer, "_publish_manifest", lambda *_: (_ for _ in ()).throw(OSError("publish failed")))
    with pytest.raises(OSError, match="publish failed"):
        full_index(str(vault), str(db_path), MockProvider())

    after, after_name = _active_frame(db_path)
    assert _read_manifest(str(db_path)) == before_manifest
    assert after_name == before_name
    columns = ["file_path", "chunk_index", "content", "content_hash"]
    assert after[columns].to_dict("records") == before[columns].to_dict("records")


def test_same_mtime_changed_content_is_reindexed(tmp_path):
    vault = _vault(tmp_path, **{"note.md": "# Note\n\nold body"})
    db_path = tmp_path / "db.lance"
    full_index(str(vault), str(db_path), MockProvider())
    before_name = _read_manifest(str(db_path))["active"]
    original_mtime = (vault / "note.md").stat().st_mtime
    (vault / "note.md").write_text("# Note\n\nnew body", encoding="utf-8")
    os.utime(vault / "note.md", (original_mtime, original_mtime))

    result = incremental_index(str(vault), str(db_path), MockProvider())
    frame, after_name = _active_frame(db_path)
    assert result["files_indexed"] == 1
    assert after_name != before_name
    assert frame["content"].str.contains("new body").any()
    assert not frame["content"].str.contains("old body").any()


def test_failed_incremental_candidate_never_mutates_active_generation(tmp_path, monkeypatch):
    vault = _vault(tmp_path, **{"note.md": "# Note\n\nold body"})
    db_path = tmp_path / "db.lance"
    full_index(str(vault), str(db_path), MockProvider())
    before_manifest = _read_manifest(str(db_path))
    before, before_name = _active_frame(db_path)
    (vault / "note.md").write_text("# Note\n\nnew body", encoding="utf-8")

    import some_vault_some_mcp.core.indexer as indexer
    real_sync = indexer._sync_table_to_scan

    def fail_after_candidate_mutation(*args, **kwargs):
        real_sync(*args, **kwargs)
        raise OSError("candidate add commit failed")

    monkeypatch.setattr(indexer, "_sync_table_to_scan", fail_after_candidate_mutation)
    with pytest.raises(OSError, match="candidate add commit failed"):
        incremental_index(str(vault), str(db_path), MockProvider())

    after, after_name = _active_frame(db_path)
    assert _read_manifest(str(db_path)) == before_manifest
    assert after_name == before_name
    columns = ["file_path", "chunk_index", "content", "content_hash"]
    assert after[columns].to_dict("records") == before[columns].to_dict("records")


def test_post_publication_cutover_failure_keeps_new_active_generation(tmp_path):
    class FailingCommitWatcher:
        def begin_buffering(self):
            pass

        def acquire_cutover(self):
            pass

        def commit_cutover(self):
            raise OSError("cutover bookkeeping failed")

        def release_cutover(self):
            pass

        def abort_cutover(self):
            pass

    vault = _vault(tmp_path, **{"note.md": "# Note\n\nold body"})
    db_path = tmp_path / "db.lance"
    full_index(str(vault), str(db_path), MockProvider())
    before_name = _read_manifest(str(db_path))["active"]
    (vault / "note.md").write_text("# Note\n\nnew body", encoding="utf-8")

    with pytest.raises(OSError, match="cutover bookkeeping failed"):
        full_index(
            str(vault),
            str(db_path),
            MockProvider(),
            watcher=FailingCommitWatcher(),
        )

    manifest = _read_manifest(str(db_path))
    frame, active_name = _active_frame(db_path)
    assert active_name == manifest["active"]
    assert active_name != before_name
    assert active_name in _get_db(str(db_path)).list_tables().tables
    assert frame["content"].str.contains("new body").any()


@pytest.mark.parametrize("operation", ["full", "incremental"])
def test_visible_manifest_replace_error_never_drops_candidate(tmp_path, monkeypatch, operation):
    vault = _vault(tmp_path, **{"note.md": "# Note\n\nold body"})
    db_path = tmp_path / "db.lance"
    full_index(str(vault), str(db_path), MockProvider())
    before_name = _read_manifest(str(db_path))["active"]
    (vault / "note.md").write_text("# Note\n\nnew body", encoding="utf-8")

    import some_vault_some_mcp.core.indexer as indexer
    real_replace = indexer._durable_replace

    def replace_then_fail(temp_path, target_path):
        real_replace(temp_path, target_path)
        raise OSError("directory fsync failed after replace")

    monkeypatch.setattr(indexer, "_durable_replace", replace_then_fail)
    with pytest.raises(OSError, match="directory fsync failed after replace"):
        if operation == "full":
            full_index(str(vault), str(db_path), MockProvider())
        else:
            incremental_index(str(vault), str(db_path), MockProvider())

    manifest = _read_manifest(str(db_path))
    frame, active_name = _active_frame(db_path)
    assert active_name == manifest["active"]
    assert active_name != before_name
    assert active_name in _get_db(str(db_path)).list_tables().tables
    assert frame["content"].str.contains("new body").any()


def test_compatible_rebuild_carries_last_good_rows_for_skipped_file(tmp_path, monkeypatch):
    vault = _vault(tmp_path, **{"bad.md": "# Bad\n\nold", "good.md": "# Good\n\nold"})
    db_path = tmp_path / "db.lance"
    full_index(str(vault), str(db_path), MockProvider())
    (vault / "bad.md").write_text("# Bad\n\nnew", encoding="utf-8")
    (vault / "good.md").write_text("# Good\n\nnew", encoding="utf-8")

    import some_vault_some_mcp.core.chunker as chunker
    real = chunker.chunk_markdown

    def fail_bad(rel_path, content, **kwargs):
        if rel_path == "bad.md":
            raise RuntimeError("bad chunk")
        return real(rel_path, content, **kwargs)

    monkeypatch.setattr(chunker, "chunk_markdown", fail_bad)
    result = full_index(str(vault), str(db_path), MockProvider())
    frame, _ = _active_frame(db_path)
    bad = frame[frame["file_path"] == "bad.md"]
    good = frame[frame["file_path"] == "good.md"]
    assert result["files_skipped"] == 1
    assert bad["content"].str.contains("old").any()
    assert not bad["content"].str.contains("new").any()
    assert good["content"].str.contains("new").any()


def test_per_entry_scan_failure_is_not_inferred_deleted(tmp_path, monkeypatch):
    vault = _vault(tmp_path, **{"bad.md": "# Bad\n\nold", "good.md": "# Good\n\nold"})
    db_path = tmp_path / "db.lance"
    full_index(str(vault), str(db_path), MockProvider())
    (vault / "bad.md").write_text("# Bad\n\nnew", encoding="utf-8")
    (vault / "good.md").write_text("# Good\n\nnew", encoding="utf-8")

    import some_vault_some_mcp.core.indexer as indexer
    good_mtime = (vault / "good.md").stat().st_mtime
    uncertain = VaultScan(
        files={"good.md": ScannedFile("good.md", good_mtime)},
        skipped={"bad.md"},
        errors=["bad.md: synthetic stat failure"],
        complete=True,
    )
    monkeypatch.setattr(indexer, "_scan_vault_complete", lambda _vault_path: uncertain)
    result = full_index(str(vault), str(db_path), MockProvider())

    frame, _ = _active_frame(db_path)
    bad = frame[frame["file_path"] == "bad.md"]
    assert result["files_skipped"] == 1
    assert bad["content"].str.contains("old").any()
    assert not bad["content"].str.contains("new").any()


def test_every_scanned_file_failure_aborts_publication(tmp_path, monkeypatch):
    vault = _vault(tmp_path, **{"note.md": "# Note\n\nbody"})
    db_path = tmp_path / "db.lance"
    full_index(str(vault), str(db_path), MockProvider())
    before = _read_manifest(str(db_path))

    import some_vault_some_mcp.core.chunker as chunker
    monkeypatch.setattr(chunker, "chunk_markdown", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="Every scanned note failed"):
        full_index(str(vault), str(db_path), MockProvider())
    assert _read_manifest(str(db_path)) == before


def test_incomplete_directory_enumeration_aborts_without_deletions(tmp_path, monkeypatch):
    vault = _vault(tmp_path, **{"ok.md": "# OK", "locked/hidden.md": "# Hidden"})
    db_path = tmp_path / "db.lance"
    full_index(str(vault), str(db_path), MockProvider())
    before = _read_manifest(str(db_path))

    real_scandir = os.scandir

    def fail_locked(path):
        if Path(path).name == "locked":
            raise PermissionError("locked subtree")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", fail_locked)
    with pytest.raises(RuntimeError, match="scan incomplete"):
        full_index(str(vault), str(db_path), MockProvider())
    assert _read_manifest(str(db_path)) == before


def test_missing_active_serves_compatible_previous_degraded(tmp_path):
    vault = _vault(tmp_path, **{"note.md": "# Note\n\nbody"})
    db_path = tmp_path / "db.lance"
    full_index(str(vault), str(db_path), MockProvider())
    full_index(str(vault), str(db_path), MockProvider())
    manifest = _read_manifest(str(db_path))
    _get_db(str(db_path)).drop_table(manifest["active"])

    _, table, name, degraded = resolve_active_table(str(db_path), 768)
    assert table is not None
    assert name == manifest["previous"]
    assert degraded is True


def test_publication_reclaims_generations_older_than_previous(tmp_path):
    vault = _vault(tmp_path, **{"note.md": "# Note\n\nbody"})
    db_path = tmp_path / "db.lance"
    full_index(str(vault), str(db_path), MockProvider())
    db = _get_db(str(db_path))
    first_name = _read_manifest(str(db_path))["active"]
    full_index(str(vault), str(db_path), MockProvider())
    assert first_name in db.list_tables().tables
    full_index(str(vault), str(db_path), MockProvider())
    assert first_name not in db.list_tables().tables
    owned = [
        name for name in db.list_tables().tables
        if name.startswith(GENERATION_PREFIX)
    ]
    assert len(owned) <= 2


def test_non_finite_vector_never_publishes(tmp_path):
    class NanProvider(MockProvider):
        def embed_texts(self, texts):
            return [[float("nan")] * self.dimensions for _ in texts]

    vault = _vault(tmp_path, **{"note.md": "# Note\n\nbody"})
    db_path = tmp_path / "db.lance"
    with pytest.raises(RuntimeError, match="non-finite"):
        full_index(str(vault), str(db_path), NanProvider())
    assert _read_manifest(str(db_path)) is None
    assert not any(name.startswith(GENERATION_PREFIX) for name in _get_db(str(db_path)).list_tables().tables)
