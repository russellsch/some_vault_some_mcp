"""Integration tests for indexer — require mock provider (no Ollama needed).

Provider-switch tests use mock-768 and mock-3072 — no Ollama.
Full Ollama tests would use @pytest.mark.integration and pytest.skip if unavailable.
"""

import json
from pathlib import Path

import pytest

from some_vault_some_mcp.core.embeddings import MockProvider, Mock3072Provider
from some_vault_some_mcp.core.indexer import (
    _check_dimension_mismatch,
    _get_db,
    _get_table,
    full_index,
    incremental_index,
    TABLE_NAME,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "vault"
SCHEMA_FIXTURE = Path(__file__).parent.parent / "fixtures" / "lancedb_schema.json"


def test_full_index_creates_table(tmp_path):
    db_path = str(tmp_path / "vault.lance")
    provider = MockProvider()
    result = full_index(str(FIXTURES), db_path, provider)
    assert result["chunks_created"] > 0
    assert result["files_indexed"] > 0


def test_schema_parity_with_fixture(tmp_path):
    """Check that indexed table schema matches the captured schema fixture."""
    db_path = str(tmp_path / "vault.lance")
    provider = MockProvider()
    full_index(str(FIXTURES), db_path, provider)

    db = _get_db(db_path)
    table = _get_table(db)
    assert table is not None

    schema_data = json.loads(SCHEMA_FIXTURE.read_text())
    expected_cols = {col["name"] for col in schema_data["columns"]}
    actual_cols = {field.name for field in table.schema}
    assert expected_cols == actual_cols


def test_vector_dimensions(tmp_path):
    """Indexed vectors should have 768 dims (MockProvider)."""
    db_path = str(tmp_path / "vault.lance")
    provider = MockProvider()
    full_index(str(FIXTURES), db_path, provider)
    db = _get_db(db_path)
    table = _get_table(db)
    schema = table.schema
    vector_field = next(f for f in schema if f.name == "vector")
    assert vector_field.type.list_size == 768


def test_excluded_dirs_not_indexed(tmp_path):
    import shutil
    from some_vault_some_mcp.core.paths import configure_excluded_dirs
    vault = tmp_path / "vault"
    shutil.copytree(str(FIXTURES), str(vault))
    (vault / ".claude" / "worktrees" / "w").mkdir(parents=True)
    (vault / ".claude" / "worktrees" / "w" / "note.md").write_text("# hidden", encoding="utf-8")
    (vault / "external").mkdir()
    (vault / "external" / "vendored.md").write_text("# vendored", encoding="utf-8")
    (vault / "kept" / ".draft.md").parent.mkdir()
    (vault / "kept" / ".draft.md").write_text("# Draft\n\nThis dot-file stays indexed.", encoding="utf-8")
    configure_excluded_dirs(["External"])

    db_path = str(tmp_path / "vault.lance")
    provider = MockProvider()
    full_index(str(vault), db_path, provider)
    db = _get_db(db_path)
    table = _get_table(db)
    df = table.to_pandas()
    paths = df["file_path"].tolist()
    for p in paths:
        parts = p.split("/")
        for seg in parts:
            assert seg not in (".obsidian", ".git", ".trash"), f"Excluded path leaked: {p}"
        assert not p.startswith(".claude/"), f"dot-dir leaked: {p}"
        assert not p.startswith("external/"), f"configured dir leaked: {p}"
    assert "kept/.draft.md" in paths


def _copy_fixture_vault(tmp_path):
    import shutil
    vault = tmp_path / "vault"
    shutil.copytree(str(FIXTURES), str(vault),
                    ignore=shutil.ignore_patterns(".git", ".trash", ".obsidian"))
    return vault


def test_incremental_skips_file_that_fails_to_chunk(tmp_path, monkeypatch):
    """A file whose chunking raises keeps its old chunks; the others reindex."""
    import os
    import some_vault_some_mcp.core.chunker as chunker
    vault = _copy_fixture_vault(tmp_path)
    db_path = str(tmp_path / "db.lance")
    full_index(str(vault), db_path, MockProvider())

    before = _get_table(_get_db(db_path)).to_pandas()
    bad_before = len(before[before["file_path"] == "simple.md"])
    assert bad_before > 0

    for name in ("simple.md", "linked-note.md"):
        p = vault / name
        p.write_text(p.read_text(encoding="utf-8") + "\n\nedited", encoding="utf-8")
        os.utime(str(p), (2_000_000_000, 2_000_000_000))

    real = chunker.chunk_markdown

    def boom(rel_path, content, **kw):
        if rel_path == "simple.md":
            raise RuntimeError("synthetic chunk failure")
        return real(rel_path, content, **kw)

    monkeypatch.setattr(chunker, "chunk_markdown", boom)
    result = incremental_index(str(vault), db_path, MockProvider(),
                               only_files={"simple.md", "linked-note.md"})

    assert result["files_skipped"] == 1
    assert result["files_indexed"] == 1
    after = _get_table(_get_db(db_path)).to_pandas()
    assert len(after[after["file_path"] == "simple.md"]) == bad_before
    assert not after[after["file_path"] == "simple.md"]["content"].str.contains("edited").any()
    assert after[after["file_path"] == "linked-note.md"]["content"].str.contains("edited").any()


def test_incremental_delete_fallback_per_path(tmp_path, monkeypatch):
    """If the batched delete fails, each path is deleted alone; no duplicates."""
    import os
    vault = _copy_fixture_vault(tmp_path)
    db_path = str(tmp_path / "db.lance")
    full_index(str(vault), db_path, MockProvider())

    for name in ("simple.md", "linked-note.md"):
        p = vault / name
        p.write_text(p.read_text(encoding="utf-8") + "\n\nedited", encoding="utf-8")
        os.utime(str(p), (2_000_000_000, 2_000_000_000))

    import lancedb.table as lt
    real_delete = lt.LanceTable.delete
    calls = []

    def flaky_delete(self, where):
        calls.append(where)
        if " OR " in where:
            raise RuntimeError("synthetic batched delete failure")
        return real_delete(self, where)

    monkeypatch.setattr(lt.LanceTable, "delete", flaky_delete)
    result = incremental_index(str(vault), db_path, MockProvider(),
                               only_files={"simple.md", "linked-note.md"})

    assert len(calls) == 3  # one batched attempt, then one per path
    assert result["files_indexed"] == 2
    assert result["files_skipped"] == 0
    after = _get_table(_get_db(db_path)).to_pandas()
    for name in ("simple.md", "linked-note.md"):
        rows = after[after["file_path"] == name]
        assert rows["content"].str.contains("edited").all(), f"stale chunks for {name}"
        assert rows["chunk_index"].is_unique, f"duplicate chunks for {name}"


def test_incremental_index_detects_changes(tmp_path):
    import shutil
    vault = tmp_path / "vault"
    shutil.copytree(str(FIXTURES), str(vault), ignore=shutil.ignore_patterns(".git", ".trash", ".obsidian"))

    db_path = str(tmp_path / "db.lance")
    provider = MockProvider()
    full_index(str(vault), db_path, provider)

    # Add a new file
    new_note = vault / "new_note.md"
    new_note.write_text("---\ntitle: New Note\n---\n\nBrand new content.", encoding="utf-8")

    result = incremental_index(str(vault), db_path, provider)
    assert result["files_indexed"] >= 1
    assert result["chunks_created"] >= 1


def test_incremental_index_handles_deletion(tmp_path):
    import shutil
    vault = tmp_path / "vault"
    shutil.copytree(str(FIXTURES), str(vault), ignore=shutil.ignore_patterns(".git", ".trash", ".obsidian"))

    db_path = str(tmp_path / "db.lance")
    provider = MockProvider()
    full_index(str(vault), db_path, provider)

    # Delete a file
    (vault / "no-frontmatter.md").unlink()
    result = incremental_index(str(vault), db_path, provider)
    assert result["files_removed"] >= 1


def test_single_file_reindex(tmp_path):
    """Single-file reindex only affects that file (upstream bug fixed)."""
    import shutil
    vault = tmp_path / "vault"
    shutil.copytree(str(FIXTURES), str(vault), ignore=shutil.ignore_patterns(".git", ".trash", ".obsidian"))

    db_path = str(tmp_path / "db.lance")
    provider = MockProvider()
    full_index(str(vault), db_path, provider)

    # Modify one file
    target = vault / "simple.md"
    content = target.read_text(encoding="utf-8")
    target.write_text(content + "\n\nAdded paragraph.", encoding="utf-8")
    import os; os.utime(str(target), None)  # update mtime

    result = incremental_index(str(vault), db_path, provider, single_file="simple.md")
    # Only the one modified file should be reindexed
    assert result["files_indexed"] == 1


def test_provider_switch_mismatch_refuses(tmp_path):
    """Index with 768-dim, switch to 3072-dim — boot must refuse."""
    db_path = str(tmp_path / "vault.lance")
    provider768 = MockProvider()
    full_index(str(FIXTURES), db_path, provider768)

    db = _get_db(db_path)
    provider3072 = Mock3072Provider()
    with pytest.raises(RuntimeError) as exc_info:
        _check_dimension_mismatch(db, provider3072.dimensions)
    assert "3072" in str(exc_info.value)
    assert "768" in str(exc_info.value)


def test_provider_restart_same_dims_ok(tmp_path):
    """Same provider dimensions on restart — boot succeeds, index reused."""
    db_path = str(tmp_path / "vault.lance")
    provider = MockProvider()
    result1 = full_index(str(FIXTURES), db_path, provider)

    db = _get_db(db_path)
    _check_dimension_mismatch(db, provider.dimensions)  # should not raise

    table = _get_table(db)
    assert table.count_rows() == result1["chunks_created"]


def test_tags_stored_as_comma_string(tmp_path):
    """tags/projects columns must be comma-separated strings (not Arrow lists)."""
    import pandas as pd
    db_path = str(tmp_path / "vault.lance")
    provider = MockProvider()
    full_index(str(FIXTURES), db_path, provider)
    db = _get_db(db_path)
    table = _get_table(db)
    df = table.to_pandas()
    # tags column should be string type (object in pandas <3, str in pandas 3+), not list
    assert pd.api.types.is_string_dtype(df["tags"]), f"Expected string dtype, got {df['tags'].dtype}"
    # Values should be strings, not lists
    non_empty = df[df["tags"] != ""]["tags"]
    if not non_empty.empty:
        sample = non_empty.iloc[0]
        assert isinstance(sample, str)


class _PartialFailProvider(MockProvider):
    """Returns None for every 3rd embedding to simulate failures."""
    def embed_texts(self, texts):
        vectors = super().embed_texts(texts)
        return [None if i % 3 == 2 else v for i, v in enumerate(vectors)]


def test_incremental_chunks_created_counts_successful_only(tmp_path):
    """chunks_created should reflect records actually added, not chunks produced."""
    import shutil
    vault = tmp_path / "vault"
    shutil.copytree(str(FIXTURES), str(vault),
                    ignore=shutil.ignore_patterns(".git", ".trash", ".obsidian"))

    db_path = str(tmp_path / "db.lance")
    full_index(str(vault), db_path, MockProvider())

    big = "---\ntitle: Big\n---\n\n# H\n\n" + "\n\n".join(
        f"Paragraph {i}. " + "x" * 800 for i in range(10)
    )
    (vault / "big-note.md").write_text(big, encoding="utf-8")

    result = incremental_index(str(vault), db_path, _PartialFailProvider(),
                               single_file="big-note.md")

    db = _get_db(db_path)
    table = _get_table(db)
    df = table.to_pandas()
    actual_big_chunks = len(df[df["file_path"] == "big-note.md"])
    assert result["chunks_created"] == actual_big_chunks
