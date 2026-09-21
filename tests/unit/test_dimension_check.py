"""Unit tests for LanceDB dimension check at startup (§6.4 step 3).

These tests use the mock provider and tmp_path — no Ollama needed.
"""

import json
from pathlib import Path

import pytest

from some_vault_some_mcp.core.embeddings import MockProvider, Mock3072Provider
from some_vault_some_mcp.core.indexer import (
    _check_dimension_mismatch,
    _get_db,
    full_index,
    TABLE_NAME,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "vault"


def test_no_table_no_error(tmp_path):
    db = _get_db(str(tmp_path / "test.lance"))
    provider = MockProvider()
    # Should not raise
    _check_dimension_mismatch(db, provider.dimensions)


def test_matching_dims_ok(tmp_path):
    db_path = str(tmp_path / "vault.lance")
    provider = MockProvider()  # 768 dims
    full_index(str(FIXTURES), db_path, provider)
    db = _get_db(db_path)
    # Same dimensions — should not raise
    _check_dimension_mismatch(db, 768)


def test_mismatched_dims_raises(tmp_path):
    db_path = str(tmp_path / "vault.lance")
    provider = MockProvider()  # 768 dims
    full_index(str(FIXTURES), db_path, provider)
    db = _get_db(db_path)
    # Provider reporting 3072 dims but table is 768
    with pytest.raises(RuntimeError) as exc_info:
        _check_dimension_mismatch(db, 3072)
    msg = str(exc_info.value)
    assert "3072" in msg
    assert "768" in msg
    assert "reindex" in msg.lower() or "required" in msg.lower()


def test_mismatched_dims_existing_index_untouched(tmp_path):
    db_path = str(tmp_path / "vault.lance")
    provider = MockProvider()
    result = full_index(str(FIXTURES), db_path, provider)
    initial_count = result["chunks_created"]

    db = _get_db(db_path)
    try:
        _check_dimension_mismatch(db, 3072)
    except RuntimeError:
        pass

    # Table should still have original row count
    from some_vault_some_mcp.core.indexer import _get_table
    table = _get_table(db)
    assert table.count_rows() == initial_count


def test_empty_table_still_checks_schema(tmp_path):
    """Empty table (0 rows but schema exists) — check runs against schema."""
    import lancedb
    import pyarrow as pa

    db_path = str(tmp_path / "vault.lance")
    db = lancedb.connect(db_path)

    # Create table with 768-dim schema but no rows
    schema = pa.schema([
        pa.field("file_path", pa.string()),
        pa.field("chunk_index", pa.int64()),
        pa.field("heading", pa.string()),
        pa.field("content", pa.string()),
        pa.field("title", pa.string()),
        pa.field("tags", pa.string()),
        pa.field("projects", pa.string()),
        pa.field("area", pa.string()),
        pa.field("status", pa.string()),
        pa.field("source", pa.string()),
        pa.field("file_mtime", pa.float64()),
        pa.field("vector", pa.list_(pa.float32(), 768)),
    ])
    db.create_table(TABLE_NAME, schema=schema)

    db2 = _get_db(db_path)
    # 3072 vs 768 — should raise even with empty table
    with pytest.raises(RuntimeError):
        _check_dimension_mismatch(db2, 3072)

    # 768 vs 768 — should not raise
    _check_dimension_mismatch(db2, 768)
