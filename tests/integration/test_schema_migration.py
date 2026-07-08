"""Schema-version sidecar + auto-reindex (plan Phase 1 / Q3)."""

import json
import os
from pathlib import Path

import lancedb
import pytest

from some_vault_some_mcp.core.embeddings import MockProvider
from some_vault_some_mcp.core.indexer import (
    SCHEMA_VERSION,
    _get_db,
    _schema_version_file,
    check_and_maybe_migrate,
    full_index,
)

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent.parent / "fixtures" / "vault"


def test_full_index_writes_marker(tmp_path):
    db_path = str(tmp_path / "db.lance")
    full_index(str(FIXTURES), db_path, MockProvider())
    marker = _schema_version_file(db_path)
    assert os.path.exists(marker)
    assert json.loads(Path(marker).read_text())["version"] == SCHEMA_VERSION
    # A freshly-written index needs no migration.
    assert check_and_maybe_migrate(_get_db(db_path), db_path) is False


def test_legacy_plain_format_triggers_reindex(tmp_path):
    """A table with old plain-format (non-sentinel) tags and no marker must
    report needs-reindex."""
    db_path = str(tmp_path / "db.lance")
    db = lancedb.connect(db_path)
    db.create_table("vault_chunks", data=[
        {"tags": "art,ideas", "projects": "", "vector": [0.0]},  # old format
    ])
    assert not os.path.exists(_schema_version_file(db_path))
    assert check_and_maybe_migrate(db, db_path) is True


def test_missing_marker_but_current_data_self_heals(tmp_path):
    """Marker lost over an already-migrated table: no reindex, marker rewritten."""
    db_path = str(tmp_path / "db.lance")
    full_index(str(FIXTURES), db_path, MockProvider())
    os.remove(_schema_version_file(db_path))
    assert check_and_maybe_migrate(_get_db(db_path), db_path) is False
    assert os.path.exists(_schema_version_file(db_path))  # healed


def test_corrupt_marker_over_current_data_self_heals(tmp_path):
    db_path = str(tmp_path / "db.lance")
    full_index(str(FIXTURES), db_path, MockProvider())
    Path(_schema_version_file(db_path)).write_text("{not valid json")
    assert check_and_maybe_migrate(_get_db(db_path), db_path) is False
    assert json.loads(Path(_schema_version_file(db_path)).read_text())["version"] == SCHEMA_VERSION


def test_no_table_no_migration(tmp_path):
    db_path = str(tmp_path / "empty.lance")
    db = lancedb.connect(db_path)
    assert check_and_maybe_migrate(db, db_path) is False
