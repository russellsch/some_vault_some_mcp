"""Unit tests for embedding provider routing and mock provider."""

import os
import pytest

from some_vault_some_mcp.core.embeddings import (
    FastEmbedProvider, MockProvider, Mock3072Provider, get_provider, PROVIDERS
)


def test_mock_provider_dimensions():
    p = MockProvider()
    assert p.dimensions == 768


def test_mock_embed_texts_returns_correct_dims():
    p = MockProvider()
    results = p.embed_texts(["hello", "world"])
    assert len(results) == 2
    assert all(len(v) == 768 for v in results)


def test_mock_embed_query_returns_correct_dims():
    p = MockProvider()
    result = p.embed_query("test query")
    assert len(result) == 768


def test_mock_deterministic():
    p = MockProvider()
    v1 = p.embed_texts(["same text"])[0]
    v2 = p.embed_texts(["same text"])[0]
    assert v1 == v2


def test_mock_different_texts_different_vectors():
    p = MockProvider()
    v1 = p.embed_texts(["text A"])[0]
    v2 = p.embed_texts(["text B"])[0]
    assert v1 != v2


def test_mock_3072_dimensions():
    p = Mock3072Provider()
    assert p.dimensions == 3072
    result = p.embed_texts(["test"])
    assert len(result[0]) == 3072


def test_provider_routing_mock(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    p = get_provider()
    assert isinstance(p, MockProvider)
    assert p.dimensions == 768


def test_provider_routing_mock_3072(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock-3072")
    p = get_provider()
    assert isinstance(p, Mock3072Provider)


def test_unknown_provider_raises_clear_error(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "nonexistent-provider")
    with pytest.raises(ValueError, match="nonexistent-provider"):
        get_provider()


def test_unknown_provider_error_mentions_available(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "bad")
    with pytest.raises(ValueError) as exc_info:
        get_provider()
    # Error should name the available providers
    msg = str(exc_info.value)
    assert "fastembed" in msg


def test_mock_query_different_from_text():
    p = MockProvider()
    doc_vec = p.embed_texts(["hello"])[0]
    query_vec = p.embed_query("hello")
    # query: prefix causes different hash seed
    assert doc_vec != query_vec


def test_all_known_providers_in_registry():
    assert "fastembed" in PROVIDERS
    assert "ollama" in PROVIDERS
    assert "openai" in PROVIDERS
    assert "mock" in PROVIDERS


def test_embed_texts_return_length_matches_input():
    """embed_texts must return same length as input — preserves alignment."""
    p = MockProvider()
    texts = ["a", "b", "c", "d"]
    results = p.embed_texts(texts)
    assert len(results) == len(texts)


def test_embed_texts_none_alignment():
    """When embed_texts returns None for some items, indexer must skip those
    and produce correct number of records aligned with non-None items."""
    from some_vault_some_mcp.core.indexer import _make_record

    # Simulate 3 chunks with 3 vectors, second one is None (failed)
    chunks = [
        {"file_path": "a.md", "chunk_index": 0, "heading": "", "content": "aaa",
         "title": "A", "tags": [], "projects": [], "area": "", "status": "",
         "source": "", "file_mtime": 0.0, "text_to_embed": "aaa"},
        {"file_path": "b.md", "chunk_index": 0, "heading": "", "content": "bbb",
         "title": "B", "tags": [], "projects": [], "area": "", "status": "",
         "source": "", "file_mtime": 0.0, "text_to_embed": "bbb"},
        {"file_path": "c.md", "chunk_index": 0, "heading": "", "content": "ccc",
         "title": "C", "tags": [], "projects": [], "area": "", "status": "",
         "source": "", "file_mtime": 0.0, "text_to_embed": "ccc"},
    ]
    vectors: list[list[float] | None] = [
        [0.1] * 768,  # chunk 0 OK
        None,          # chunk 1 FAILED
        [0.3] * 768,  # chunk 2 OK
    ]

    records = [_make_record(c, v) for c, v in zip(chunks, vectors) if v is not None]

    # Should produce 2 records (chunks 0 and 2), skipping the None
    assert len(records) == 2
    # Correct alignment: file_path should be a.md and c.md, not b.md
    assert records[0]["file_path"] == "a.md"
    assert records[1]["file_path"] == "c.md"


# --- FastEmbedProvider tests ---

@pytest.fixture(scope="session")
def fastembed_provider():
    return FastEmbedProvider()


def test_fastembed_provider_routing(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    p = get_provider()
    assert isinstance(p, FastEmbedProvider)


def test_fastembed_dimensions(fastembed_provider):
    assert fastembed_provider.dimensions == 768


def test_fastembed_embed_texts_returns_correct_dims(fastembed_provider):
    results = fastembed_provider.embed_texts(["hello", "world"])
    assert len(results) == 2
    assert all(len(v) == 768 for v in results)


def test_fastembed_embed_query_returns_correct_dims(fastembed_provider):
    result = fastembed_provider.embed_query("test query")
    assert len(result) == 768


def test_fastembed_embed_texts_return_length_matches_input(fastembed_provider):
    texts = ["a", "b", "c", "d"]
    results = fastembed_provider.embed_texts(texts)
    assert len(results) == len(texts)


def test_fastembed_embed_and_query_both_produce_vectors(fastembed_provider):
    doc_vec = fastembed_provider.embed_texts(["hello"])[0]
    query_vec = fastembed_provider.embed_query("hello")
    assert len(doc_vec) == 768
    assert len(query_vec) == 768


def test_fastembed_empty_input(fastembed_provider):
    assert fastembed_provider.embed_texts([]) == []


def test_fastembed_dimensions_via_registry_no_download(monkeypatch):
    """Dimensions come from fastembed's static registry — no model instantiation
    at construction (plan Phase 4 / O12/F12)."""
    monkeypatch.delenv("FASTEMBED_DIMENSIONS", raising=False)
    from some_vault_some_mcp.core.embeddings import FastEmbedProvider
    p = FastEmbedProvider()
    assert p.dimensions == 768
    assert p._embedding is None  # lazy — registry lookup did not build the model
