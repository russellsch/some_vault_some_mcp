"""EmbeddingProvider protocol and concrete implementations.

Provider selected by EMBEDDING_PROVIDER env (default: fastembed).
Each implementation imports its SDK lazily — only loaded when that provider
is active.

embed_texts: for document chunks (indexing)
embed_query: for search queries
"""

import hashlib
import logging
import os
import platform
import random
from typing import Protocol

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        """Embed document chunks for indexing.

        Returns a list of the same length as texts. Failed items are None
        instead of being skipped — preserves positional alignment with input.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        ...


class OllamaProvider:
    """Embeddings via Ollama REST API (batch-capable)."""

    model = "nomic-embed-text"
    dimensions = 768
    _max_chars = 8192 * 4  # ~8k tokens at ~4 chars/token

    def __init__(self, url: str | None = None):
        self._url = url or os.getenv("OLLAMA_URL", "http://localhost:11434")

    def _client(self):
        import ollama
        return ollama.Client(host=self._url)

    def _truncate(self, text: str) -> str:
        if len(text) > self._max_chars:
            logger.warning(f"Truncating text from {len(text)} to {self._max_chars} chars for Ollama embedding")
            return text[:self._max_chars]
        return text

    def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        """Embed document chunks. Returns None for failed items (preserves alignment)."""
        if not texts:
            return []
        client = self._client()
        # Add document prefix for nomic-embed-text
        prefixed = [f"search_document: {self._truncate(t)}" for t in texts]
        try:
            response = client.embed(model=self.model, input=prefixed)
            embs = response.get("embeddings") or response.get("embedding") or []
            if embs and not isinstance(embs[0], list):
                embs = [embs]
            # Enforce the positional-alignment contract — a short/miscounted batch
            # would otherwise be silently truncated by zip() in the indexer.
            if len(embs) != len(prefixed) or not all(isinstance(e, list) for e in embs):
                raise ValueError(
                    f"Ollama returned {len(embs)} embeddings for {len(prefixed)} inputs"
                )
            return embs
        except Exception as e:
            logger.warning(f"Batch embed_texts failed: {e}. Embedding individually.")
            results: list[list[float] | None] = []
            for i, text in enumerate(prefixed):
                try:
                    resp = client.embed(model=self.model, input=text)
                    embs = resp.get("embeddings") or resp.get("embedding") or []
                    if embs and isinstance(embs[0], list):
                        results.append(embs[0])
                    elif embs:
                        results.append(embs)
                    else:
                        logger.error(f"Empty embedding for chunk {i} — skipping")
                        results.append(None)
                except Exception as e2:
                    logger.error(f"Single embed failed for chunk {i}: {e2} — None")
                    results.append(None)
            return results

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query with query prefix."""
        client = self._client()
        prefixed = f"search_query: {self._truncate(text)}"
        resp = client.embed(model=self.model, input=prefixed)
        embs = resp.get("embeddings") or resp.get("embedding") or []
        if embs and isinstance(embs[0], list):
            return embs[0]
        return embs


class OpenAIProvider:
    """Embeddings via OpenAI text-embedding-3-large."""

    model = "text-embedding-3-large"
    dimensions = 3072
    _max_chars = 8000 * 4  # ~8k tokens at ~4 chars/token

    def __init__(self):
        self._api_key = os.getenv("OPENAI_API_KEY")

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=self._api_key)

    def _truncate(self, text: str) -> str:
        if len(text) > self._max_chars:
            return text[:self._max_chars]
        return text

    def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        """Embed document chunks. Returns None for failed items (preserves alignment)."""
        if not texts:
            return []
        client = self._client()
        results: list[list[float] | None] = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = [self._truncate(t) for t in texts[i:i + batch_size]]
            try:
                resp = client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                )
                results.extend([item.embedding for item in resp.data])
            except Exception as e:
                logger.warning(f"Batch embedding failed at offset {i}: {e}. Embedding individually.")
                for j, text in enumerate(batch):
                    try:
                        resp2 = client.embeddings.create(
                            model=self.model,
                            input=[self._truncate(text[:32000])],
                            dimensions=self.dimensions,
                        )
                        results.append(resp2.data[0].embedding)
                    except Exception as e2:
                        logger.error(f"Single embedding failed at batch[{i}][{j}]: {e2} — None")
                        results.append(None)
        return results

    def embed_query(self, text: str) -> list[float]:
        client = self._client()
        resp = client.embeddings.create(
            model=self.model,
            input=[self._truncate(text)],
            dimensions=self.dimensions,
        )
        return resp.data[0].embedding


class FastEmbedProvider:
    """Embeddings via fastembed (ONNX, local, no server needed)."""

    def __init__(self):
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            default_model = "nomic-ai/nomic-embed-text-v1.5"
        else:
            default_model = "nomic-ai/nomic-embed-text-v1.5-Q"
        self.model = os.getenv("FASTEMBED_MODEL", default_model)
        self._embedding = None
        dims_env = os.getenv("FASTEMBED_DIMENSIONS")
        if dims_env:
            self.dimensions = int(dims_env)
        else:
            self.dimensions = self._resolve_dimensions()

    def _resolve_dimensions(self) -> int:
        """Dimension from fastembed's static model registry — avoids downloading
        and instantiating the model at construction just to measure its width.
        Falls back to a one-off probe (via the lazy client) for unknown models."""
        try:
            from fastembed import TextEmbedding
            for m in TextEmbedding.list_supported_models():
                if m.get("model") == self.model and m.get("dim"):
                    return int(m["dim"])
        except Exception:
            pass
        return len(list(self._client().embed(["_"]))[0])

    def _client(self):
        if self._embedding is None:
            from fastembed import TextEmbedding
            self._embedding = TextEmbedding(model_name=self.model)
        return self._embedding

    def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []
        try:
            embeddings = list(self._client().embed(texts))
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.warning(f"Batch embed failed: {e}. Embedding individually.")
            results: list[list[float] | None] = []
            for text in texts:
                try:
                    emb = list(self._client().embed([text]))[0]
                    results.append(emb.tolist())
                except Exception as e2:
                    logger.error(f"Single embed failed: {e2}")
                    results.append(None)
            return results

    def embed_query(self, text: str) -> list[float]:
        return list(self._client().query_embed(text))[0].tolist()


class MockProvider:
    """Deterministic mock provider for unit tests — no Ollama needed.

    Returns seeded pseudo-random vectors derived from the text hash.
    """

    model = "mock"
    dimensions = 768

    def _hash_vector(self, text: str, dims: int) -> list[float]:
        h = hashlib.sha256(text.encode()).hexdigest()
        seed = int(h[:16], 16)
        rng = random.Random(seed)
        return [rng.gauss(0, 1) for _ in range(dims)]

    def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        return [self._hash_vector(t, self.dimensions) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_vector(f"query:{text}", self.dimensions)


class Mock3072Provider(MockProvider):
    """MockProvider that reports 3072 dimensions — for testing provider-switch checks."""
    model = "mock-3072"
    dimensions = 3072

    def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        return [self._hash_vector(t, self.dimensions) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_vector(f"query:{text}", self.dimensions)


PROVIDERS: dict[str, type] = {
    "fastembed": FastEmbedProvider,
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "mock": MockProvider,
    "mock-3072": Mock3072Provider,
}


def get_provider() -> EmbeddingProvider:
    """Return a provider instance based on EMBEDDING_PROVIDER env."""
    name = os.getenv("EMBEDDING_PROVIDER", "fastembed")
    cls = PROVIDERS.get(name)
    if cls is None:
        available = ", ".join(sorted(PROVIDERS.keys()))
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{name}'. Available: {available}"
        )
    return cls()
