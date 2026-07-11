FROM python:3.12-slim@sha256:ec948fa5f90f4f8907e89f4800cfd2d2e91e391a4bce4a6afa77ba265bc3a2fe

COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /uvx /bin/

RUN groupadd -g 1000 vault && useradd -u 1000 -g vault -m -s /bin/bash vault

WORKDIR /app

# Deps layer (cached until pyproject.toml or uv.lock change)
COPY pyproject.toml uv.lock README.md /app/
RUN mkdir -p src/some_vault_some_mcp && touch src/some_vault_some_mcp/__init__.py \
    && uv sync --frozen --no-dev --no-editable

COPY src/ /app/src/
RUN uv sync --frozen --no-dev --no-editable

RUN mkdir -p /opt/data && chown -R vault:vault /opt/data

USER vault

ENV MCP_TRANSPORT=sse \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=3789 \
    EMBEDDING_PROVIDER=fastembed \
    LANCE_DB_PATH=/opt/data/vault.lance \
    VAULT_PATH=/opt/vault \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 3789

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3789/', timeout=5)"

ENTRYPOINT ["some-vault-some-mcp", "serve"]
