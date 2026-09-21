"""CLI entrypoint: some-vault-some-mcp serve

Boot sequence (§6.4):
1. Load config from env
2. Validate vault path
3. Initialize embedding provider
4. Validate/clean the generation manifest and dimensions
5. Start filesystem watcher in buffering mode
6. Build or reconcile the active index
7. Start MCP transport (SSE or stdio)
"""

import argparse
import hmac
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _wait_for_ollama(url: str, timeout: int = 60) -> bool:
    """Poll Ollama health endpoint until it responds."""
    import time
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/tags", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def serve(args) -> None:
    from some_vault_some_mcp.config import load_config
    from some_vault_some_mcp.core.embeddings import get_provider
    from some_vault_some_mcp.core.indexer import (
        _check_dimension_mismatch, _get_db, _get_table, check_and_maybe_migrate,
        cleanup_inactive_generations, full_index, incremental_index, resolve_active_table,
    )
    from some_vault_some_mcp.core.watcher import start_watcher
    from some_vault_some_mcp.server import build_server

    config = load_config()

    # Folders hidden from the index and from listings (see paths.is_index_excluded).
    from some_vault_some_mcp.core.paths import configure_excluded_dirs
    configure_excluded_dirs(config.excluded_dirs)

    # CLI args override env
    if args.transport:
        config.transport = args.transport
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port

    if not config.vault_path:
        logger.error("VAULT_PATH is not set — exiting")
        sys.exit(1)
    if not os.path.isdir(config.vault_path):
        logger.error(f"VAULT_PATH does not exist or is not a directory: {config.vault_path}")
        sys.exit(1)

    logger.info(f"Vault path: {config.vault_path}")
    logger.info(f"LanceDB path: {config.db_path}")
    logger.info(f"Transport: {config.transport}")

    raw_db_path = Path(config.db_path).expanduser()
    resolved_db_path = raw_db_path.resolve()
    vault_root = Path(config.vault_path).resolve()
    cwd_root = Path.cwd().resolve()
    if not raw_db_path.is_absolute():
        logger.warning("LANCE_DB_PATH is relative; use a trusted absolute path outside the vault and workspace")
    if resolved_db_path.is_relative_to(vault_root):
        logger.warning("LANCE_DB_PATH is inside the vault; index data is attacker-accessible")
    if resolved_db_path.is_relative_to(cwd_root):
        logger.warning("LANCE_DB_PATH is inside the current working directory; use a trusted external location")

    # Step 1: wait for Ollama if using it
    provider_name = os.getenv("EMBEDDING_PROVIDER", "fastembed")
    if provider_name == "ollama":
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        logger.info(f"Waiting for Ollama at {ollama_url}...")
        if not _wait_for_ollama(ollama_url):
            logger.warning(
                "Ollama not reachable after 60s — starting anyway, but embedding "
                "(indexing and semantic/hybrid search) will fail until it is up. "
                "Use EMBEDDING_PROVIDER=fastembed for a local, server-less provider."
            )

    # Step 2: initialize provider
    try:
        provider = get_provider()
        logger.info(f"Embedding provider: {provider_name} ({provider.dimensions} dims)")
    except ValueError as e:
        logger.error(f"Embedding provider error: {e}")
        sys.exit(1)

    # Step 3: clean only stale owned generations before any reader can open one.
    cleanup_inactive_generations(config.db_path)

    # Step 4: dimension check. An explicit forced rebuild is the only allowed
    # dimension migration path.
    db = _get_db(config.db_path)
    if not getattr(args, "reindex_force", False):
        try:
            _check_dimension_mismatch(db, provider.dimensions)
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)

    # Step 4: initial index + watcher
    from some_vault_some_mcp.server import IndexGate
    import threading

    gate = IndexGate()
    table = _get_table(db)
    needs_full_index = table is None
    if not needs_full_index and check_and_maybe_migrate(db, config.db_path):
        logger.warning("Index schema is outdated — rebuilding from scratch (one-time reindex).")
        needs_full_index = True
    if getattr(args, "reindex_force", False):
        logger.info("--reindex-force: rebuilding the index from scratch.")
        needs_full_index = True

    # Capture changes before any scan. Full publication is refused if capture
    # cannot start, because a scan/cutover gap would otherwise lose changes.
    watcher = start_watcher(config.vault_path, config.db_path, provider, buffering=True)

    def _background_index():
        try:
            if needs_full_index:
                if watcher is None:
                    raise RuntimeError("filesystem watcher capture is unavailable; refusing full index publication")
                logger.info("Background full_index started (server is accepting connections)...")
                result = full_index(config.vault_path, config.db_path, provider, watcher=watcher)
                logger.info(f"Full index complete: {result}")
            else:
                logger.info("Background incremental_index started (server is accepting connections)...")
                if watcher is not None:
                    watcher.acquire_cutover()
                try:
                    result = incremental_index(config.vault_path, config.db_path, provider)
                    if watcher is not None:
                        watcher.commit_cutover()
                finally:
                    if watcher is not None:
                        watcher.release_cutover()
                logger.info(f"Incremental index complete: {result}")
            gate.set_ready()
        except Exception as e:
            if watcher is not None:
                watcher.abort_cutover()
            logger.error(f"Background indexing failed: {e}")
            # A compatible published generation remains usable when a staged
            # rebuild fails. Keep search available in degraded mode; only fail
            # closed when there is no valid table to serve.
            _, active, _, _ = resolve_active_table(config.db_path, provider.dimensions)
            if active is None:
                gate.set_failed(str(e))
            else:
                gate.set_ready()

    threading.Thread(target=_background_index, daemon=True, name="background-indexer").start()

    # Step 5: build and run server
    mcp = build_server(config, provider, gate)

    if config.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        import uvicorn

        loopback = config.host in ("127.0.0.1", "::1", "localhost")
        if not config.api_key and not loopback:
            logger.warning(
                f"Binding {config.host}:{config.port} with NO VAULT_API_KEY set — "
                "the vault is exposed to the network without authentication. Set "
                "VAULT_API_KEY, or bind 127.0.0.1."
            )

        sse_app = mcp.http_app(transport="sse")
        if config.api_key:
            app = _APIKeyMiddleware(sse_app, config.api_key, config.allow_unauth_sse)
            logger.info(f"Starting SSE with API key auth on {config.host}:{config.port}")
        else:
            app = _HealthMiddleware(sse_app)
            logger.info(f"Starting SSE on {config.host}:{config.port}")
        uvicorn.run(app, host=config.host, port=config.port)


def _health_response():
    return (
        {"type": "http.response.start", "status": 200,
         "headers": [(b"content-type", b"application/json")]},
        {"type": "http.response.body",
         "body": b'{"status":"ok","service":"some-vault-some-mcp"}'},
    )


class _HealthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "") == "/":
            start, body = _health_response()
            await send(start)
            await send(body)
            return
        return await self.app(scope, receive, send)


class _APIKeyMiddleware:
    def __init__(self, app, api_key, allow_unauth_sse=False):
        self.app = app
        self._expected = f"Bearer {api_key}"
        self.allow_unauth_sse = allow_unauth_sse

    def _authorized(self, scope) -> bool:
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()
        return hmac.compare_digest(auth, self._expected)

    async def __call__(self, scope, receive, send):
        stype = scope.get("type")
        if stype not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        # Never speak HTTP to a websocket — reject cleanly at the WS layer.
        if stype == "websocket":
            if self._authorized(scope):
                return await self.app(scope, receive, send)
            await send({"type": "websocket.close", "code": 1008})
            return
        path = scope.get("path", "")
        if path == "/":
            start, body = _health_response()
            await send(start)
            await send(body)
            return
        # Opt-in escape hatch for clients that cannot send a header on the SSE GET.
        if self.allow_unauth_sse and path in ("/sse", "/sse/"):
            return await self.app(scope, receive, send)
        if self._authorized(scope):
            return await self.app(scope, receive, send)
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({
            "type": "http.response.body",
            "body": b'{"error":"unauthorized"}',
        })


def main():
    parser = argparse.ArgumentParser(prog="some-vault-some-mcp")
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Start the MCP server")
    serve_p.add_argument("--transport", choices=["sse", "stdio"], default=None)
    serve_p.add_argument("--host", default=None)
    serve_p.add_argument("--port", type=int, default=None)
    serve_p.add_argument("--reindex-force", action="store_true",
                         help="Drop the existing index and rebuild it from scratch")

    args = parser.parse_args()
    if args.command == "serve":
        serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
