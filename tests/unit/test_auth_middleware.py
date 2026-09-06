"""Unit tests for the SSE auth middleware (plan Phase 0.4 / O6/F5)."""

import pytest

from some_vault_some_mcp.cli import _APIKeyMiddleware, _validate_sse_security
from some_vault_some_mcp.config import VaultMcpConfig, load_config


KEY = "s3cret"


async def _downstream(scope, receive, send):
    await send({"type": "app.reached"})


async def _drive(app, scope):
    sent = []

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "noop"}

    await app(scope, receive, send)
    return sent


def _http(path, auth=None):
    headers = [(b"authorization", f"Bearer {auth}".encode())] if auth is not None else []
    return {"type": "http", "path": path, "headers": headers}


def _reached(sent):
    return any(m.get("type") == "app.reached" for m in sent)


def _status(sent):
    return next((m["status"] for m in sent if m.get("type") == "http.response.start"), None)


@pytest.mark.asyncio
async def test_valid_key_passes():
    app = _APIKeyMiddleware(_downstream, KEY)
    assert _reached(await _drive(app, _http("/messages", auth=KEY)))


@pytest.mark.asyncio
async def test_missing_or_wrong_key_401():
    app = _APIKeyMiddleware(_downstream, KEY)
    assert _status(await _drive(app, _http("/messages"))) == 401
    assert _status(await _drive(app, _http("/messages", auth="nope"))) == 401


@pytest.mark.asyncio
async def test_sse_get_requires_key_by_default():
    app = _APIKeyMiddleware(_downstream, KEY, allow_unauth_sse=False)
    assert _status(await _drive(app, _http("/sse"))) == 401
    assert _reached(await _drive(app, _http("/sse", auth=KEY)))


@pytest.mark.asyncio
async def test_sse_opt_out_allows_unauth_get():
    app = _APIKeyMiddleware(_downstream, KEY, allow_unauth_sse=True)
    assert _reached(await _drive(app, _http("/sse")))


@pytest.mark.asyncio
async def test_health_path_no_auth():
    app = _APIKeyMiddleware(_downstream, KEY)
    assert _status(await _drive(app, _http("/"))) == 200


@pytest.mark.asyncio
async def test_websocket_unauth_closed_not_http():
    app = _APIKeyMiddleware(_downstream, KEY)
    sent = await _drive(app, {"type": "websocket", "path": "/sse", "headers": []})
    assert sent == [{"type": "websocket.close", "code": 1008}]
    assert not any(m.get("type", "").startswith("http.") for m in sent)


def test_default_host_is_loopback(monkeypatch):
    for var in ("MCP_HOST", "VAULT_API_KEY", "some_vault_some_mcp_OVERRIDES"):
        monkeypatch.delenv(var, raising=False)
    assert load_config().host == "127.0.0.1"


def test_public_sse_requires_api_key():
    config = VaultMcpConfig(transport="sse", host="0.0.0.0", api_key="")
    with pytest.raises(ValueError, match="Refusing to bind unauthenticated SSE"):
        _validate_sse_security(config)


@pytest.mark.parametrize(
    "config",
    [
        VaultMcpConfig(transport="sse", host="0.0.0.0", api_key=KEY),
        VaultMcpConfig(transport="sse", host="127.0.0.1", api_key=""),
        VaultMcpConfig(transport="sse", host="127.0.0.2", api_key=""),
        VaultMcpConfig(transport="sse", host="::1", api_key=""),
        VaultMcpConfig(transport="stdio", host="0.0.0.0", api_key=""),
    ],
)
def test_secure_or_non_network_config_is_allowed(config):
    _validate_sse_security(config)
