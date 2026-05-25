"""E2E: build Docker image, start container, connect via MCP SSE, call tools.

Uses the existing test vault at tests/fixtures/vault.
Read-only tests mount the vault as "ro"; write tests copy it to a tmpdir
and mount "rw".  EMBEDDING_PROVIDER=mock so no Ollama needed.
"""

import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_VAULT = Path(__file__).parent.parent / "fixtures" / "vault"


def _poll_health(url: str, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _start_container(image, vault_host_path: str, mode: str = "ro"):
    from testcontainers.core.container import DockerContainer

    container = (
        DockerContainer(str(image))
        .with_env("EMBEDDING_PROVIDER", "mock")
        .with_env("VAULT_PATH", "/opt/vault")
        .with_env("LANCE_DB_PATH", "/opt/data/vault.lance")
        .with_env("MCP_TRANSPORT", "sse")
        .with_env("MCP_HOST", "0.0.0.0")
        .with_env("MCP_PORT", "3789")
        .with_volume_mapping(vault_host_path, "/opt/vault", mode)
        .with_exposed_ports(3789)
    )
    return container


@pytest.fixture(scope="module")
def docker_image():
    from testcontainers.core.image import DockerImage

    with DockerImage(path=str(PROJECT_ROOT), tag="vault-mcp-e2e:latest") as image:
        yield image


@pytest.fixture(scope="module")
def mcp_url(docker_image):
    """Container with test vault mounted read-only."""
    container = _start_container(docker_image, str(TEST_VAULT.resolve()), "ro")
    with container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(3789)
        url = f"http://{host}:{port}"
        assert _poll_health(url), "Container did not become healthy within 60s"
        yield url


def _make_world_writable(root: Path) -> None:
    """Grant write access to every user in the tree.

    The container runs as the image's "vault" user (uid 1000), which differs
    from the CI runner uid that owns this copy. Without this, the bind-mounted
    vault is read-only to the container and write tools fail with EACCES.
    """
    os.chmod(root, 0o777)
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames:
            os.chmod(Path(dirpath) / name, 0o777)
        for name in filenames:
            os.chmod(Path(dirpath) / name, 0o666)


@pytest.fixture(scope="module")
def mcp_rw_url(docker_image, tmp_path_factory):
    """Container with a writable copy of the test vault."""
    vault_copy = tmp_path_factory.mktemp("vault_rw")
    shutil.copytree(TEST_VAULT, vault_copy, dirs_exist_ok=True)
    _make_world_writable(vault_copy)
    container = _start_container(docker_image, str(vault_copy.resolve()), "rw")
    with container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(3789)
        url = f"http://{host}:{port}"
        assert _poll_health(url), "Container did not become healthy within 60s"
        yield url


async def _mcp_call(url: str, tool: str, args: dict | None = None) -> str:
    """Open a fresh MCP SSE session, call one tool, return text content."""
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(url=f"{url}/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments=args or {})
            return result.content[0].text


# ── container build + health ─────────────────────────────────────────────


class TestContainerHealth:
    def test_health_returns_ok(self, mcp_url):
        with urllib.request.urlopen(f"{mcp_url}/") as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
            assert body["status"] == "ok"
            assert body["service"] == "some-vault-some-mcp"


# ── MCP protocol ─────────────────────────────────────────────────────────


class TestMCPProtocol:
    async def test_lists_expected_tools(self, mcp_url):
        from mcp.client.session import ClientSession
        from mcp.client.sse import sse_client

        async with sse_client(url=f"{mcp_url}/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                names = {t.name for t in result.tools}

        expected = {
            "search", "get_note", "list_notes", "create_note",
            "append_to_note", "prepend_to_note", "update_frontmatter",
            "move_note", "delete_note",
            "get_daily_note", "create_daily_note",
            "get_tags", "get_backlinks", "get_outlinks",
            "find_orphans", "find_broken_links", "get_graph_neighbors",
            "vault_index_status", "vault_reindex",
        }
        assert expected <= names, f"Missing tools: {expected - names}"


# ── read-only vault tool operations ─────────────────────────────────────


class TestVaultTools:
    async def test_list_notes_finds_fixtures(self, mcp_url):
        text = await _mcp_call(mcp_url, "list_notes")
        assert "simple.md" in text
        assert "linked-note.md" in text
        assert "large-note.md" in text

    async def test_read_note_content(self, mcp_url):
        text = await _mcp_call(mcp_url, "get_note", {"path": "simple.md"})
        assert "Simple Note" in text
        assert "linked-note" in text

    async def test_read_note_not_found(self, mcp_url):
        text = await _mcp_call(mcp_url, "get_note", {"path": "nonexistent.md"})
        assert "not found" in text.lower()

    async def test_tags_from_vault(self, mcp_url):
        text = await _mcp_call(mcp_url, "get_tags")
        assert "reference" in text
        assert "test" in text
        assert "project" in text

    async def test_backlinks(self, mcp_url):
        text = await _mcp_call(mcp_url, "get_backlinks", {"path": "linked-note.md"})
        assert "simple" in text.lower()

    async def test_outlinks(self, mcp_url):
        text = await _mcp_call(mcp_url, "get_outlinks", {"path": "simple.md"})
        assert "linked-note" in text
        assert "deep-note" in text

    async def test_broken_links(self, mcp_url):
        text = await _mcp_call(mcp_url, "find_broken_links")
        assert "does-not-exist" in text

    async def test_find_orphans(self, mcp_url):
        text = await _mcp_call(mcp_url, "find_orphans")
        assert "no-frontmatter.md" in text

    async def test_graph_neighbors(self, mcp_url):
        text = await _mcp_call(mcp_url, "get_graph_neighbors", {
            "path": "simple.md",
            "depth": 2,
        })
        assert "linked-note" in text

    async def test_exact_search(self, mcp_url):
        text = await _mcp_call(mcp_url, "search", {
            "query": "simple note with frontmatter",
            "mode": "exact",
        })
        assert "simple" in text.lower()

    async def test_semantic_search(self, mcp_url):
        text = await _mcp_call(mcp_url, "search", {
            "query": "deeply nested architecture",
            "mode": "semantic",
        })
        assert "result" in text.lower()

    async def test_index_status(self, mcp_url):
        text = await _mcp_call(mcp_url, "vault_index_status")
        assert "Files indexed:" in text
        assert "Total chunks:" in text

    async def test_vault_reindex(self, mcp_url):
        text = await _mcp_call(mcp_url, "vault_reindex")
        assert "Reindex complete" in text
        assert "Files indexed:" in text


# ── read-only canvas operations ─────────────────────────────────────────


class TestCanvasReadOnly:
    async def test_list_canvases(self, mcp_url):
        text = await _mcp_call(mcp_url, "list_canvases")
        assert "test-canvas.canvas" in text

    async def test_read_canvas(self, mcp_url):
        text = await _mcp_call(mcp_url, "read_canvas", {"path": "test-canvas.canvas"})
        assert "node1aaa00000000" in text
        assert "Hello world" in text
        assert "edge1aaa00000000" in text

    async def test_read_canvas_not_found(self, mcp_url):
        text = await _mcp_call(mcp_url, "read_canvas", {"path": "nope.canvas"})
        assert "not found" in text.lower()


# ── daily notes (read-only) ─────────────────────────────────────────────


class TestDailyReadOnly:
    async def test_get_daily_note_by_date(self, mcp_url):
        text = await _mcp_call(mcp_url, "get_daily_note", {"date": "2026-05-08"})
        assert "2026-05-08" in text
        assert "Today's daily note" in text

    async def test_get_daily_note_not_found(self, mcp_url):
        text = await _mcp_call(mcp_url, "get_daily_note", {"date": "1999-01-01"})
        assert "not found" in text.lower()


# ── write operations (writable vault) ───────────────────────────────────


class TestWriteTools:
    async def test_create_note(self, mcp_rw_url):
        text = await _mcp_call(mcp_rw_url, "create_note", {
            "path": "new-note.md",
            "content": "Hello from E2E test.",
        })
        assert "Created" in text

        text = await _mcp_call(mcp_rw_url, "get_note", {"path": "new-note.md"})
        assert "Hello from E2E test" in text

    async def test_create_note_with_frontmatter(self, mcp_rw_url):
        text = await _mcp_call(mcp_rw_url, "create_note", {
            "path": "fm-note.md",
            "content": "Body text.",
            "frontmatter": '{"title": "FM Note", "tags": ["e2e"]}',
        })
        assert "Created" in text

        text = await _mcp_call(mcp_rw_url, "get_note", {"path": "fm-note.md"})
        assert "FM Note" in text
        assert "e2e" in text

    async def test_create_note_duplicate_fails(self, mcp_rw_url):
        await _mcp_call(mcp_rw_url, "create_note", {
            "path": "dup-note.md",
            "content": "first",
        })
        text = await _mcp_call(mcp_rw_url, "create_note", {
            "path": "dup-note.md",
            "content": "second",
        })
        assert "error" in text.lower() or "exists" in text.lower()

    async def test_append_to_note(self, mcp_rw_url):
        await _mcp_call(mcp_rw_url, "create_note", {
            "path": "append-target.md",
            "content": "Original.",
        })
        text = await _mcp_call(mcp_rw_url, "append_to_note", {
            "path": "append-target.md",
            "content": "\nAppended line.",
        })
        assert "Appended" in text

        text = await _mcp_call(mcp_rw_url, "get_note", {"path": "append-target.md"})
        assert "Original" in text
        assert "Appended line" in text

    async def test_prepend_to_note(self, mcp_rw_url):
        await _mcp_call(mcp_rw_url, "create_note", {
            "path": "prepend-target.md",
            "content": "Body text.",
            "frontmatter": '{"title": "Prepend Test"}',
        })
        text = await _mcp_call(mcp_rw_url, "prepend_to_note", {
            "path": "prepend-target.md",
            "content": "Prepended line.\n",
        })
        assert "Prepended" in text

        text = await _mcp_call(mcp_rw_url, "get_note", {"path": "prepend-target.md"})
        assert "Prepend Test" in text
        assert "Prepended line" in text
        assert text.index("Prepended line") < text.index("Body text")

    async def test_update_frontmatter(self, mcp_rw_url):
        await _mcp_call(mcp_rw_url, "create_note", {
            "path": "fm-update.md",
            "content": "Content.",
            "frontmatter": '{"status": "draft"}',
        })
        text = await _mcp_call(mcp_rw_url, "update_frontmatter", {
            "path": "fm-update.md",
            "properties": '{"status": "done", "priority": "high"}',
        })
        assert "Updated" in text

        text = await _mcp_call(mcp_rw_url, "get_note", {"path": "fm-update.md"})
        assert "done" in text
        assert "high" in text

    async def test_delete_note_soft(self, mcp_rw_url):
        await _mcp_call(mcp_rw_url, "create_note", {
            "path": "to-delete.md",
            "content": "Goodbye.",
        })
        text = await _mcp_call(mcp_rw_url, "delete_note", {"path": "to-delete.md"})
        assert "trash" in text.lower()

        text = await _mcp_call(mcp_rw_url, "get_note", {"path": "to-delete.md"})
        assert "not found" in text.lower()

    async def test_delete_note_permanent(self, mcp_rw_url):
        await _mcp_call(mcp_rw_url, "create_note", {
            "path": "to-perm-delete.md",
            "content": "Gone forever.",
        })
        text = await _mcp_call(mcp_rw_url, "delete_note", {
            "path": "to-perm-delete.md",
            "permanent": True,
        })
        assert "permanently" in text.lower()

    async def test_move_note(self, mcp_rw_url):
        await _mcp_call(mcp_rw_url, "create_note", {
            "path": "move-source.md",
            "content": "Moving this.",
        })
        text = await _mcp_call(mcp_rw_url, "move_note", {
            "old_path": "move-source.md",
            "new_path": "moved/move-dest.md",
        })
        assert "Moved" in text

        text = await _mcp_call(mcp_rw_url, "get_note", {"path": "move-source.md"})
        assert "not found" in text.lower()

        text = await _mcp_call(mcp_rw_url, "get_note", {"path": "moved/move-dest.md"})
        assert "Moving this" in text

    async def test_append_to_missing_note_fails(self, mcp_rw_url):
        text = await _mcp_call(mcp_rw_url, "append_to_note", {
            "path": "no-such-note.md",
            "content": "whatever",
        })
        assert "error" in text.lower() or "not found" in text.lower()


# ── canvas write operations (writable vault) ────────────────────────────


class TestCanvasWrite:
    async def test_create_canvas(self, mcp_rw_url):
        text = await _mcp_call(mcp_rw_url, "create_canvas", {
            "path": "e2e-canvas.canvas",
            "nodes": json.dumps([
                {"type": "text", "text": "Node A"},
                {"type": "text", "text": "Node B"},
            ]),
        })
        assert "Created" in text

        text = await _mcp_call(mcp_rw_url, "read_canvas", {"path": "e2e-canvas.canvas"})
        assert "Node A" in text
        assert "Node B" in text

    async def test_create_canvas_duplicate_fails(self, mcp_rw_url):
        await _mcp_call(mcp_rw_url, "create_canvas", {"path": "dup.canvas"})
        text = await _mcp_call(mcp_rw_url, "create_canvas", {"path": "dup.canvas"})
        assert "error" in text.lower() or "exists" in text.lower()

    async def test_add_and_update_canvas_node(self, mcp_rw_url):
        await _mcp_call(mcp_rw_url, "create_canvas", {"path": "node-ops.canvas"})

        text = await _mcp_call(mcp_rw_url, "add_canvas_node", {
            "canvas_path": "node-ops.canvas",
            "node_type": "text",
            "text": "Original text",
        })
        assert "Added" in text
        node_id = text.split("[")[1].split("]")[0]

        text = await _mcp_call(mcp_rw_url, "update_canvas_node", {
            "canvas_path": "node-ops.canvas",
            "node_id": node_id,
            "text": "Updated text",
            "color": "1",
        })
        assert "Updated" in text

        text = await _mcp_call(mcp_rw_url, "read_canvas", {"path": "node-ops.canvas"})
        assert "Updated text" in text

    async def test_add_edge_and_remove(self, mcp_rw_url):
        await _mcp_call(mcp_rw_url, "create_canvas", {"path": "edge-ops.canvas"})

        text_a = await _mcp_call(mcp_rw_url, "add_canvas_node", {
            "canvas_path": "edge-ops.canvas",
            "node_type": "text",
            "text": "From",
        })
        id_a = text_a.split("[")[1].split("]")[0]

        text_b = await _mcp_call(mcp_rw_url, "add_canvas_node", {
            "canvas_path": "edge-ops.canvas",
            "node_type": "text",
            "text": "To",
        })
        id_b = text_b.split("[")[1].split("]")[0]

        text = await _mcp_call(mcp_rw_url, "add_canvas_edge", {
            "canvas_path": "edge-ops.canvas",
            "from_node": id_a,
            "to_node": id_b,
            "label": "connects",
        })
        assert "Added edge" in text
        edge_id = text.split("[")[1].split("]")[0]

        text = await _mcp_call(mcp_rw_url, "update_canvas_edge", {
            "canvas_path": "edge-ops.canvas",
            "edge_id": edge_id,
            "label": "updated-label",
        })
        assert "Updated" in text

        text = await _mcp_call(mcp_rw_url, "remove_canvas_edges", {
            "canvas_path": "edge-ops.canvas",
            "edge_ids": json.dumps([edge_id]),
        })
        assert "Removed" in text

    async def test_remove_node_cleans_edges(self, mcp_rw_url):
        await _mcp_call(mcp_rw_url, "create_canvas", {"path": "cleanup.canvas"})

        text_a = await _mcp_call(mcp_rw_url, "add_canvas_node", {
            "canvas_path": "cleanup.canvas",
            "node_type": "text",
            "text": "Keep",
        })
        id_a = text_a.split("[")[1].split("]")[0]

        text_b = await _mcp_call(mcp_rw_url, "add_canvas_node", {
            "canvas_path": "cleanup.canvas",
            "node_type": "text",
            "text": "Remove me",
        })
        id_b = text_b.split("[")[1].split("]")[0]

        await _mcp_call(mcp_rw_url, "add_canvas_edge", {
            "canvas_path": "cleanup.canvas",
            "from_node": id_a,
            "to_node": id_b,
        })

        text = await _mcp_call(mcp_rw_url, "remove_canvas_nodes", {
            "canvas_path": "cleanup.canvas",
            "node_ids": json.dumps([id_b]),
        })
        assert "Removed 1 node" in text
        assert "dangling edge" in text.lower()


# ── daily note write operations ─────────────────────────────────────────


class TestDailyWrite:
    async def test_create_daily_note(self, mcp_rw_url):
        text = await _mcp_call(mcp_rw_url, "create_daily_note", {
            "date": "2099-12-31",
            "content": "Future daily note.",
        })
        assert "Created" in text

        text = await _mcp_call(mcp_rw_url, "get_daily_note", {"date": "2099-12-31"})
        assert "Future daily note" in text

    async def test_create_daily_note_duplicate_fails(self, mcp_rw_url):
        await _mcp_call(mcp_rw_url, "create_daily_note", {
            "date": "2099-01-01",
            "content": "First.",
        })
        text = await _mcp_call(mcp_rw_url, "create_daily_note", {
            "date": "2099-01-01",
            "content": "Second.",
        })
        assert "error" in text.lower() or "exists" in text.lower()


# ── cross-tool workflow ─────────────────────────────────────────────────


class TestWorkflow:
    async def test_create_link_move_verify(self, mcp_rw_url):
        """Create two linked notes, move one, verify links updated."""
        await _mcp_call(mcp_rw_url, "create_note", {
            "path": "wf-target.md",
            "content": "I am the target.",
            "frontmatter": '{"tags": ["workflow"]}',
        })
        await _mcp_call(mcp_rw_url, "create_note", {
            "path": "wf-source.md",
            "content": "Link to [[wf-target]].",
        })

        text = await _mcp_call(mcp_rw_url, "get_backlinks", {"path": "wf-target.md"})
        assert "wf-source" in text

        text = await _mcp_call(mcp_rw_url, "get_outlinks", {"path": "wf-source.md"})
        assert "wf-target" in text

        text = await _mcp_call(mcp_rw_url, "search", {
            "query": "I am the target",
            "mode": "exact",
        })
        assert "wf-target" in text

        await _mcp_call(mcp_rw_url, "move_note", {
            "old_path": "wf-target.md",
            "new_path": "workflow/wf-target-moved.md",
        })

        text = await _mcp_call(mcp_rw_url, "get_note", {"path": "wf-source.md"})
        assert "wf-target-moved" in text

        text = await _mcp_call(mcp_rw_url, "get_note", {"path": "wf-target.md"})
        assert "not found" in text.lower()

        text = await _mcp_call(mcp_rw_url, "get_note", {
            "path": "workflow/wf-target-moved.md",
        })
        assert "I am the target" in text

    async def test_note_lifecycle(self, mcp_rw_url):
        """Create → update frontmatter → append → prepend → delete."""
        await _mcp_call(mcp_rw_url, "create_note", {
            "path": "lifecycle.md",
            "content": "Initial body.",
            "frontmatter": '{"status": "draft"}',
        })

        await _mcp_call(mcp_rw_url, "update_frontmatter", {
            "path": "lifecycle.md",
            "properties": '{"status": "active", "priority": "high"}',
        })

        await _mcp_call(mcp_rw_url, "append_to_note", {
            "path": "lifecycle.md",
            "content": "\nAppended.",
        })

        await _mcp_call(mcp_rw_url, "prepend_to_note", {
            "path": "lifecycle.md",
            "content": "Prepended.\n",
        })

        text = await _mcp_call(mcp_rw_url, "get_note", {"path": "lifecycle.md"})
        assert "active" in text
        assert "high" in text
        assert "Prepended" in text
        assert "Initial body" in text
        assert "Appended" in text
        assert text.index("Prepended") < text.index("Initial body")
        assert text.index("Initial body") < text.index("Appended")

        text = await _mcp_call(mcp_rw_url, "delete_note", {
            "path": "lifecycle.md",
            "permanent": True,
        })
        assert "permanently" in text.lower()

        text = await _mcp_call(mcp_rw_url, "get_note", {"path": "lifecycle.md"})
        assert "not found" in text.lower()
