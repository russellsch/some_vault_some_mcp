"""Unit tests for canvas tools: list, read, create, node/edge CRUD."""

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from some_vault_some_mcp.tools.canvas import (
    add_canvas_edge,
    add_canvas_node,
    create_canvas,
    list_canvases,
    read_canvas,
    remove_canvas_edges,
    remove_canvas_nodes,
    update_canvas_edge,
    update_canvas_node,
)
from some_vault_some_mcp.core.paths import VaultPathError


FIXTURES = Path(__file__).parent.parent / "fixtures" / "vault"


@pytest.fixture()
def vault(tmp_path):
    vault_dir = tmp_path / "vault"
    shutil.copytree(str(FIXTURES), str(vault_dir))
    return str(vault_dir)


# --- list_canvases ---

def test_list_canvases(vault):
    result = list_canvases(vault)
    assert "test-canvas.canvas" in result


def test_list_canvases_folder_filter(vault):
    result = list_canvases(vault, folder="nonexistent")
    assert result == []


# --- read_canvas ---

def test_read_canvas(vault):
    canvas = read_canvas(vault, "test-canvas")
    assert canvas is not None
    assert len(canvas.nodes) == 3
    assert len(canvas.edges) == 1
    assert canvas.nodes[0].text == "Hello world"
    assert canvas.nodes[1].file == "simple.md"
    assert canvas.nodes[2].url == "https://example.com"


def test_read_canvas_with_extension(vault):
    canvas = read_canvas(vault, "test-canvas.canvas")
    assert canvas is not None
    assert len(canvas.nodes) == 3


def test_read_canvas_not_found(vault):
    assert read_canvas(vault, "nonexistent") is None


def test_read_canvas_preserves_viewport(vault):
    canvas = read_canvas(vault, "test-canvas")
    assert "viewport" in canvas.extra


# --- create_canvas ---

@pytest.mark.asyncio
async def test_create_canvas_empty(vault):
    path = await create_canvas(vault, "new-canvas")
    assert path == "new-canvas.canvas"
    assert (Path(vault) / "new-canvas.canvas").exists()
    data = json.loads((Path(vault) / "new-canvas.canvas").read_text())
    assert data["nodes"] == []
    assert data["edges"] == []


@pytest.mark.asyncio
async def test_create_canvas_with_nodes(vault):
    nodes = [
        {"type": "text", "text": "Node A"},
        {"type": "text", "text": "Node B"},
    ]
    await create_canvas(vault, "with-nodes", nodes=nodes)
    data = json.loads((Path(vault) / "with-nodes.canvas").read_text())
    assert len(data["nodes"]) == 2
    assert all("id" in n for n in data["nodes"])
    assert all("x" in n and "y" in n for n in data["nodes"])


@pytest.mark.asyncio
async def test_create_canvas_already_exists(vault):
    with pytest.raises(FileExistsError):
        await create_canvas(vault, "test-canvas")


@pytest.mark.asyncio
async def test_create_canvas_in_subfolder(vault):
    path = await create_canvas(vault, "sub/deep-canvas")
    assert path == "sub/deep-canvas.canvas"
    assert (Path(vault) / "sub" / "deep-canvas.canvas").exists()


# --- add_canvas_node ---

@pytest.mark.asyncio
async def test_add_node_text(vault):
    result = await add_canvas_node(vault, "test-canvas", "text", text="New node", x=500, y=500)
    assert "id" in result
    canvas = read_canvas(vault, "test-canvas")
    ids = [n.id for n in canvas.nodes]
    assert result["id"] in ids


@pytest.mark.asyncio
async def test_add_node_auto_layout(vault):
    result = await add_canvas_node(vault, "test-canvas", "text", text="Auto placed")
    assert result["x"] is not None
    assert result["y"] is not None


@pytest.mark.asyncio
async def test_add_node_link(vault):
    result = await add_canvas_node(vault, "test-canvas", "link", url="https://test.com")
    canvas = read_canvas(vault, "test-canvas")
    node = next(n for n in canvas.nodes if n.id == result["id"])
    assert node.url == "https://test.com"
    assert node.type == "link"


@pytest.mark.asyncio
async def test_add_node_file_validates_target(vault):
    with pytest.raises(FileNotFoundError, match="not found"):
        await add_canvas_node(vault, "test-canvas", "file", file="nonexistent.md")


@pytest.mark.asyncio
async def test_add_node_file_resolves_extensionless(vault):
    # Bare markdown name is resolved and stored with the .md extension
    result = await add_canvas_node(vault, "test-canvas", "file", file="simple")
    canvas = read_canvas(vault, "test-canvas")
    node = next(n for n in canvas.nodes if n.id == result["id"])
    assert node.file == "simple.md"


@pytest.mark.asyncio
async def test_add_node_file_resolves_basename(vault):
    # A bare basename resolves to a note nested in a folder (Obsidian-style)
    result = await add_canvas_node(vault, "test-canvas", "file", file="deep-note")
    canvas = read_canvas(vault, "test-canvas")
    node = next(n for n in canvas.nodes if n.id == result["id"])
    assert node.file == "projects/alpha/deep-note.md"


@pytest.mark.asyncio
async def test_add_node_file_attachment_kept_literal(vault):
    # Non-markdown attachments keep their literal path + extension
    (Path(vault) / "diagram.png").write_bytes(b"\x89PNG\r\n")
    result = await add_canvas_node(vault, "test-canvas", "file", file="diagram.png")
    canvas = read_canvas(vault, "test-canvas")
    node = next(n for n in canvas.nodes if n.id == result["id"])
    assert node.file == "diagram.png"


@pytest.mark.asyncio
async def test_create_canvas_file_node_resolves_extensionless(vault):
    nodes = [{"type": "file", "file": "simple"}]
    await create_canvas(vault, "fnode", nodes=nodes)
    data = json.loads((Path(vault) / "fnode.canvas").read_text())
    assert data["nodes"][0]["file"] == "simple.md"


@pytest.mark.asyncio
async def test_update_node_file_resolves_extensionless(vault):
    result = await add_canvas_node(vault, "test-canvas", "file", file="simple.md")
    await update_canvas_node(vault, "test-canvas", result["id"], file="deep-note")
    canvas = read_canvas(vault, "test-canvas")
    node = next(n for n in canvas.nodes if n.id == result["id"])
    assert node.file == "projects/alpha/deep-note.md"


@pytest.mark.asyncio
async def test_add_node_to_missing_canvas(vault):
    with pytest.raises(FileNotFoundError):
        await add_canvas_node(vault, "missing", "text", text="Fail")


@pytest.mark.asyncio
async def test_add_node_group(vault):
    result = await add_canvas_node(
        vault, "test-canvas", "group", label="My Group", width=600, height=400,
    )
    canvas = read_canvas(vault, "test-canvas")
    node = next(n for n in canvas.nodes if n.id == result["id"])
    assert node.type == "group"
    assert node.label == "My Group"
    assert node.width == 600


# --- update_canvas_node ---

@pytest.mark.asyncio
async def test_update_node_text(vault):
    await update_canvas_node(vault, "test-canvas", "node1aaa00000000", text="Updated text")
    canvas = read_canvas(vault, "test-canvas")
    node = next(n for n in canvas.nodes if n.id == "node1aaa00000000")
    assert node.text == "Updated text"


@pytest.mark.asyncio
async def test_update_node_position(vault):
    await update_canvas_node(vault, "test-canvas", "node1aaa00000000", x=999, y=888)
    canvas = read_canvas(vault, "test-canvas")
    node = next(n for n in canvas.nodes if n.id == "node1aaa00000000")
    assert node.x == 999
    assert node.y == 888


@pytest.mark.asyncio
async def test_update_node_not_found(vault):
    with pytest.raises(ValueError, match="Node not found"):
        await update_canvas_node(vault, "test-canvas", "nonexistent", text="Fail")


@pytest.mark.asyncio
async def test_update_node_no_changes(vault):
    with pytest.raises(ValueError, match="No properties"):
        await update_canvas_node(vault, "test-canvas", "node1aaa00000000")


# --- remove_canvas_nodes ---

@pytest.mark.asyncio
async def test_remove_nodes_basic(vault):
    result = await remove_canvas_nodes(vault, "test-canvas", ["node3ccc00000000"])
    assert "node3ccc00000000" in result["removed_nodes"]
    canvas = read_canvas(vault, "test-canvas")
    assert all(n.id != "node3ccc00000000" for n in canvas.nodes)


@pytest.mark.asyncio
async def test_remove_nodes_with_edge_cleanup(vault):
    result = await remove_canvas_nodes(vault, "test-canvas", ["node1aaa00000000"])
    assert "node1aaa00000000" in result["removed_nodes"]
    assert "edge1aaa00000000" in result["removed_edges"]
    canvas = read_canvas(vault, "test-canvas")
    assert len(canvas.edges) == 0


@pytest.mark.asyncio
async def test_remove_nodes_partial_not_found(vault):
    result = await remove_canvas_nodes(vault, "test-canvas", ["node3ccc00000000", "bogus"])
    assert "node3ccc00000000" in result["removed_nodes"]
    assert "bogus" in result["not_found"]


@pytest.mark.asyncio
async def test_remove_nodes_preserves_viewport(vault):
    await remove_canvas_nodes(vault, "test-canvas", ["node3ccc00000000"])
    canvas = read_canvas(vault, "test-canvas")
    assert "viewport" in canvas.extra


# --- add_canvas_edge ---

@pytest.mark.asyncio
async def test_add_edge(vault):
    edge_id = await add_canvas_edge(
        vault, "test-canvas", "node1aaa00000000", "node3ccc00000000",
        from_side="bottom", to_side="top", label="test link",
    )
    canvas = read_canvas(vault, "test-canvas")
    edge = next(e for e in canvas.edges if e.id == edge_id)
    assert edge.fromNode == "node1aaa00000000"
    assert edge.toNode == "node3ccc00000000"
    assert edge.fromSide == "bottom"
    assert edge.label == "test link"


@pytest.mark.asyncio
async def test_add_edge_with_arrow_ends(vault):
    edge_id = await add_canvas_edge(
        vault, "test-canvas", "node1aaa00000000", "node3ccc00000000",
        from_end="none", to_end="arrow", color="3",
    )
    canvas = read_canvas(vault, "test-canvas")
    edge = next(e for e in canvas.edges if e.id == edge_id)
    assert edge.fromEnd == "none"
    assert edge.toEnd == "arrow"
    assert edge.color == "3"


@pytest.mark.asyncio
async def test_add_edge_invalid_from_node(vault):
    with pytest.raises(ValueError, match="fromNode not found"):
        await add_canvas_edge(vault, "test-canvas", "bogus", "node1aaa00000000")


@pytest.mark.asyncio
async def test_add_edge_invalid_to_node(vault):
    with pytest.raises(ValueError, match="toNode not found"):
        await add_canvas_edge(vault, "test-canvas", "node1aaa00000000", "bogus")


# --- update_canvas_edge ---

@pytest.mark.asyncio
async def test_update_edge(vault):
    await update_canvas_edge(vault, "test-canvas", "edge1aaa00000000", label="renamed", color="5")
    canvas = read_canvas(vault, "test-canvas")
    edge = next(e for e in canvas.edges if e.id == "edge1aaa00000000")
    assert edge.label == "renamed"
    assert edge.color == "5"


@pytest.mark.asyncio
async def test_update_edge_not_found(vault):
    with pytest.raises(ValueError, match="Edge not found"):
        await update_canvas_edge(vault, "test-canvas", "bogus", label="Fail")


@pytest.mark.asyncio
async def test_update_edge_no_changes(vault):
    with pytest.raises(ValueError, match="No properties"):
        await update_canvas_edge(vault, "test-canvas", "edge1aaa00000000")


# --- remove_canvas_edges ---

@pytest.mark.asyncio
async def test_remove_edges(vault):
    result = await remove_canvas_edges(vault, "test-canvas", ["edge1aaa00000000"])
    assert "edge1aaa00000000" in result["removed"]
    canvas = read_canvas(vault, "test-canvas")
    assert len(canvas.edges) == 0


@pytest.mark.asyncio
async def test_remove_edges_not_found(vault):
    result = await remove_canvas_edges(vault, "test-canvas", ["bogus"])
    assert "bogus" in result["not_found"]
    assert result["removed"] == []


# --- concurrent writes ---

@pytest.mark.asyncio
async def test_concurrent_canvas_writes(vault):
    tasks = [
        add_canvas_node(vault, "test-canvas", "text", text=f"Concurrent {i}")
        for i in range(5)
    ]
    results = await asyncio.gather(*tasks)
    ids = [r["id"] for r in results]
    assert len(set(ids)) == 5

    canvas = read_canvas(vault, "test-canvas")
    assert len(canvas.nodes) == 8  # 3 original + 5 new


# --- path traversal ---

@pytest.mark.asyncio
async def test_path_traversal_rejected(vault):
    with pytest.raises(ValueError):
        await create_canvas(vault, "../../../etc/passwd")


def test_read_canvas_path_traversal(vault):
    result = read_canvas(vault, "../../../etc/passwd")
    assert result is None


@pytest.mark.asyncio
async def test_create_canvas_blocked_suffix_rejected(vault):
    with pytest.raises(ValueError):
        await create_canvas(vault, "evil.backup", blocked_suffixes=[".backup"],
                            blocked_message="blocked")
    assert not (Path(vault) / "evil.backup.canvas").exists()


@pytest.mark.asyncio
async def test_create_canvas_exclusive_no_clobber(vault):
    await create_canvas(vault, "dup", nodes=[{"type": "text", "text": "one"}])
    with pytest.raises(FileExistsError):
        await create_canvas(vault, "dup", nodes=[{"type": "text", "text": "two"}])
    assert "one" in (Path(vault) / "dup.canvas").read_text()
