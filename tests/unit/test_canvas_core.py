"""Unit tests for core/canvas.py and canvas path utilities."""

import json

import pytest

from some_vault_some_mcp.core.canvas import (
    auto_position,
    find_edge,
    find_node,
    generate_canvas_id,
    parse_canvas,
    remove_dangling_edges,
    serialize_canvas,
)
from some_vault_some_mcp.core.paths import ensure_canvas_extension, walk_canvas
from some_vault_some_mcp.models import CanvasData, CanvasEdge, CanvasNode


VALID_CANVAS = json.dumps({
    "nodes": [
        {"id": "n1", "type": "text", "x": 0, "y": 0, "width": 250, "height": 60, "text": "Hello"},
        {"id": "n2", "type": "file", "x": 300, "y": 0, "width": 400, "height": 400, "file": "note.md"},
    ],
    "edges": [
        {"id": "e1", "fromNode": "n1", "toNode": "n2", "fromSide": "right", "toSide": "left"},
    ],
    "viewport": {"x": 0, "y": 0, "zoom": 1},
})


# --- parse_canvas ---

def test_parse_canvas_valid():
    canvas = parse_canvas(VALID_CANVAS)
    assert len(canvas.nodes) == 2
    assert len(canvas.edges) == 1
    assert canvas.nodes[0].type == "text"
    assert canvas.nodes[0].text == "Hello"
    assert canvas.nodes[1].type == "file"
    assert canvas.nodes[1].file == "note.md"


def test_parse_canvas_empty():
    canvas = parse_canvas('{"nodes":[],"edges":[]}')
    assert canvas.nodes == []
    assert canvas.edges == []


def test_parse_canvas_missing_edges():
    canvas = parse_canvas('{"nodes":[]}')
    assert canvas.edges == []


def test_parse_canvas_malformed_json():
    with pytest.raises(ValueError, match="Malformed"):
        parse_canvas("{not valid json}")


def test_parse_canvas_non_object():
    with pytest.raises(ValueError, match="root must be an object"):
        parse_canvas("[]")


def test_parse_canvas_preserves_extra_keys():
    canvas = parse_canvas(VALID_CANVAS)
    assert "viewport" in canvas.extra
    assert canvas.extra["viewport"]["zoom"] == 1


# --- serialize_canvas ---

def test_serialize_roundtrip():
    canvas = parse_canvas(VALID_CANVAS)
    serialized = serialize_canvas(canvas)
    reparsed = parse_canvas(serialized)
    assert len(reparsed.nodes) == len(canvas.nodes)
    assert len(reparsed.edges) == len(canvas.edges)
    for orig, rt in zip(canvas.nodes, reparsed.nodes):
        assert orig.id == rt.id
        assert orig.type == rt.type


def test_serialize_preserves_viewport():
    canvas = parse_canvas(VALID_CANVAS)
    serialized = serialize_canvas(canvas)
    data = json.loads(serialized)
    assert "viewport" in data
    assert data["viewport"]["zoom"] == 1


def test_serialize_excludes_none():
    canvas = CanvasData(
        nodes=[CanvasNode(id="n1", type="text", x=0, y=0, width=250, height=60, text="Hi")],
        edges=[],
    )
    serialized = serialize_canvas(canvas)
    data = json.loads(serialized)
    node = data["nodes"][0]
    assert "file" not in node
    assert "url" not in node
    assert "color" not in node


# --- generate_canvas_id ---

def test_generate_canvas_id_uniqueness():
    ids = {generate_canvas_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_generate_canvas_id_format():
    cid = generate_canvas_id()
    assert len(cid) == 16
    int(cid, 16)  # must be valid hex


# --- find_node / find_edge ---

def test_find_node_found():
    canvas = parse_canvas(VALID_CANVAS)
    node = find_node(canvas, "n1")
    assert node is not None
    assert node.text == "Hello"


def test_find_node_missing():
    canvas = parse_canvas(VALID_CANVAS)
    assert find_node(canvas, "nonexistent") is None


def test_find_edge_found():
    canvas = parse_canvas(VALID_CANVAS)
    edge = find_edge(canvas, "e1")
    assert edge is not None
    assert edge.fromNode == "n1"


def test_find_edge_missing():
    canvas = parse_canvas(VALID_CANVAS)
    assert find_edge(canvas, "nonexistent") is None


# --- remove_dangling_edges ---

def test_remove_dangling_edges():
    canvas = parse_canvas(VALID_CANVAS)
    removed = remove_dangling_edges(canvas, {"n1"})
    assert "e1" in removed
    assert len(canvas.edges) == 0


def test_remove_dangling_edges_preserves_unrelated():
    raw = json.dumps({
        "nodes": [
            {"id": "a", "type": "text", "x": 0, "y": 0, "width": 100, "height": 50, "text": "A"},
            {"id": "b", "type": "text", "x": 100, "y": 0, "width": 100, "height": 50, "text": "B"},
            {"id": "c", "type": "text", "x": 200, "y": 0, "width": 100, "height": 50, "text": "C"},
        ],
        "edges": [
            {"id": "e_ab", "fromNode": "a", "toNode": "b"},
            {"id": "e_bc", "fromNode": "b", "toNode": "c"},
        ],
    })
    canvas = parse_canvas(raw)
    removed = remove_dangling_edges(canvas, {"a"})
    assert "e_ab" in removed
    assert "e_bc" not in removed
    assert len(canvas.edges) == 1


# --- auto_position ---

def test_auto_position_empty_canvas():
    canvas = CanvasData()
    assert auto_position(canvas, 250, 60) == (0, 0)


def test_auto_position_single_node():
    canvas = CanvasData(
        nodes=[CanvasNode(id="n1", type="text", x=0, y=0, width=250, height=60, text="A")]
    )
    x, y = auto_position(canvas, 250, 60)
    assert x > 0 or y > 0  # not at origin (occupied)


def test_auto_position_fills_grid():
    canvas = CanvasData()
    positions = []
    for i in range(5):
        x, y = auto_position(canvas, 250, 60)
        positions.append((x, y))
        canvas.nodes.append(
            CanvasNode(id=f"n{i}", type="text", x=x, y=y, width=250, height=60, text=str(i))
        )
    # 4 cols wide, so 5th node wraps to row 2
    assert positions[4][1] > positions[0][1]
    # All positions unique
    assert len(set(positions)) == 5


# --- ensure_canvas_extension ---

def test_ensure_canvas_extension_appends():
    assert ensure_canvas_extension("test") == "test.canvas"


def test_ensure_canvas_extension_idempotent():
    assert ensure_canvas_extension("test.canvas") == "test.canvas"


def test_ensure_canvas_extension_case_insensitive():
    assert ensure_canvas_extension("test.CANVAS") == "test.CANVAS"


# --- walk_canvas ---

def test_walk_canvas(tmp_path):
    (tmp_path / "a.canvas").write_text("{}")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.canvas").write_text("{}")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "c.canvas").write_text("{}")

    result = walk_canvas(str(tmp_path))
    assert "a.canvas" in result
    assert "sub/b.canvas" in result
    assert all(".obsidian" not in p for p in result)


def test_walk_canvas_empty(tmp_path):
    assert walk_canvas(str(tmp_path)) == []


def test_roundtrip_preserves_unknown_node_and_edge_fields():
    """Unknown/future JSON-Canvas fields must survive parse -> serialize
    (plan fable-fixes, Phase 0.2 / O1). Regression: they were silently dropped."""
    raw = json.dumps({
        "nodes": [
            {"id": "g1", "type": "group", "x": 0, "y": 0, "width": 400, "height": 400,
             "label": "Grp", "background": "bg.png", "backgroundStyle": "cover"},
            {"id": "f1", "type": "file", "x": 0, "y": 0, "width": 200, "height": 100,
             "file": "note.md", "subpath": "#Heading"},
        ],
        "edges": [
            {"id": "e1", "fromNode": "g1", "toNode": "f1", "someFutureKey": "keepme"},
        ],
    })
    out = json.loads(serialize_canvas(parse_canvas(raw)))
    g1 = next(n for n in out["nodes"] if n["id"] == "g1")
    f1 = next(n for n in out["nodes"] if n["id"] == "f1")
    e1 = out["edges"][0]
    assert g1["background"] == "bg.png"
    assert g1["backgroundStyle"] == "cover"
    assert f1["subpath"] == "#Heading"
    assert e1["someFutureKey"] == "keepme"
