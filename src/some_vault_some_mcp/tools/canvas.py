"""Canvas tools: list, read, create, add/update/remove nodes, add/update/remove edges."""

import logging
from pathlib import Path

from some_vault_some_mcp.core.atomic_write import atomic_write, with_file_lock
from some_vault_some_mcp.core.canvas import (
    auto_position,
    find_edge,
    find_node,
    generate_canvas_id,
    parse_canvas,
    remove_dangling_edges,
    serialize_canvas,
)
from some_vault_some_mcp.core.paths import (
    VaultPathError,
    ensure_canvas_extension,
    resolve_note_path,
    resolve_vault_path,
    walk_canvas,
    walk_vault,
)
from some_vault_some_mcp.models import CanvasData, CanvasEdge, CanvasNode

logger = logging.getLogger(__name__)


def list_canvases(vault_path: str, folder: str | None = None) -> list[str]:
    paths = walk_canvas(vault_path)
    if folder:
        prefix = folder.rstrip("/") + "/"
        paths = [p for p in paths if p.startswith(prefix) or p == folder]
    return paths


def read_canvas(vault_path: str, path: str) -> CanvasData | None:
    resolved_path = ensure_canvas_extension(path)
    try:
        full_path = resolve_vault_path(vault_path, resolved_path)
    except VaultPathError:
        return None

    p = Path(full_path)
    if not p.exists():
        return None

    raw = p.read_text(encoding="utf-8", errors="replace")
    return parse_canvas(raw)


async def create_canvas(
    vault_path: str,
    path: str,
    nodes: list[dict] | None = None,
    edges: list[dict] | None = None,
) -> str:
    resolved_path = ensure_canvas_extension(path)
    try:
        full_path = resolve_vault_path(vault_path, resolved_path)
    except VaultPathError as e:
        raise ValueError(str(e))

    canvas = CanvasData()

    if nodes:
        for nd in nodes:
            nd.setdefault("id", generate_canvas_id())
            nd.setdefault("width", 250)
            nd.setdefault("height", 60)
            if "x" not in nd or "y" not in nd:
                x, y = auto_position(canvas, nd["width"], nd["height"])
                nd.setdefault("x", x)
                nd.setdefault("y", y)
            if nd.get("type") == "file" and nd.get("file"):
                nd["file"] = _resolve_file_node_target(vault_path, nd["file"])
            canvas.nodes.append(CanvasNode(**nd))

    if edges:
        for ed in edges:
            ed.setdefault("id", generate_canvas_id())
            canvas.edges.append(CanvasEdge(**ed))

    content = serialize_canvas(canvas)

    async def _create():
        p = Path(full_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise FileExistsError(f"Canvas already exists at '{resolved_path}'")
        await atomic_write(full_path, content)

    await with_file_lock(full_path, _create)
    return resolved_path


def _resolve_file_node_target(vault_path: str, file_ref: str) -> str:
    """Resolve a canvas file-node target to the vault-relative path to store.

    Obsidian stores the full vault-relative path with extension, so the
    literal reference is kept when it exists on disk (this covers attachments
    like images/PDFs, which must keep their extension). An extensionless
    markdown name is resolved Obsidian-style to its '.md' path. Raises if the
    target can't be resolved to an existing file.
    """
    try:
        full = resolve_vault_path(vault_path, file_ref)
    except VaultPathError as e:
        raise ValueError(f"Invalid file node target: {e}")
    if Path(full).exists():
        return file_ref
    resolved = resolve_note_path(file_ref, walk_vault(vault_path))
    if resolved is None:
        raise FileNotFoundError(f"File node target not found: {file_ref}")
    return resolved


async def add_canvas_node(
    vault_path: str,
    canvas_path: str,
    node_type: str,
    text: str | None = None,
    file: str | None = None,
    url: str | None = None,
    label: str | None = None,
    x: int | None = None,
    y: int | None = None,
    width: int = 250,
    height: int = 60,
    color: str | None = None,
) -> dict:
    resolved_path = ensure_canvas_extension(canvas_path)
    try:
        full_path = resolve_vault_path(vault_path, resolved_path)
    except VaultPathError as e:
        raise ValueError(str(e))

    if node_type == "file" and file:
        file = _resolve_file_node_target(vault_path, file)

    node_id = generate_canvas_id()
    result = {}

    async def _add():
        nonlocal result
        p = Path(full_path)
        if not p.exists():
            raise FileNotFoundError(f"Canvas not found: {resolved_path}")

        raw = p.read_text(encoding="utf-8", errors="replace")
        canvas = parse_canvas(raw)

        pos_x, pos_y = x, y
        if pos_x is None or pos_y is None:
            auto_x, auto_y = auto_position(canvas, width, height)
            pos_x = pos_x if pos_x is not None else auto_x
            pos_y = pos_y if pos_y is not None else auto_y

        node = CanvasNode(
            id=node_id, type=node_type,
            x=pos_x, y=pos_y, width=width, height=height,
            color=color, text=text, file=file, url=url, label=label,
        )
        canvas.nodes.append(node)
        await atomic_write(full_path, serialize_canvas(canvas))
        result = {"id": node_id, "x": pos_x, "y": pos_y}

    await with_file_lock(full_path, _add)
    return result


async def update_canvas_node(
    vault_path: str,
    canvas_path: str,
    node_id: str,
    x: int | None = None,
    y: int | None = None,
    width: int | None = None,
    height: int | None = None,
    color: str | None = None,
    text: str | None = None,
    file: str | None = None,
    url: str | None = None,
    label: str | None = None,
) -> None:
    resolved_path = ensure_canvas_extension(canvas_path)
    try:
        full_path = resolve_vault_path(vault_path, resolved_path)
    except VaultPathError as e:
        raise ValueError(str(e))

    if file is not None:
        file = _resolve_file_node_target(vault_path, file)

    updates = {
        k: v for k, v in {
            "x": x, "y": y, "width": width, "height": height,
            "color": color, "text": text, "file": file, "url": url, "label": label,
        }.items() if v is not None
    }
    if not updates:
        raise ValueError("No properties to update")

    async def _update():
        p = Path(full_path)
        if not p.exists():
            raise FileNotFoundError(f"Canvas not found: {resolved_path}")

        raw = p.read_text(encoding="utf-8", errors="replace")
        canvas = parse_canvas(raw)
        node = find_node(canvas, node_id)
        if node is None:
            raise ValueError(f"Node not found: {node_id}")

        for k, v in updates.items():
            setattr(node, k, v)

        await atomic_write(full_path, serialize_canvas(canvas))

    await with_file_lock(full_path, _update)


async def remove_canvas_nodes(
    vault_path: str,
    canvas_path: str,
    node_ids: list[str],
) -> dict:
    resolved_path = ensure_canvas_extension(canvas_path)
    try:
        full_path = resolve_vault_path(vault_path, resolved_path)
    except VaultPathError as e:
        raise ValueError(str(e))

    result = {}

    async def _remove():
        nonlocal result
        p = Path(full_path)
        if not p.exists():
            raise FileNotFoundError(f"Canvas not found: {resolved_path}")

        raw = p.read_text(encoding="utf-8", errors="replace")
        canvas = parse_canvas(raw)

        ids_to_remove = set(node_ids)
        existing_ids = {n.id for n in canvas.nodes}
        not_found = list(ids_to_remove - existing_ids)
        actual_remove = ids_to_remove & existing_ids

        canvas.nodes = [n for n in canvas.nodes if n.id not in actual_remove]
        removed_edges = remove_dangling_edges(canvas, actual_remove)

        await atomic_write(full_path, serialize_canvas(canvas))
        result = {
            "removed_nodes": sorted(actual_remove),
            "removed_edges": removed_edges,
            "not_found": not_found,
        }

    await with_file_lock(full_path, _remove)
    return result


async def add_canvas_edge(
    vault_path: str,
    canvas_path: str,
    from_node: str,
    to_node: str,
    from_side: str | None = None,
    to_side: str | None = None,
    from_end: str | None = None,
    to_end: str | None = None,
    color: str | None = None,
    label: str | None = None,
) -> str:
    resolved_path = ensure_canvas_extension(canvas_path)
    try:
        full_path = resolve_vault_path(vault_path, resolved_path)
    except VaultPathError as e:
        raise ValueError(str(e))

    edge_id = generate_canvas_id()

    async def _add():
        p = Path(full_path)
        if not p.exists():
            raise FileNotFoundError(f"Canvas not found: {resolved_path}")

        raw = p.read_text(encoding="utf-8", errors="replace")
        canvas = parse_canvas(raw)

        node_ids = {n.id for n in canvas.nodes}
        if from_node not in node_ids:
            raise ValueError(f"fromNode not found: {from_node}")
        if to_node not in node_ids:
            raise ValueError(f"toNode not found: {to_node}")

        edge = CanvasEdge(
            id=edge_id, fromNode=from_node, toNode=to_node,
            fromSide=from_side, toSide=to_side,
            fromEnd=from_end, toEnd=to_end,
            color=color, label=label,
        )
        canvas.edges.append(edge)
        await atomic_write(full_path, serialize_canvas(canvas))

    await with_file_lock(full_path, _add)
    return edge_id


async def update_canvas_edge(
    vault_path: str,
    canvas_path: str,
    edge_id: str,
    from_side: str | None = None,
    to_side: str | None = None,
    from_end: str | None = None,
    to_end: str | None = None,
    color: str | None = None,
    label: str | None = None,
) -> None:
    resolved_path = ensure_canvas_extension(canvas_path)
    try:
        full_path = resolve_vault_path(vault_path, resolved_path)
    except VaultPathError as e:
        raise ValueError(str(e))

    updates = {
        k: v for k, v in {
            "fromSide": from_side, "toSide": to_side,
            "fromEnd": from_end, "toEnd": to_end,
            "color": color, "label": label,
        }.items() if v is not None
    }
    if not updates:
        raise ValueError("No properties to update")

    async def _update():
        p = Path(full_path)
        if not p.exists():
            raise FileNotFoundError(f"Canvas not found: {resolved_path}")

        raw = p.read_text(encoding="utf-8", errors="replace")
        canvas = parse_canvas(raw)
        edge = find_edge(canvas, edge_id)
        if edge is None:
            raise ValueError(f"Edge not found: {edge_id}")

        for k, v in updates.items():
            setattr(edge, k, v)

        await atomic_write(full_path, serialize_canvas(canvas))

    await with_file_lock(full_path, _update)


async def remove_canvas_edges(
    vault_path: str,
    canvas_path: str,
    edge_ids: list[str],
) -> dict:
    resolved_path = ensure_canvas_extension(canvas_path)
    try:
        full_path = resolve_vault_path(vault_path, resolved_path)
    except VaultPathError as e:
        raise ValueError(str(e))

    result = {}

    async def _remove():
        nonlocal result
        p = Path(full_path)
        if not p.exists():
            raise FileNotFoundError(f"Canvas not found: {resolved_path}")

        raw = p.read_text(encoding="utf-8", errors="replace")
        canvas = parse_canvas(raw)

        ids_to_remove = set(edge_ids)
        existing_ids = {e.id for e in canvas.edges}
        not_found = list(ids_to_remove - existing_ids)
        actual_remove = ids_to_remove & existing_ids

        canvas.edges = [e for e in canvas.edges if e.id not in actual_remove]

        await atomic_write(full_path, serialize_canvas(canvas))
        result = {
            "removed": sorted(actual_remove),
            "not_found": not_found,
        }

    await with_file_lock(full_path, _remove)
    return result
