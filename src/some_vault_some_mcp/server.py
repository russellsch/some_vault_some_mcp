"""FastMCP server: tool registration with override support + MCP resources."""

import datetime
import json
import logging
import os
import threading
from pathlib import Path

from fastmcp import FastMCP

from some_vault_some_mcp.config import VaultMcpConfig, apply_override
from some_vault_some_mcp.core.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class IndexGate:
    """Signals whether the search index is ready for queries.

    Written once from the background indexing thread; read from tool handlers.
    """

    def __init__(self):
        self._ready = threading.Event()
        self._error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self._error is None

    @property
    def error(self) -> str | None:
        return self._error

    def set_ready(self):
        self._ready.set()

    def set_failed(self, error: str):
        self._error = error
        self._ready.set()


def _check_index_gate(gate: "IndexGate | None", tool_name: str) -> str | None:
    """Return an error message if the index is unavailable, else None."""
    if gate is None or gate.is_ready:
        return None
    if gate.error:
        return (
            f"Tool '{tool_name}' is unavailable: initial indexing failed.\n"
            f"Error: {gate.error}\n"
            f"Restart the server to retry."
        )
    return (
        f"Tool '{tool_name}' is unavailable: the search index is being built.\n"
        f"You can use: get_note, search (with mode='exact'), get_tags, "
        f"get_backlinks, get_outlinks, and all write/canvas tools.\n"
        f"Call vault_index_status to check progress."
    )


def _json_default(obj):
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def build_server(config: VaultMcpConfig, provider: EmbeddingProvider, gate: IndexGate | None = None) -> FastMCP:
    """Create and configure the FastMCP server with all tools registered."""
    mcp = FastMCP("some-vault-some-mcp")

    vault_path = config.vault_path
    db_path = config.db_path
    overrides = config.tool_overrides
    disabled = config.disabled_tools

    def _reg(default_name: str, default_desc: str, fn, **kwargs):
        """Register a tool with override applied, skip if disabled."""
        if default_name in disabled:
            logger.info(f"Tool '{default_name}' disabled by override file")
            return
        name, desc = apply_override(default_name, default_desc, overrides)
        mcp.tool(name=name, description=desc, **kwargs)(fn)

    # ── search ──────────────────────────────────────────────────────────
    def search(
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        tags: list[str] | None = None,
        folder: str | None = None,
        case_sensitive: bool = False,
    ):
        """Find notes by text, meaning, or exact string.

        mode: hybrid (default), semantic, or exact.
        """
        from some_vault_some_mcp.tools.search import hybrid_search, semantic_search, exact_search
        if mode in ("hybrid", "semantic"):
            gate_msg = _check_index_gate(gate, "search")
            if gate_msg:
                return gate_msg + "\nYou can call this tool with mode='exact' to search without the index."
        if mode == "semantic":
            try:
                results = semantic_search(query, db_path, provider, top_k, tags, folder)
            except Exception as e:
                return f"Embedding provider error: {e}\nUse mode='exact' or retry when Ollama is back."
            if not results:
                return "No results found."
            lines = [f"Found {len(results)} results (semantic):\n"]
            for i, r in enumerate(results, 1):
                heading = f" > {r.heading}" if r.heading else ""
                lines.append(f"**{i}. {r.title or r.file_path}**{heading}")
                lines.append(f"   Path: {r.file_path}")
                lines.append(f"   Score: {r.score:.3f}")
                lines.append(f"   {r.snippet}...")
                lines.append("")
            return "\n".join(lines)

        elif mode == "exact":
            results = exact_search(query, vault_path, top_k, folder, case_sensitive, tags)
            if not results:
                return f'No results found for "{query}"'
            lines = [f'Found {len(results)} result(s) for "{query}":\n']
            for r in results:
                lines.append(f"## {r.relative_path}")
                for m in r.matches:
                    lines.append(f"  Line {m.line}: {m.content}")
                lines.append("")
            return "\n".join(lines)

        else:  # hybrid (default)
            try:
                results = hybrid_search(query, db_path, provider, top_k, tags, folder)
            except Exception as e:
                return f"Embedding provider error: {e}\nUse mode='exact' or retry when Ollama is back."
            if not results:
                return "No results found."
            lines = [f"Found {len(results)} results (hybrid):\n"]
            for i, r in enumerate(results, 1):
                heading = f" > {r.heading}" if r.heading else ""
                lines.append(f"**{i}. {r.title or r.file_path}**{heading}")
                lines.append(f"   Path: {r.file_path}")
                lines.append(f"   Score: {r.score:.3f}")
                lines.append(f"   {r.snippet}...")
                lines.append("")
            return "\n".join(lines)

    _reg("search", "Find notes by text, meaning, or exact string.", search)

    # ── read ─────────────────────────────────────────────────────────────
    def get_note(path: str):
        """Read the full content of a note by vault-relative path.

        The .md extension is optional ("todo" and "todo.md" both work), and a
        bare note name resolves to a matching note anywhere in the vault.
        """
        from some_vault_some_mcp.tools.read import get_note as _get_note
        note = _get_note(vault_path, path)
        if note is None:
            return f"Note not found: {path}"
        header = []
        if note.frontmatter:
            header.append("--- Frontmatter ---")
            for k, v in note.frontmatter.items():
                header.append(f"{k}: {json.dumps(v, default=_json_default)}")
            header.append("--- End Frontmatter ---")
            header.append("")
        if note.tags:
            header.append(f"Tags: {', '.join(note.tags)}")
            header.append("")
        return "\n".join(header) + note.content

    _reg("get_note", "Read a single note with parsed frontmatter and tags. Path extension is optional.", get_note)

    def list_notes(
        folder: str | None = None,
        tags: list[str] | None = None,
        projects: list[str] | None = None,
        status: str | None = None,
        area: str | None = None,
        frontmatter_property: str | None = None,
        frontmatter_value: str | None = None,
        include_content: bool = False,
        limit: int = 50,
    ):
        """Enumerate vault notes with optional metadata filters."""
        if tags or projects or status or area:
            gate_msg = _check_index_gate(gate, "list_notes")
            if gate_msg:
                return gate_msg + "\nFiltering by tags/projects/status/area requires the index. Use list_notes with a folder filter or get_tags instead."
        from some_vault_some_mcp.tools.read import list_notes as _list_notes
        results, total = _list_notes(
            vault_path, db_path, folder, tags, projects, status, area,
            frontmatter_property, frontmatter_value, include_content, limit,
        )
        lines = [f"Found {total} note(s){f' (showing first {limit})' if total > limit else ''}:\n"]
        for r in results:
            tags_str = f" [{', '.join(r.tags)}]" if r.tags else ""
            status_str = f" ({r.status})" if r.status else ""
            lines.append(f"- {r.file_path}{status_str}{tags_str}")
        return "\n".join(lines)

    _reg(
        "list_notes",
        "List and filter vault notes by folder, tags, projects, status, area, or frontmatter.",
        list_notes,
    )

    # ── write ─────────────────────────────────────────────────────────────
    async def create_note(path: str, content: str, frontmatter: str | None = None):
        """Create a new note. Fails if it already exists.

        frontmatter: JSON string of frontmatter fields, e.g. '{"title":"My Note","tags":["idea"]}'.
        """
        from some_vault_some_mcp.tools.write import create_note as _create_note
        fm = None
        if frontmatter:
            try:
                fm = json.loads(frontmatter)
            except json.JSONDecodeError:
                return "Error: Invalid JSON in frontmatter parameter."
        try:
            resolved = await _create_note(vault_path, path, content, fm)
            return f"Created note at '{resolved}'."
        except FileExistsError as e:
            return f"Error: {e} Use append or update tools instead."
        except Exception as e:
            return f"Error creating note: {e}"

    _reg("create_note", "Create a new note. Fails if it already exists.", create_note)

    async def append_to_note(path: str, content: str):
        """Append text to end of an existing note."""
        from some_vault_some_mcp.tools.write import append_to_note as _append
        try:
            await _append(vault_path, path, content)
            return f"Appended content to '{path}'."
        except FileNotFoundError:
            return f"Error: Note not found: {path}"
        except Exception as e:
            return f"Error appending: {e}"

    _reg("append_to_note", "Append text to the end of an existing note.", append_to_note)

    async def prepend_to_note(path: str, content: str):
        """Insert content after frontmatter, before body."""
        from some_vault_some_mcp.tools.write import prepend_to_note as _prepend
        try:
            await _prepend(vault_path, path, content)
            return f"Prepended content to '{path}'."
        except FileNotFoundError:
            return f"Error: Note not found: {path}"
        except Exception as e:
            return f"Error prepending: {e}"

    _reg("prepend_to_note", "Insert text after frontmatter, before the note body.", prepend_to_note)

    async def update_frontmatter(path: str, properties: str):
        """Merge key-value pairs into YAML frontmatter.

        properties: JSON string of key-value pairs, e.g. '{"status":"done","tags":["review"]}'.
        """
        from some_vault_some_mcp.tools.write import update_note_frontmatter
        try:
            parsed = json.loads(properties)
        except json.JSONDecodeError:
            return "Error: Invalid JSON in properties parameter."
        try:
            count = await update_note_frontmatter(vault_path, path, parsed)
            return f"Updated frontmatter of '{path}' with {count} properties."
        except FileNotFoundError:
            return f"Error: Note not found: {path}"
        except Exception as e:
            return f"Error updating frontmatter: {e}"

    _reg("update_frontmatter", "Merge key-value pairs into YAML frontmatter. Unlisted keys preserved.", update_frontmatter)

    async def move_note(old_path: str, new_path: str, update_links: bool = True):
        """Move or rename a note, optionally rewriting wikilinks."""
        from some_vault_some_mcp.tools.write import move_note as _move
        try:
            result = await _move(vault_path, old_path, new_path, update_links)
            lines = [f"Moved '{old_path}' to '{new_path}'."]
            n_updated = len(result["updated_referrers"])
            if update_links:
                lines.append(
                    f"Updated references in {n_updated} file(s)."
                    if n_updated else "No other notes referenced this file."
                )
            return "\n".join(lines)
        except (FileNotFoundError, FileExistsError, ValueError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error moving note: {e}"

    _reg("move_note", "Move or rename a note. Rewrites wikilinks vault-wide by default.", move_note)

    async def delete_note(path: str, permanent: bool = False):
        """Delete a note. Default: move to .trash. permanent=True: hard delete."""
        from some_vault_some_mcp.tools.write import delete_note as _delete
        try:
            effective = permanent or config.soft_delete_is_permanent
            await _delete(vault_path, path, effective)
            method = "permanently deleted" if effective else "moved to trash"
            return f"Note '{path}' {method}."
        except FileNotFoundError:
            return f"Error: Note not found: {path}"
        except Exception as e:
            return f"Error deleting note: {e}"

    _reg("delete_note", "Delete a note. Moves to .trash by default; use permanent for hard delete.", delete_note)

    # ── daily ─────────────────────────────────────────────────────────────
    def get_daily_note(date: str | None = None):
        """Read today's (or a date's) daily note."""
        from some_vault_some_mcp.tools.daily import get_daily_note as _get_daily
        result = _get_daily(vault_path, date)
        if result is None:
            target = date or "today"
            return f"Daily note not found for {target}."
        lines = [f"Daily Note: {result['date']}", f"Path: {result['path']}", ""]
        if result["frontmatter"]:
            lines.append("--- Frontmatter ---")
            for k, v in result["frontmatter"].items():
                lines.append(f"{k}: {json.dumps(v, default=_json_default)}")
            lines.append("--- End Frontmatter ---")
            lines.append("")
        lines.append(result["content"])
        return "\n".join(lines)

    _reg("get_daily_note", "Read today's or a specific date's daily note.", get_daily_note)

    async def create_daily_note(
        date: str | None = None,
        content: str | None = None,
        template_path: str | None = None,
    ):
        """Create a daily note from optional template."""
        from some_vault_some_mcp.tools.daily import create_daily_note as _create_daily
        try:
            path = await _create_daily(vault_path, date, content, template_path)
            return f"Created daily note at '{path}'."
        except FileExistsError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error creating daily note: {e}"

    _reg("create_daily_note", "Create a daily note for today or a given date. Fails if one exists.", create_daily_note)

    # ── tags ──────────────────────────────────────────────────────────────
    def get_tags(sort_by: str = "count"):
        """Enumerate all unique tags with usage counts."""
        from some_vault_some_mcp.tools.tags import get_tags as _get_tags
        tag_list = _get_tags(vault_path, sort_by)
        lines = [f"Total unique tags: {len(tag_list)}", ""]
        for item in tag_list:
            n = item["count"]
            lines.append(f"#{item['tag']} ({n} {'note' if n == 1 else 'notes'})")
        return "\n".join(lines)

    _reg("get_tags", "List all unique tags in the vault with usage counts.", get_tags)

    # ── links ─────────────────────────────────────────────────────────────
    def get_backlinks(path: str):
        """List notes that link to a given note."""
        from some_vault_some_mcp.tools.links import get_backlinks as _backlinks
        results = _backlinks(vault_path, path)
        if not results:
            return f"No backlinks found for: {path}"
        lines = [f"Backlinks to: {path}", f"Found: {len(results)} backlink(s)\n"]
        for r in results:
            line_str = f":{r['line']}" if r["line"] > 0 else ""
            ctx = f"  → {r['context']}" if r["context"] else ""
            lines.append(f"- {r['source']}{line_str}{ctx}")
        return "\n".join(lines)

    _reg("get_backlinks", "Find notes that link to a given note.", get_backlinks)

    def get_outlinks(path: str):
        """List all outgoing wikilinks from a note."""
        from some_vault_some_mcp.tools.links import get_outlinks as _outlinks
        try:
            result = _outlinks(vault_path, path)
        except FileNotFoundError:
            return f"Note not found: {path}"
        valid = result["valid"]
        broken = result["broken"]
        embeds = result["embeds"]
        total = len(valid) + len(broken) + len(embeds)
        lines = [
            f"Outgoing links from: {path}",
            f"Total: {total} ({len(valid)} valid, {len(broken)} broken)\n",
        ]
        if valid:
            lines.append("Valid links:")
            for r in valid:
                lines.append(f"  [[{r['target']}]] → {r['resolved_path']}")
        if broken:
            lines.append("\nBroken links:")
            for r in broken:
                lines.append(f"  [[{r['target']}]] → (not found)")
        if embeds:
            lines.append("\nEmbeds:")
            for r in embeds:
                lines.append(f"  ![[{r['target']}]] → {r['resolved_path'] or '(not found)'}")
        return "\n".join(lines)

    _reg("get_outlinks", "List all outgoing wikilinks from a note. Shows valid and broken links.", get_outlinks)

    def find_orphans(include_outlinks_check: bool = True, max_results: int = 200):
        """Find notes with no connections."""
        from some_vault_some_mcp.tools.links import find_orphans as _orphans
        result = _orphans(vault_path, include_outlinks_check, max_results)
        lines = [f"Orphan analysis ({result['total_notes']} notes total)\n"]
        lines.append(f"Fully isolated (no links in or out): {result['fully_isolated_total']}")
        for p in result["fully_isolated"]:
            lines.append(f"  - {p}")
        lines.append(f"\nNo backlinks: {result['no_backlinks_total']}")
        for p in result["no_backlinks"]:
            lines.append(f"  - {p}")
        if include_outlinks_check:
            lines.append(f"\nNo outlinks: {result['no_outlinks_total']}")
            for p in result["no_outlinks"]:
                lines.append(f"  - {p}")
        return "\n".join(lines)

    _reg("find_orphans", "Find disconnected notes — no inbound links, no outbound links, or both.", find_orphans)

    def find_broken_links(folder: str | None = None, max_results: int = 200):
        """Find wikilinks pointing at non-existent notes."""
        from some_vault_some_mcp.tools.links import find_broken_links as _broken
        results = _broken(vault_path, folder, max_results)
        if not results:
            scope = f" in folder: {folder}" if folder else ""
            return f"No broken links found{scope}"
        lines = [f"Broken links report{f' (folder: {folder})' if folder else ''}\n"]
        by_source: dict[str, list] = {}
        for r in results:
            by_source.setdefault(r["source_path"], []).append(r)
        for src, items in by_source.items():
            lines.append(f"{src}:")
            for item in items:
                line_str = f" (line {item['line']})" if item["line"] > 0 else ""
                lines.append(f"  - [[{item['target_link']}]]{line_str}")
            lines.append("")
        lines.append(f"Total: {len(results)} broken link(s) across {len(by_source)} file(s)")
        return "\n".join(lines)

    _reg("find_broken_links", "Find wikilinks pointing at notes that don't exist.", find_broken_links)

    def get_graph_neighbors(path: str, depth: int = 1, direction: str = "both"):
        """BFS traversal of the wikilink graph."""
        from some_vault_some_mcp.tools.links import get_graph_neighbors as _neighbors
        try:
            results = _neighbors(vault_path, path, depth, direction)
        except FileNotFoundError:
            return f"Note not found: {path}"
        if not results:
            return f"No neighbors found for: {path} (depth: {depth}, direction: {direction})"
        lines = [
            f"Graph neighbors of: {path}",
            f"Direction: {direction} | Max depth: {depth} | Found: {len(results)} note(s)\n",
        ]
        for r in results:
            arrow = "←" if r["direction"] == "inbound" else ("→" if r["direction"] == "outbound" else "↔")
            indent = "  " * r["depth"]
            lines.append(f"{indent}{arrow} {r['path']} (depth {r['depth']})")
        return "\n".join(lines)

    _reg("get_graph_neighbors", "Walk the link graph outward from a note. Depth 1-5, default 1.", get_graph_neighbors)

    # ── index ─────────────────────────────────────────────────────────────
    def vault_index_status():
        """Check search index health."""
        from some_vault_some_mcp.tools.index import vault_index_status as _status
        status = _status(vault_path, db_path, provider.dimensions)
        index_state = "READY"
        if gate is not None and not gate.is_ready:
            index_state = "FAILED: " + (gate.error or "unknown error") if gate.error else "INDEXING"
        return (
            f"Vault Index Status:\n"
            f"  Status: {index_state}\n"
            f"  Files indexed: {status.total_files}\n"
            f"  Total chunks: {status.total_chunks}\n"
            f"  Pending reindex: {status.pending_reindex}\n"
            f"  DB size: {status.db_size_mb} MB"
        )

    _reg("vault_index_status", "Check search index health and statistics.", vault_index_status)

    async def vault_reindex(path: str | None = None):
        """Trigger incremental reindex (single file or full vault)."""
        if gate is not None and not gate.is_ready:
            if gate.error:
                return (
                    f"vault_reindex is unavailable: initial indexing failed.\n"
                    f"Error: {gate.error}\n"
                    f"Restart the server to retry."
                )
            return "vault_reindex is unavailable: the initial index is already being built."
        from some_vault_some_mcp.tools.index import vault_reindex as _reindex
        result = _reindex(vault_path, db_path, provider, single_file=path)
        return (
            f"Reindex complete:\n"
            f"  Files indexed: {result.files_indexed}\n"
            f"  Chunks created: {result.chunks_created}\n"
            f"  Files removed: {result.files_removed}\n"
            f"  Duration: {result.duration_seconds}s"
        )

    _reg("vault_reindex", "Trigger incremental reindex for one note or the entire vault.", vault_reindex)

    # ── canvas ────────────────────────────────────────────────────────────

    def list_canvases(folder: str | None = None):
        """List all .canvas files in the vault."""
        from some_vault_some_mcp.tools.canvas import list_canvases as _list
        paths = _list(vault_path, folder)
        if not paths:
            scope = f" in folder: {folder}" if folder else ""
            return f"No canvases found{scope}."
        lines = [f"Found {len(paths)} canvas(es):\n"]
        for p in paths:
            lines.append(f"- {p}")
        return "\n".join(lines)

    _reg("list_canvases", "List all .canvas files in the vault.", list_canvases)

    def read_canvas(path: str):
        """Read the structure of a canvas file."""
        from some_vault_some_mcp.tools.canvas import read_canvas as _read
        canvas = _read(vault_path, path)
        if canvas is None:
            return f"Canvas not found: {path}"
        lines = [f"Canvas: {path}"]
        lines.append(f"Nodes: {len(canvas.nodes)} | Edges: {len(canvas.edges)}\n")
        if canvas.nodes:
            lines.append("Nodes:")
            for n in canvas.nodes:
                detail = n.text or n.file or n.url or n.label or ""
                if len(detail) > 80:
                    detail = detail[:77] + "..."
                lines.append(f"  [{n.id}] {n.type} ({n.x},{n.y} {n.width}x{n.height}) {detail}")
        if canvas.edges:
            lines.append("\nEdges:")
            for e in canvas.edges:
                lbl = f' "{e.label}"' if e.label else ""
                lines.append(f"  [{e.id}] {e.fromNode} -> {e.toNode}{lbl}")
        return "\n".join(lines)

    _reg("read_canvas", "Read canvas structure (nodes + edges).", read_canvas)

    async def create_canvas(path: str, nodes: str | None = None, edges: str | None = None):
        """Create a new canvas file. Fails if it already exists.

        nodes: JSON array of node objects, e.g. '[{"type":"text","text":"Hello"}]'
        edges: JSON array of edge objects.
        """
        from some_vault_some_mcp.tools.canvas import create_canvas as _create
        parsed_nodes = None
        parsed_edges = None
        if nodes:
            try:
                parsed_nodes = json.loads(nodes)
            except json.JSONDecodeError:
                return "Error: Invalid JSON in nodes parameter."
        if edges:
            try:
                parsed_edges = json.loads(edges)
            except json.JSONDecodeError:
                return "Error: Invalid JSON in edges parameter."
        try:
            resolved = await _create(vault_path, path, parsed_nodes, parsed_edges)
            return f"Created canvas at '{resolved}'."
        except FileExistsError as e:
            return f"Error: {e}"
        except (ValueError, FileNotFoundError) as e:
            return f"Error: {e}"

    _reg("create_canvas", "Create a new .canvas file with optional initial nodes/edges.", create_canvas)

    async def add_canvas_node(
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
    ):
        """Add a node to an existing canvas. Position auto-computed if omitted.

        For file nodes, the .md extension is optional on markdown targets (the
        full vault path is stored); attachments need their literal extension.
        """
        from some_vault_some_mcp.tools.canvas import add_canvas_node as _add
        try:
            result = await _add(
                vault_path, canvas_path, node_type,
                text=text, file=file, url=url, label=label,
                x=x, y=y, width=width, height=height, color=color,
            )
            return f"Added {node_type} node [{result['id']}] at ({result['x']}, {result['y']})."
        except (ValueError, FileNotFoundError) as e:
            return f"Error: {e}"

    _reg("add_canvas_node", "Add a node (text/file/link/group) to a canvas. Auto-layout when position omitted.", add_canvas_node)

    async def update_canvas_node(
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
    ):
        """Update properties of an existing canvas node."""
        from some_vault_some_mcp.tools.canvas import update_canvas_node as _update
        try:
            await _update(
                vault_path, canvas_path, node_id,
                x=x, y=y, width=width, height=height,
                color=color, text=text, file=file, url=url, label=label,
            )
            return f"Updated node [{node_id}]."
        except (ValueError, FileNotFoundError) as e:
            return f"Error: {e}"

    _reg("update_canvas_node", "Update any property of an existing canvas node by ID.", update_canvas_node)

    async def remove_canvas_nodes(canvas_path: str, node_ids: str):
        """Remove nodes from a canvas. Auto-cleans dangling edges.

        node_ids: JSON array of node ID strings, e.g. '["id1","id2"]'
        """
        from some_vault_some_mcp.tools.canvas import remove_canvas_nodes as _remove
        try:
            ids = json.loads(node_ids)
        except json.JSONDecodeError:
            return "Error: Invalid JSON in node_ids parameter."
        try:
            result = await _remove(vault_path, canvas_path, ids)
            lines = [f"Removed {len(result['removed_nodes'])} node(s)."]
            if result["removed_edges"]:
                lines.append(f"Cleaned up {len(result['removed_edges'])} dangling edge(s).")
            if result["not_found"]:
                lines.append(f"Not found: {', '.join(result['not_found'])}")
            return "\n".join(lines)
        except (ValueError, FileNotFoundError) as e:
            return f"Error: {e}"

    _reg("remove_canvas_nodes", "Remove nodes by ID. Auto-removes dangling edges.", remove_canvas_nodes)

    async def add_canvas_edge(
        canvas_path: str,
        from_node: str,
        to_node: str,
        from_side: str | None = None,
        to_side: str | None = None,
        from_end: str | None = None,
        to_end: str | None = None,
        color: str | None = None,
        label: str | None = None,
    ):
        """Add an edge between two nodes in a canvas."""
        from some_vault_some_mcp.tools.canvas import add_canvas_edge as _add
        try:
            edge_id = await _add(
                vault_path, canvas_path, from_node, to_node,
                from_side=from_side, to_side=to_side,
                from_end=from_end, to_end=to_end,
                color=color, label=label,
            )
            return f"Added edge [{edge_id}] from {from_node} to {to_node}."
        except (ValueError, FileNotFoundError) as e:
            return f"Error: {e}"

    _reg("add_canvas_edge", "Add an edge between two canvas nodes with full property support.", add_canvas_edge)

    async def update_canvas_edge(
        canvas_path: str,
        edge_id: str,
        from_side: str | None = None,
        to_side: str | None = None,
        from_end: str | None = None,
        to_end: str | None = None,
        color: str | None = None,
        label: str | None = None,
    ):
        """Update properties of an existing canvas edge."""
        from some_vault_some_mcp.tools.canvas import update_canvas_edge as _update
        try:
            await _update(
                vault_path, canvas_path, edge_id,
                from_side=from_side, to_side=to_side,
                from_end=from_end, to_end=to_end,
                color=color, label=label,
            )
            return f"Updated edge [{edge_id}]."
        except (ValueError, FileNotFoundError) as e:
            return f"Error: {e}"

    _reg("update_canvas_edge", "Update properties of an existing canvas edge by ID.", update_canvas_edge)

    async def remove_canvas_edges(canvas_path: str, edge_ids: str):
        """Remove edges from a canvas.

        edge_ids: JSON array of edge ID strings.
        """
        from some_vault_some_mcp.tools.canvas import remove_canvas_edges as _remove
        try:
            ids = json.loads(edge_ids)
        except json.JSONDecodeError:
            return "Error: Invalid JSON in edge_ids parameter."
        try:
            result = await _remove(vault_path, canvas_path, ids)
            lines = [f"Removed {len(result['removed'])} edge(s)."]
            if result["not_found"]:
                lines.append(f"Not found: {', '.join(result['not_found'])}")
            return "\n".join(lines)
        except (ValueError, FileNotFoundError) as e:
            return f"Error: {e}"

    _reg("remove_canvas_edges", "Remove edges from a canvas by ID.", remove_canvas_edges)

    # ── MCP resources ─────────────────────────────────────────────────────
    @mcp.resource("obsidian://note/{path}")
    def resource_note(path: str) -> str:
        """Read any note by vault-relative path."""
        from some_vault_some_mcp.tools.read import get_note as _get_note
        note = _get_note(vault_path, path)
        if note is None:
            return f"Note not found: {path}"
        return note.content

    @mcp.resource("obsidian://tags")
    def resource_tags() -> str:
        """Tag index with usage counts as JSON."""
        from some_vault_some_mcp.tools.tags import get_tags as _get_tags
        tags = _get_tags(vault_path)
        return json.dumps(tags)

    @mcp.resource("obsidian://daily")
    def resource_daily() -> str:
        """Today's daily note content."""
        from some_vault_some_mcp.tools.daily import get_daily_note as _get_daily
        result = _get_daily(vault_path)
        if result is None:
            return "No daily note for today."
        return result["content"]

    return mcp
