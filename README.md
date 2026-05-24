# some-vault-some-mcp

MCP server that gives AI models read/write access to Obsidian vaults with semantic search, link graph analysis, canvas manipulation, and incremental indexing.

Built on FastMCP + LanceDB. Embeds notes locally with fastembed (nomic-embed-text-v1.5-Q, 768 dims) by default — no server required. Optionally supports Ollama or OpenAI embeddings. Watches the vault filesystem and re-indexes on change.

Warning: This is a hot vibe coded mess, user beware

## What it does

- Hybrid search - vector similarity (70%) + full-text keyword (30%), with overlap boost
- Semantic search - pure embedding-based retrieval
- Exact search - literal substring matching, optional case sensitivity and regex
- Note CRUD - create, read, append, prepend, move, delete (soft delete to .trash by default)
- Frontmatter parsing - YAML extraction, tag collection (frontmatter + inline #hashtags), property filtering
- Wikilink graph - backlinks, outlinks, orphan detection, broken link detection, BFS neighbor traversal (depth 1-5)
- Canvas CRUD - create, read, add/update/remove nodes and edges, grid auto-layout, dangling edge cleanup
- Daily notes - Moment.js-style date formatting, template support, reads Obsidian's daily-notes config
- Incremental indexing - filesystem watcher with 2s debounce, mtime-based change detection
- Atomic writes - temp file + POSIX rename, per-path asyncio locks
- Tool overrides - rename or disable any tool via YAML config (per-agent customization)

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (for `uvx`)

## Quick start

Add this to your MCP client config (Cursor, Windsurf, Claude Code, etc.):

```json
{
  "mcpServers": {
    "obsidian-vault": {
      "command": "uvx",
      "args": ["some-vault-some-mcp", "serve", "--transport", "stdio"],
      "env": {
        "VAULT_PATH": "/path/to/your/obsidian/vault"
      }
    }
  }
}
```

That's it. `uvx` installs the package from PyPI on first run and keeps it cached.

For alternative embedding providers, add `--from` with extras:

```json
{
  "command": "uvx",
  "args": ["--from", "some-vault-some-mcp[ollama]", "some-vault-some-mcp", "serve", "--transport", "stdio"],
  "env": {
    "VAULT_PATH": "/path/to/your/obsidian/vault"
  }
}
```

Replace `[ollama]` with `[openai]` or `[gpu]` as needed.

### SSE mode

If running the server separately (Docker, remote, etc.), use the SSE URL instead:

```json
{
  "mcpServers": {
    "obsidian-vault": {
      "url": "http://localhost:3789/sse"
    }
  }
}
```

### Running directly

```bash
VAULT_PATH=/path/to/vault uvx some-vault-some-mcp serve
```

First run does a full index of the vault - takes a few minutes depending on size. Subsequent starts run incremental index only.

### CLI flags

```
some-vault-some-mcp serve [--transport sse|stdio] [--host 0.0.0.0] [--port 3789]
```

CLI flags override env vars.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `VAULT_PATH` | (required) | Absolute path to Obsidian vault |
| `LANCE_DB_PATH` | `./data/vault.lance` | Where the vector index lives |
| `MCP_TRANSPORT` | `sse` | `sse` or `stdio` |
| `MCP_HOST` | `0.0.0.0` | SSE bind address |
| `MCP_PORT` | `3789` | SSE port |
| `EMBEDDING_PROVIDER` | `fastembed` | `fastembed`, `ollama`, `openai`, or `mock` |
| `FASTEMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5-Q` | Any fastembed-supported model. On Apple Silicon, auto-detects and uses the non-quantized variant (`v1.5` instead of `v1.5-Q`) since the quantized ONNX ops are x86-optimized |
| `FASTEMBED_DIMENSIONS` | (auto-detected) | Override dimension auto-detection |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint (requires `--extra ollama`) |
| `OPENAI_API_KEY` | | Required if provider is `openai` (requires `--extra openai`) |
| `VAULT_API_KEY` | | Enables Bearer token auth on SSE transport |
| `VAULT_SOFT_DELETE_IS_PERMANENT` | `false` | `true` makes delete_note do hard deletes |
| `some_vault_some_mcp_OVERRIDES` | | Path to YAML override file |

## Docker

```bash
docker build -t some-vault-some-mcp .

docker run -p 3789:3789 \
  -v /path/to/vault:/opt/vault:ro \
  -v /data:/opt/data \
  some-vault-some-mcp
```

Runs as non-root user (vault:vault, UID 1000). Index data persists in `/opt/data`. Vault mounted read-only. Container is self-contained — fastembed runs in-process, no Ollama sidecar needed.

To use Ollama instead: `docker run ... -e EMBEDDING_PROVIDER=ollama -e OLLAMA_URL=http://ollama:11434 some-vault-some-mcp`

## Tools (27)

### Search

#### `search`

Find notes by text, meaning, or exact string.

| Param | Type | Default | Notes |
|---|---|---|---|
| `query` | string | (required) | Search query text |
| `mode` | string | `"hybrid"` | `hybrid`, `semantic`, or `exact` |
| `top_k` | int | `10` | Max results to return |
| `tags` | string[] | `null` | Pre-filter by tags |
| `folder` | string | `null` | Pre-filter by folder path prefix |
| `case_sensitive` | bool | `false` | Exact mode only - match case |

### Read

#### `get_note`

Read a single note with parsed frontmatter and tags.

| Param | Type | Default | Notes |
|---|---|---|---|
| `path` | string | (required) | Vault-relative path to the note. Extension-agnostic — `todo` and `todo.md` both resolve to `todo.md`. A bare basename resolves Obsidian-style to a matching note anywhere in the vault. |

#### `list_notes`

Enumerate vault notes with optional metadata filters. Index-backed fields (tags, projects, status, area) query LanceDB when available. Frontmatter property filtering scans files directly.

| Param | Type | Default | Notes |
|---|---|---|---|
| `folder` | string | `null` | Filter by folder path prefix |
| `tags` | string[] | `null` | Filter by tags (index-backed) |
| `projects` | string[] | `null` | Filter by projects (index-backed) |
| `status` | string | `null` | Filter by status field (index-backed) |
| `area` | string | `null` | Filter by area field (index-backed) |
| `frontmatter_property` | string | `null` | Arbitrary frontmatter key to filter on |
| `frontmatter_value` | string | `null` | Value to match for `frontmatter_property` (case-insensitive) |
| `include_content` | bool | `false` | Include note content in results |
| `limit` | int | `50` | Max results to return |

### Write

#### `create_note`

Create a new note. Fails if it already exists.

| Param | Type | Default | Notes |
|---|---|---|---|
| `path` | string | (required) | Vault-relative path for the new note |
| `content` | string | (required) | Note body text |
| `frontmatter` | string | `null` | JSON string of frontmatter fields, e.g. `'{"title":"My Note","tags":["idea"]}'` |

#### `append_to_note`

Append text to the end of an existing note.

| Param | Type | Default | Notes |
|---|---|---|---|
| `path` | string | (required) | Vault-relative path |
| `content` | string | (required) | Text to append |

#### `prepend_to_note`

Insert text after frontmatter, before the note body.

| Param | Type | Default | Notes |
|---|---|---|---|
| `path` | string | (required) | Vault-relative path |
| `content` | string | (required) | Text to prepend |

#### `update_frontmatter`

Merge key-value pairs into YAML frontmatter. Unlisted keys preserved.

| Param | Type | Default | Notes |
|---|---|---|---|
| `path` | string | (required) | Vault-relative path |
| `properties` | string | (required) | JSON string of key-value pairs, e.g. `'{"status":"done","tags":["review"]}'` |

#### `move_note`

Move or rename a note. Rewrites wikilinks vault-wide by default.

| Param | Type | Default | Notes |
|---|---|---|---|
| `old_path` | string | (required) | Current vault-relative path |
| `new_path` | string | (required) | Destination vault-relative path |
| `update_links` | bool | `true` | Rewrite wikilinks in all referencing notes |

#### `delete_note`

Delete a note. Moves to .trash by default; use permanent for hard delete.

| Param | Type | Default | Notes |
|---|---|---|---|
| `path` | string | (required) | Vault-relative path |
| `permanent` | bool | `false` | `true` for hard delete, `false` for soft delete to .trash |

### Daily notes

#### `get_daily_note`

Read today's or a specific date's daily note. Uses Obsidian's daily-notes config for folder and date format. Returns a text message (not an error) if no note exists for the requested date.

| Param | Type | Default | Notes |
|---|---|---|---|
| `date` | string | `null` | Date string (e.g. `"2024-03-15"`). Omit for today |

#### `create_daily_note`

Create a daily note for today or a given date. Fails if one exists. Supports `{{date}}` placeholder in templates.

| Param | Type | Default | Notes |
|---|---|---|---|
| `date` | string | `null` | Date string. Omit for today |
| `content` | string | `null` | Note body text |
| `template_path` | string | `null` | Vault-relative path to a template note |

### Tags

#### `get_tags`

List all unique tags in the vault with per-note usage counts.

| Param | Type | Default | Notes |
|---|---|---|---|
| `sort_by` | string | `"count"` | `count` (descending) or `name` (alphabetical) |

### Link graph

#### `get_backlinks`

Find notes that link to a given note.

| Param | Type | Default | Notes |
|---|---|---|---|
| `path` | string | (required) | Vault-relative path of the target note |

#### `get_outlinks`

List all outgoing wikilinks from a note. Shows valid, broken, and embed links separately.

| Param | Type | Default | Notes |
|---|---|---|---|
| `path` | string | (required) | Vault-relative path |

#### `find_orphans`

Find disconnected notes - no inbound links, no outbound links, or both.

| Param | Type | Default | Notes |
|---|---|---|---|
| `include_outlinks_check` | bool | `true` | Also report notes with no outgoing links |
| `max_results` | int | `200` | Cap per category |

#### `find_broken_links`

Find wikilinks pointing at notes that don't exist.

| Param | Type | Default | Notes |
|---|---|---|---|
| `folder` | string | `null` | Limit scan to a folder |
| `max_results` | int | `200` | Max broken links to return |

#### `get_graph_neighbors`

Walk the link graph outward from a note via BFS.

| Param | Type | Default | Notes |
|---|---|---|---|
| `path` | string | (required) | Starting note path |
| `depth` | int | `1` | BFS depth, 1-5 |
| `direction` | string | `"both"` | `inbound`, `outbound`, or `both` |

### Canvas

#### `list_canvases`

List all .canvas files in the vault.

| Param | Type | Default | Notes |
|---|---|---|---|
| `folder` | string | `null` | Filter by folder path prefix |

#### `read_canvas`

Read canvas structure (nodes + edges).

| Param | Type | Default | Notes |
|---|---|---|---|
| `path` | string | (required) | Vault-relative path to .canvas file |

#### `create_canvas`

Create a new .canvas file with optional initial nodes/edges. Fails if it already exists.

| Param | Type | Default | Notes |
|---|---|---|---|
| `path` | string | (required) | Vault-relative path for the new canvas |
| `nodes` | string | `null` | JSON array of node objects, e.g. `'[{"type":"text","text":"Hello"}]'` |
| `edges` | string | `null` | JSON array of edge objects |

#### `add_canvas_node`

Add a node (text/file/link/group) to a canvas. Position auto-computed via grid layout if omitted.

| Param | Type | Default | Notes |
|---|---|---|---|
| `canvas_path` | string | (required) | Path to the canvas |
| `node_type` | string | (required) | `text`, `file`, `link`, or `group` |
| `text` | string | `null` | Text content (for text nodes) |
| `file` | string | `null` | Vault-relative file path (for file nodes) |
| `url` | string | `null` | URL (for link nodes) |
| `label` | string | `null` | Display label |
| `x` | int | `null` | X position. Auto-placed if omitted |
| `y` | int | `null` | Y position. Auto-placed if omitted |
| `width` | int | `250` | Node width in pixels |
| `height` | int | `60` | Node height in pixels |
| `color` | string | `null` | Obsidian preset `"1"`-`"6"` (red, orange, yellow, green, cyan, purple) or hex `"#FF0000"` |

#### `update_canvas_node`

Update any property of an existing canvas node by ID. Only provided fields are changed.

| Param | Type | Default | Notes |
|---|---|---|---|
| `canvas_path` | string | (required) | Path to the canvas |
| `node_id` | string | (required) | ID of the node to update |
| `x` | int | `null` | New X position |
| `y` | int | `null` | New Y position |
| `width` | int | `null` | New width |
| `height` | int | `null` | New height |
| `color` | string | `null` | Preset `"1"`-`"6"` or hex `"#FF0000"` |
| `text` | string | `null` | New text content |
| `file` | string | `null` | New file reference |
| `url` | string | `null` | New URL |
| `label` | string | `null` | New label |

#### `remove_canvas_nodes`

Remove nodes by ID. Auto-removes dangling edges.

| Param | Type | Default | Notes |
|---|---|---|---|
| `canvas_path` | string | (required) | Path to the canvas |
| `node_ids` | string | (required) | JSON array of node ID strings, e.g. `'["id1","id2"]'` |

#### `add_canvas_edge`

Add an edge between two canvas nodes with full property support.

| Param | Type | Default | Notes |
|---|---|---|---|
| `canvas_path` | string | (required) | Path to the canvas |
| `from_node` | string | (required) | Source node ID |
| `to_node` | string | (required) | Target node ID |
| `from_side` | string | `null` | Anchor side on source: `top`, `bottom`, `left`, `right` |
| `to_side` | string | `null` | Anchor side on target |
| `from_end` | string | `null` | End style on source side (e.g. `arrow`, `none`) |
| `to_end` | string | `null` | End style on target side |
| `color` | string | `null` | Preset `"1"`-`"6"` or hex `"#FF0000"` |
| `label` | string | `null` | Edge label text |

#### `update_canvas_edge`

Update properties of an existing canvas edge by ID. Only provided fields are changed.

| Param | Type | Default | Notes |
|---|---|---|---|
| `canvas_path` | string | (required) | Path to the canvas |
| `edge_id` | string | (required) | ID of the edge to update |
| `from_side` | string | `null` | New source anchor side |
| `to_side` | string | `null` | New target anchor side |
| `from_end` | string | `null` | New source end style |
| `to_end` | string | `null` | New target end style |
| `color` | string | `null` | Preset `"1"`-`"6"` or hex `"#FF0000"` |
| `label` | string | `null` | New label |

#### `remove_canvas_edges`

Remove edges from a canvas by ID.

| Param | Type | Default | Notes |
|---|---|---|---|
| `canvas_path` | string | (required) | Path to the canvas |
| `edge_ids` | string | (required) | JSON array of edge ID strings |

### Index management

#### `vault_index_status`

Check search index health and statistics. No arguments.

#### `vault_reindex`

Trigger incremental reindex for one note or the entire vault.

| Param | Type | Default | Notes |
|---|---|---|---|
| `path` | string | `null` | Vault-relative path to reindex a single note. Omit for full vault |

## MCP resources

- `obsidian://note/{path}` - read a note by path
- `obsidian://tags` - all tags
- `obsidian://daily` - today's daily note

## Tool name overrides

When you need to change the tool names and descriotions, a YAML file can be used. Create a YAML file, and set `some_vault_some_mcp_OVERRIDES` to its path:

```yaml
tools:
  search:
    name: "vault_search"
    description: "Custom description for this agent"
  get_note:
    name: "read_note"
disabled:
  - get_tags
  - find_orphans
```

Overrides apply at registration time. Useful for per-agent tool namespacing or disabling tools an agent shouldn't use if your agent doesn't support tool disabling.

## How search works

**Hybrid** (default) - runs both semantic and FTS (LanceDB full-text search, BM25-style) at 2x requested top_k, normalizes scores, combines with `semantic * 0.7 + FTS * 0.3`. Results appearing in both get a 1.2x boost. Returns top_k from merged set.

**Semantic** - embeds the query, runs vector similarity search against LanceDB.

**Exact** - scans raw vault files for literal substring matches. Not index-backed. Optional case sensitivity.

All search modes accept tag and folder pre-filters. Tags use SQL LIKE against comma-separated tag strings in the index. Folders use path prefix matching.

## Chunking

Splits markdown by heading hierarchy first, then by paragraph for oversized sections. Tracks heading breadcrumbs through the split - a chunk under `# A / ## B / ### C` gets heading field `"A > B > C"`. Metadata header prepended to embedding text: `[Title: X | Section: A > B | Tags: foo, bar]`.

## Wikilink resolution

Matches Obsidian's behavior in 4 steps:
1. Exact relative-path match (case-insensitive)
2. Path-suffix match (if link contains `/`)
3. Basename match with proximity tie-break (deepest shared path prefix with source)
4. Alias match (frontmatter aliases)

First matching step wins.

## Security boundaries

- All paths validated against vault root - rejects `../`, null bytes, symlink escapes
- `.obsidian`, `.git`, `.trash` excluded from indexing and tool access
- Optional Bearer token auth on SSE transport
- Docker container runs as non-root
- Bounded frontmatter parsing prevents YAML bombs

## Developing

### Setup

```bash
git clone https://github.com/russellsch/some_obsidian_some_mcp.git
cd some_obsidian_some_mcp
uv sync                           # default (fastembed CPU)
uv sync --extra gpu               # GPU-accelerated embeddings
uv sync --extra ollama            # Ollama provider
uv sync --extra openai            # OpenAI provider
```

### Running from source

```bash
VAULT_PATH=/path/to/vault uv run some-vault-some-mcp serve
```

To use a local checkout in your MCP client config instead of the PyPI package:

```json
{
  "mcpServers": {
    "obsidian-vault": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/some_obsidian_some_mcp", "some-vault-some-mcp", "serve", "--transport", "stdio"],
      "env": {
        "VAULT_PATH": "/path/to/your/obsidian/vault"
      }
    }
  }
}
```

### Tests

```bash
uv run pytest tests/unit           # unit tests (fast, no external deps)
uv run pytest tests/integration    # integration tests (real LanceDB, may download ~130MB model)
uv run pytest                      # everything
```

Test fixtures live in `tests/fixtures/vault/`.

### Releasing

See [RELEASING.md](RELEASING.md).
