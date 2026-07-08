"""Domain Pydantic models per §6.2.1."""

from pydantic import BaseModel, ConfigDict


class NoteContent(BaseModel):
    file_path: str
    title: str
    content: str
    frontmatter: dict = {}
    tags: list[str] = []


class NoteMetadata(BaseModel):
    file_path: str
    title: str
    tags: list[str] = []
    projects: list[str] = []
    status: str | None = None
    area: str | None = None
    created: str | None = None


class SearchResult(BaseModel):
    title: str
    file_path: str
    heading: str | None = None
    snippet: str
    score: float
    tags: list[str] = []
    projects: list[str] = []
    area: str | None = None


class TextSearchMatch(BaseModel):
    line: int
    content: str


class TextSearchResult(BaseModel):
    relative_path: str
    matches: list[TextSearchMatch]


class LinkInfo(BaseModel):
    target: str
    resolved_path: str | None = None
    is_valid: bool
    is_embed: bool = False


class BacklinkInfo(BaseModel):
    source: str
    line: int
    context: str


class OrphanInfo(BaseModel):
    fully_isolated: list[str]
    no_backlinks: list[str]
    no_outlinks: list[str]
    total_notes: int


class BrokenLinkEntry(BaseModel):
    source_path: str
    target_link: str
    line: int


class GraphNeighbor(BaseModel):
    path: str
    depth: int
    direction: str


class IndexStatus(BaseModel):
    total_chunks: int
    total_files: int
    pending_reindex: int = 0
    db_size_mb: float = 0.0


class ReindexResult(BaseModel):
    files_indexed: int
    chunks_created: int
    files_removed: int
    duration_seconds: float


# ── Canvas models ────────────────────────────────────────────────────────

class CanvasNode(BaseModel):
    # Preserve unknown/future JSON-Canvas fields (background, backgroundStyle,
    # subpath, …) so round-tripping a node never drops data the user didn't touch.
    model_config = ConfigDict(extra="allow")

    id: str
    type: str  # text, file, link, group
    x: int
    y: int
    width: int
    height: int
    color: str | None = None
    text: str | None = None
    file: str | None = None
    url: str | None = None
    label: str | None = None


class CanvasEdge(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    fromNode: str
    toNode: str
    fromSide: str | None = None
    toSide: str | None = None
    fromEnd: str | None = None
    toEnd: str | None = None
    color: str | None = None
    label: str | None = None


class CanvasData(BaseModel):
    nodes: list[CanvasNode] = []
    edges: list[CanvasEdge] = []
    extra: dict = {}
