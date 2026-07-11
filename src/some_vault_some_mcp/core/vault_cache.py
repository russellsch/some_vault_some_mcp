"""mtime-validated cache of note contents.

Link/graph/tag tools re-read the entire vault on every call. This cache keeps
parsed file text keyed by (vault, path) and re-reads a file only when its mtime
changes — turning repeated calls from O(vault) reads into O(vault) stats (~10-50x
cheaper on warm calls). Staleness is bounded by an mtime check per file per call,
so results stay correct. Unbounded by design (bound ≈ vault size).
"""

from pathlib import Path

from some_vault_some_mcp.core.paths import walk_vault

_cache: dict[str, tuple[float, str]] = {}


def read_all(vault_path: str) -> tuple[list[str], dict[str, str]]:
    """Return (note_paths, {rel: content}) for the vault, using the mtime cache."""
    notes = walk_vault(vault_path)
    out: dict[str, str] = {}
    for rel in notes:
        key = f"{vault_path}\0{rel}"
        p = Path(vault_path) / rel
        try:
            mtime = p.stat().st_mtime
        except OSError:
            _cache.pop(key, None)
            continue
        hit = _cache.get(key)
        if hit is not None and hit[0] == mtime:
            out[rel] = hit[1]
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            _cache.pop(key, None)
            continue
        _cache[key] = (mtime, content)
        out[rel] = content
    return notes, out
