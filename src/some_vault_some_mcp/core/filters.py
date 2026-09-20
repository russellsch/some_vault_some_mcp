"""Helpers for building safe LanceDB WHERE clause filter expressions.

LanceDB parses filter strings with DataFusion SQL. DataFusion treats a
double-quoted token as a column identifier, so ``file_path = "a.md"`` raises
``Schema error: No field named "a.md"`` on LanceDB 0.39. LanceDB 0.30 accepted
that form only through a legacy fallback. A single-quoted token is the only
string-literal form, and it works on both versions.

Two escaping contexts, which must NOT be conflated:

- Exact string comparison (``col = '...'``) — only ``'`` is special; escape it
  by doubling (SQL-standard). A backslash is NOT an escape char here and is a
  literal character.
- ``LIKE`` patterns (``col LIKE '%...%'``) — ``%`` and ``_`` are wildcards.
  LanceDB honours backslash as the escape character inside ``LIKE`` *without*
  an explicit ``ESCAPE`` clause (verified on 0.30.2 and 0.39.0), so
  backslash-escaping them matches the literal character.

Use `escape_string` for `=` sites and `escape_like` for `LIKE` sites.
"""


def escape_string(val: str) -> str:
    """Escape a value for use inside a single-quoted LanceDB string literal (`=`)."""
    return val.replace("'", "''")


def escape_like(val: str) -> str:
    """Escape a value for use inside a single-quoted LIKE pattern.

    Doubles `'` to stay inside the string literal, then backslash-escapes the
    `%`/`_` wildcards so they match literally.
    """
    return escape_string(val).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def store_tokens(vals) -> str:
    """Serialize tag/project values for storage: lowercased, comma-delimited and
    sentinel-wrapped (``",a,b,"``) so ``LIKE '%,tok,%'`` matches whole tokens
    instead of substrings. Empty string when there are no values.
    """
    if isinstance(vals, list):
        items = [str(v).strip().lower() for v in vals if v is not None and str(v).strip()]
    elif vals:
        items = [str(vals).strip().lower()]
    else:
        items = []
    return "," + ",".join(items) + "," if items else ""


def split_tokens(stored: str) -> list[str]:
    """Inverse of :func:`store_tokens` — strip the sentinel commas and empties."""
    return [t for t in stored.strip(",").split(",") if t] if stored else []


def like_token(col: str, val: str) -> str:
    """Whole-token LIKE condition for a sentinel-wrapped column (tags/projects)."""
    return f"{col} LIKE '%,{escape_like(val.strip().lower())},%'"
