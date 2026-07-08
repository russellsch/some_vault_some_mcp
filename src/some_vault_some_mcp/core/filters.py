"""Helpers for building safe LanceDB WHERE clause filter expressions.

LanceDB (0.30.x) uses SQL-like filter expressions with double-quoted string
literals. Two escaping contexts, which must NOT be conflated:

- Exact string comparison (`col = "..."`) — only `"` is special; escape it by
  doubling (SQL-standard). A backslash is NOT an escape char here and would be a
  literal character, so the old backslash-escaping silently matched zero rows.
- `LIKE` patterns (`col LIKE "%...%"`) — `%` and `_` are wildcards. LanceDB
  0.30.2 honours backslash as the escape character inside `LIKE` *without* an
  explicit `ESCAPE` clause (verified empirically), so backslash-escaping them
  matches the literal character.

Use `escape_string` for `=` sites and `escape_like` for `LIKE` sites.
"""


def escape_string(val: str) -> str:
    """Escape a value for use inside a double-quoted LanceDB string literal (`=`)."""
    return val.replace('"', '""')


def escape_like(val: str) -> str:
    """Escape a value for use inside a LIKE pattern.

    Doubles `"` to stay inside the string literal, then backslash-escapes the
    `%`/`_` wildcards so they match literally.
    """
    return escape_string(val).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def store_tokens(vals) -> str:
    """Serialize tag/project values for storage: lowercased, comma-delimited and
    sentinel-wrapped (``",a,b,"``) so ``LIKE "%,tok,%"`` matches whole tokens
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
    return f'{col} LIKE "%,{escape_like(val.strip().lower())},%"'
