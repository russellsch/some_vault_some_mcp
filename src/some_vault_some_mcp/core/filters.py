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
