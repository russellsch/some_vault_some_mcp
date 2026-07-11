"""Moment.js-style date formatter for Obsidian daily-note filename patterns.

Port of obsidian-mcp-pro/build/lib/dates.js.

Supported tokens (longest-first match):
  YYYY   four-digit year          2026
  YY     two-digit year           26
  MMMM   month name               January
  MMM    short month name         Jan
  MM     zero-padded month        01
  Mo     ordinal month            1st
  M      month number             1
  DDDD   zero-padded day-of-year  045
  DDD    day of year              45
  DD     zero-padded date         07
  Do     ordinal date             7th
  D      date                     7
  dddd   weekday name             Monday
  ddd    short weekday name       Mon
  dd     two-letter weekday       Mo
  HH/H   zero-padded/hour 24h     09 / 9
  hh/h   zero-padded/hour 12h     09 / 9
  mm/m   zero-padded/minute       05 / 5
  ss/s   zero-padded/second       05 / 5
  Q      quarter (1-4)            1
  [..]   literal text             [Q] → Q

Uses LOCAL time (matching Obsidian's rendering).
"""

from datetime import date, datetime


_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_WEEKDAYS = [
    "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
]


def _pad2(n: int) -> str:
    return str(n).zfill(2)


def _pad3(n: int) -> str:
    return str(n).zfill(3)


def _ordinal(n: int) -> str:
    mod100 = n % 100
    if 11 <= mod100 <= 13:
        return f"{n}th"
    mod10 = n % 10
    if mod10 == 1:
        return f"{n}st"
    if mod10 == 2:
        return f"{n}nd"
    if mod10 == 3:
        return f"{n}rd"
    return f"{n}th"


def _day_of_year(d: datetime) -> int:
    jan1 = datetime(d.year, 1, 1)
    return (d - jan1).days + 1


def format_moment_date(dt: datetime, fmt: str) -> str:
    """Format a datetime using a moment.js-style format string.

    Uses LOCAL time (dt should be a naive local datetime or have tzinfo stripped).
    """
    Y = dt.year
    Mo_ = dt.month
    D_ = dt.day
    wd = dt.weekday()  # Python: 0=Monday; moment.js: 0=Sunday
    # Convert to JS weekday (0=Sunday)
    js_wd = (wd + 1) % 7
    H = dt.hour
    m = dt.minute
    s = dt.second
    Q = (Mo_ - 1) // 3 + 1
    doy = _day_of_year(dt)

    tokens = {
        "YYYY": str(Y),
        "YY": str(Y)[-2:],
        "MMMM": _MONTHS[Mo_ - 1],
        "MMM": _MONTHS[Mo_ - 1][:3],
        "MM": _pad2(Mo_),
        "Mo": _ordinal(Mo_),
        "M": str(Mo_),
        "DDDD": _pad3(doy),
        "DDD": str(doy),
        "DD": _pad2(D_),
        "Do": _ordinal(D_),
        "D": str(D_),
        "dddd": _WEEKDAYS[js_wd],
        "ddd": _WEEKDAYS[js_wd][:3],
        "dd": _WEEKDAYS[js_wd][:2],
        "HH": _pad2(H),
        "H": str(H),
        "hh": _pad2(((H + 11) % 12) + 1),
        "h": str(((H + 11) % 12) + 1),
        "mm": _pad2(m),
        "m": str(m),
        "ss": _pad2(s),
        "s": str(s),
        "Q": str(Q),
    }
    # Sort tokens by descending length (longest-first wins)
    sorted_tokens = sorted(tokens.items(), key=lambda kv: -len(kv[0]))

    out: list[str] = []
    i = 0
    while i < len(fmt):
        ch = fmt[i]
        # Bracketed literal
        if ch == "[":
            end = fmt.find("]", i + 1)
            if end == -1:
                out.append(fmt[i:])
                break
            out.append(fmt[i + 1:end])
            i = end + 1
            continue
        # Try longest-first token match
        rest = fmt[i:]
        matched = False
        for token, value in sorted_tokens:
            if rest.startswith(token):
                out.append(value)
                i += len(token)
                matched = True
                break
        if not matched:
            out.append(ch)
            i += 1

    return "".join(out)


def parse_date_str(date_str: str) -> datetime:
    """Parse a YYYY-MM-DD string to a local-midnight datetime.

    Raises ValueError with a clear message on malformed input (previously raised
    a bare IndexError/ValueError that surfaced as an unhandled 500).
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date '{date_str}': expected YYYY-MM-DD")
