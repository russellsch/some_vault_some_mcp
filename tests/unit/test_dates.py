"""Unit tests for moment.js-style date formatter.

Golden-file tests for all supported tokens.
"""

from datetime import datetime

import pytest

from some_vault_some_mcp.core.dates import format_moment_date


# Reference date: 2026-05-08 (Friday)
REF = datetime(2026, 5, 8, 9, 5, 7)


def fmt(f):
    return format_moment_date(REF, f)


def test_yyyy():
    assert fmt("YYYY") == "2026"


def test_yy():
    assert fmt("YY") == "26"


def test_mmmm():
    assert fmt("MMMM") == "May"


def test_mmm():
    assert fmt("MMM") == "May"


def test_mm():
    assert fmt("MM") == "05"


def test_m():
    assert fmt("M") == "5"


def test_dd():
    assert fmt("DD") == "08"


def test_do():
    # 8th → 8th
    assert fmt("Do") == "8th"


def test_d():
    assert fmt("D") == "8"


def test_ddd_day():
    # 2026-05-08 is a Friday
    assert fmt("dddd") == "Friday"


def test_ddd_short():
    assert fmt("ddd") == "Fri"


def test_dd_two():
    assert fmt("dd") == "Fr"


def test_hh_24():
    assert fmt("HH") == "09"


def test_h_24():
    assert fmt("H") == "9"


def test_hh_12():
    # 9 AM → 9
    assert fmt("hh") == "09"


def test_mm_minute():
    assert fmt("mm") == "05"


def test_ss():
    assert fmt("ss") == "07"


def test_q():
    # May = Q2
    assert fmt("Q") == "2"


def test_bracket_literal():
    assert fmt("[Q]") == "Q"
    assert fmt("[YYYY]") == "YYYY"


def test_combined_format():
    assert fmt("YYYY-MM-DD") == "2026-05-08"


def test_ddd_day_of_year():
    # 2026-05-08: Jan=31, Feb=28, Mar=31, Apr=30, May 1-8 = 8. Total = 31+28+31+30+8 = 128
    assert fmt("DDD") == "128"


def test_dddd_zero_padded_day_of_year():
    assert fmt("DDDD") == "128"


def test_ordinal_1st():
    d = datetime(2026, 5, 1)
    assert format_moment_date(d, "Do") == "1st"


def test_ordinal_2nd():
    d = datetime(2026, 5, 2)
    assert format_moment_date(d, "Do") == "2nd"


def test_ordinal_3rd():
    d = datetime(2026, 5, 3)
    assert format_moment_date(d, "Do") == "3rd"


def test_ordinal_11th():
    d = datetime(2026, 5, 11)
    assert format_moment_date(d, "Do") == "11th"


def test_ordinal_21st():
    d = datetime(2026, 5, 21)
    assert format_moment_date(d, "Do") == "21st"


def test_unknown_token_passthrough():
    # Unknown tokens pass through as-is
    result = fmt("YYYY-X-DD")
    assert "2026" in result
    assert "08" in result


def test_parse_date_str_valid():
    from some_vault_some_mcp.core.dates import parse_date_str
    assert parse_date_str("2026-07-08") == datetime(2026, 7, 8)


@pytest.mark.parametrize("bad", ["today", "2026/07/08", "2026-7", "garbage", "2026-13-01"])
def test_parse_date_str_invalid_raises_valueerror(bad):
    from some_vault_some_mcp.core.dates import parse_date_str
    with pytest.raises(ValueError) as e:
        parse_date_str(bad)
    assert "YYYY-MM-DD" in str(e.value)
