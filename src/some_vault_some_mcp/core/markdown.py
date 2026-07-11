"""Shared markdown scanning helpers.

`iter_lines_with_fence_state` is the single fenced-code state machine reused by
the chunker, frontmatter tag extraction, and wikilink extraction (previously
duplicated in three places). It yields each line together with whether that line
is inside (or is a delimiter of) a fenced code block, so callers can skip code.
"""

import re
from collections.abc import Iterator


def iter_lines_with_fence_state(text: str) -> Iterator[tuple[str, bool]]:
    """Yield (line, in_code) for each line of text.

    in_code is True for the opening fence line, every line inside the fence, and
    the closing fence line. Supports ``` and ~~~ fences of length >= 3; a fence
    closes on a delimiter run of at least the opening length.
    """
    in_fence = False
    fence_char = ""
    fence_len = 0
    for line in text.split("\n"):
        stripped = line.lstrip()
        if in_fence:
            close_re = re.compile(rf"^{re.escape(fence_char)}{{{fence_len},}}\s*$")
            if close_re.match(stripped):
                in_fence = False
            yield line, True  # closing delimiter (or body) counts as code
            continue
        backtick_m = re.match(r"^(`{3,})", stripped)
        tilde_m = re.match(r"^(~{3,})", stripped)
        if backtick_m:
            in_fence = True
            fence_char = "`"
            fence_len = len(backtick_m.group(1))
            yield line, True
            continue
        if tilde_m:
            in_fence = True
            fence_char = "~"
            fence_len = len(tilde_m.group(1))
            yield line, True
            continue
        yield line, False
