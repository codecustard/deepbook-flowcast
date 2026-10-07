"""Complete fetching under the indexer's 10,000-row cap.

The indexer returns the NEWEST rows in a window, capped at the limit. When a page comes
back full, keep every second it covers completely and ask again for the older remainder,
up to and including the page's oldest (possibly partial) second. Every row is fetched
about once, so a busy hour costs ~rows/limit requests. A single second with more rows
than the cap can't be split further: its newest `limit` rows are kept and the window is
flagged truncated.
"""

from __future__ import annotations

from collections.abc import Callable

from flowcast.collector.indexer_api import LIMIT

FetchFn = Callable[[str, int, int, int], list[dict]]


def fetch_window(fn: FetchFn, pool: str, start: int, end: int, limit: int = LIMIT) -> tuple[list[dict], bool]:
    """All rows in [start, end) seconds. Returns (rows, truncated)."""
    lo, hi = start * 1000, end * 1000  # the indexer also returns rows stamped exactly at `end`
    rows: list[dict] = []
    truncated = False
    while end > start:
        page = fn(pool, start, end, limit)
        if len(page) < limit:
            rows += page
            break
        oldest = min(r["timestamp"] for r in page) // 1000  # ms -> s
        complete = [r for r in page if r["timestamp"] // 1000 > oldest]
        if complete:
            rows += complete
            end = oldest + 1  # re-fetch the oldest second whole
        else:  # the whole page is one second over the cap
            rows += page
            truncated = True
            end = oldest
    return [r for r in rows if lo <= r["timestamp"] < hi], truncated
