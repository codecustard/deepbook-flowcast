"""Complete fetching under the indexer's 10,000-row cap.

Ask for the whole window; if the answer comes back full, split the window in half and
ask again, recursively. Busy stretches get small windows, quiet ones stay large, and
nothing is silently cut off. A 1-second window that is still full is flagged truncated.
"""

from __future__ import annotations

from collections.abc import Callable

from flowcast.collector.indexer_api import LIMIT

FetchFn = Callable[[str, int, int, int], list[dict]]


def fetch_window(fn: FetchFn, pool: str, start: int, end: int, limit: int = LIMIT) -> tuple[list[dict], bool]:
    """All rows in [start, end) seconds. Returns (rows, truncated)."""
    rows = fn(pool, start, end, limit)
    if len(rows) < limit:
        return rows, False
    if end - start <= 1:
        return rows, True
    mid = (start + end) // 2
    left, t1 = fetch_window(fn, pool, start, mid, limit)
    right, t2 = fetch_window(fn, pool, mid, end, limit)
    return left + right, t1 or t2
