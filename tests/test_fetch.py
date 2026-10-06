"""Adaptive fetching returns every row even when windows hit the cap."""

from flowcast.collector.fetch import fetch_window


def _fake_indexer(events_per_second: dict[int, int]):
    calls = []

    def fn(pool, start, end, limit):
        calls.append((start, end))
        rows = [{"t": s, "i": i} for s in range(start, end) for i in range(events_per_second.get(s, 0))]
        return rows[:limit]  # the real indexer silently cuts off at `limit`

    return fn, calls


def test_small_window_single_call():
    fn, calls = _fake_indexer({s: 1 for s in range(100)})
    rows, truncated = fetch_window(fn, "P", 0, 100, limit=1000)
    assert len(rows) == 100 and not truncated and len(calls) == 1


def test_busy_window_splits_until_complete():
    fn, calls = _fake_indexer({s: 30 for s in range(100)})  # 3,000 rows, cap 1,000
    rows, truncated = fetch_window(fn, "P", 0, 100, limit=1000)
    assert len(rows) == 3000 and not truncated and len(calls) > 1


def test_single_second_over_cap_is_flagged():
    fn, _ = _fake_indexer({5: 50})
    rows, truncated = fetch_window(fn, "P", 0, 10, limit=20)
    assert truncated
