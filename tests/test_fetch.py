"""Fetching returns every row exactly once even when pages hit the cap."""

from flowcast.collector.fetch import fetch_window


def _fake_indexer(events_per_second: dict[int, int]):
    calls = []

    def fn(pool, start, end, limit):
        calls.append((start, end))
        rows = [{"timestamp": s * 1000 + i, "i": i} for s in range(start, end)
                for i in range(events_per_second.get(s, 0))]
        return sorted(rows, key=lambda r: -r["timestamp"])[:limit]  # newest first, silently capped

    return fn, calls


def test_small_window_single_call():
    fn, calls = _fake_indexer({s: 1 for s in range(100)})
    rows, truncated = fetch_window(fn, "P", 0, 100, limit=1000)
    assert len(rows) == 100 and not truncated and len(calls) == 1


def test_busy_window_pages_until_complete():
    fn, calls = _fake_indexer({s: 30 for s in range(100)})  # 3,000 rows, cap 1,000
    rows, truncated = fetch_window(fn, "P", 0, 100, limit=1000)
    assert len(rows) == 3000 and len({r["timestamp"] for r in rows}) == 3000  # all, no duplicates
    assert not truncated and len(calls) <= 5


def test_single_second_over_cap_is_flagged():
    fn, _ = _fake_indexer({5: 50, 3: 2, 8: 1})
    rows, truncated = fetch_window(fn, "P", 0, 10, limit=20)
    assert truncated
    assert sorted(r["timestamp"] // 1000 for r in rows) == [3, 3] + [5] * 20 + [8]


def test_rows_stamped_exactly_at_end_are_dropped():
    def fn(pool, start, end, limit):  # like the real indexer: end is inclusive at .000 ms
        return [{"timestamp": end * 1000}, {"timestamp": start * 1000}]
    rows, _ = fetch_window(fn, "P", 0, 10, limit=100)
    assert [r["timestamp"] for r in rows] == [0]
