"""Resumable backfill of DeepBook order updates and trades.

Unit of work = one (stream, pool, hour). Hours already on disk are skipped, so the
command can be interrupted and re-run any time, and ranges can be extended later.
The current, still-open hour is never fetched.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta

import pandas as pd
from tqdm import tqdm

from flowcast.collector import store
from flowcast.collector.fetch import fetch_window
from flowcast.collector.indexer_api import DeepBookIndexer
from flowcast.config import settings

_local = threading.local()


def _api() -> DeepBookIndexer:
    if not hasattr(_local, "api"):
        _local.api = DeepBookIndexer()
    return _local.api


def parse_day(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def hours(start: datetime, end: datetime) -> list[datetime]:
    """Hour starts in [start, end), stopping before the still-open hour."""
    cutoff = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    h = start.replace(minute=0, second=0, microsecond=0)
    out = []
    while h < end and h < cutoff:
        out.append(h)
        h += timedelta(hours=1)
    return out


def collect_one(stream: str, pool: str, hour: datetime) -> tuple[int, bool]:
    api = _api()
    fn = api.order_updates if stream == "orders" else api.trades
    s = int(hour.timestamp())
    rows, truncated = fetch_window(fn, pool, s, s + 3600)
    return store.write_hour(stream, pool, hour, rows, truncated), truncated


def run(pools: list[str], streams: list[str], start: datetime, end: datetime,
        workers: int | None = None, redo: bool = False) -> None:
    todo = [(st, p, h) for p in pools for st in streams for h in hours(start, end)
            if redo or not store.is_done(st, p, h)]
    total = len(pools) * len(streams) * len(hours(start, end))
    print(f"[collect] {len(todo)} of {total} (stream, pool, hour) units to fetch; "
          f"{total - len(todo)} already on disk")
    if not todo:
        return
    rows = truncs = fails = 0
    with ThreadPoolExecutor(max_workers=workers or settings.workers) as ex:
        futs = {ex.submit(collect_one, *u): u for u in todo}
        with tqdm(total=len(futs), desc="collect", unit="h") as bar:
            for f in as_completed(futs):
                st, p, h = futs[f]
                try:
                    n, t = f.result()
                    rows += n
                    truncs += t
                    if t:
                        tqdm.write(f"[collect] WARNING truncated: {st} {p} {h:%Y-%m-%d %H}:00 "
                                   "(a single second exceeded the row cap)")
                except Exception as e:  # noqa: BLE001 - keep going; the hour stays missing and is retried next run
                    fails += 1
                    tqdm.write(f"[collect] failed {st} {p} {h:%Y-%m-%d %H}:00: {e}")
                bar.update(1)
                bar.set_postfix(rows=rows, failed=fails)
    print(f"[collect] done: {rows:,} rows written, {truncs} truncated hours, {fails} failed "
          f"(failed hours are retried on the next run)")


def status() -> None:
    m = store.manifest()
    if m.empty:
        print("nothing collected yet")
        return
    m["hour"] = pd.to_datetime(m["hour"], utc=True)
    for (st, p), g in m.groupby(["stream", "pool"]):
        span_h = int((g["hour"].max() - g["hour"].min()).total_seconds() // 3600) + 1
        print(f"{st:7} {p:12} {len(g):5} h  {g['hour'].min():%Y-%m-%d %H} -> {g['hour'].max():%Y-%m-%d %H}  "
              f"gaps={span_h - len(g):4}  rows={int(g['rows'].sum()):>12,}  "
              f"rows/h={int(g['rows'].mean()):>7,}  truncated={int(g['truncated'].sum())}")
