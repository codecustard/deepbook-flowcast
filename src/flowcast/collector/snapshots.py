"""Live order book recorder.

The indexer only serves the CURRENT book, so historical snapshots don't exist unless
someone records them. This polls /orderbook every few seconds and appends to parquet:

  data/snapshots/pool={POOL}/date=YYYY-MM-DD/part-{unix_ms}.parquet

Long format: one row per (poll, side, level). Used to validate the book the ML side
rebuilds from raw events, and as live input for the deployed predictor.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pandas as pd

from flowcast.collector.indexer_api import DeepBookIndexer
from flowcast.config import SNAPSHOTS


def _flush(buf: dict[str, list[dict]]) -> None:
    now = datetime.now(UTC)
    for pool, rows in buf.items():
        if not rows:
            continue
        d = SNAPSHOTS / f"pool={pool}" / f"date={now:%Y-%m-%d}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(d / f"part-{int(now.timestamp() * 1000)}.parquet", index=False)
        rows.clear()


def record(pools: list[str], every: float = 2.0, depth: int = 20, flush_every: float = 60.0) -> None:
    api = DeepBookIndexer(timeout=15)
    buf: dict[str, list[dict]] = {p: [] for p in pools}
    last_flush = time.monotonic()
    print(f"[snapshots] recording {', '.join(pools)} every {every}s, depth {depth}; Ctrl-C to stop")
    try:
        while True:
            t0 = time.monotonic()
            for pool in pools:
                try:
                    ob = api.orderbook(pool, depth)
                except Exception as e:  # noqa: BLE001 - a missed poll is fine; keep recording
                    print(f"[snapshots] {pool}: {e}")
                    continue
                recv_ms = int(time.time() * 1000)
                ts = int(ob.get("timestamp") or recv_ms)
                for side in ("bids", "asks"):
                    for lvl, (price, size) in enumerate(ob.get(side) or []):
                        buf[pool].append({"pool": pool, "timestamp": ts, "recv_ms": recv_ms,
                                          "side": side[:-1], "level": lvl,
                                          "price": float(price), "size": float(size)})
            if time.monotonic() - last_flush >= flush_every:
                _flush(buf)
                last_flush = time.monotonic()
            time.sleep(max(0.0, every - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        _flush(buf)
        print("[snapshots] stopped, buffer flushed")
