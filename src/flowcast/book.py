"""Order book reconstruction. ML side: reads data/raw, writes data/processed.

Replays orders and fills in one timeline:

  Placed              add the order: total = remaining_quantity
  Modified            new total = remaining_quantity (it INCLUDES what already filled);
                      filled_quantity = cumulative fills so far
  fill (trades)       add base_volume to the maker order's filled amount
  Canceled, Expired   remove the order (remaining_quantity = the open size cancelled)

Open size = total - filled, with filled = max(fills seen in trades, latest filled_quantity
reported by Modified): the two overlap when fills and a Modified share a checkpoint.

Sui stamps every event in a checkpoint (~0.25s of transactions) with the same millisecond,
and order events carry no transaction index, so order within a millisecond is unknown. We
apply Placed, Modified, fills, then Canceled/Expired; the book is exact at checkpoint
boundaries, which is finer than the 1-second snapshots.

The indexer keeps no historical snapshots, so the replay starts WARMUP_H hours before the
requested start with an empty book; orders placed before that are invisible until they
next change. The book state then carries across days.

Output, one file per day, one row per second (state after all events before the second ends):
  data/processed/book/pool={POOL}/date=YYYY-MM-DD.parquet
  ts (ms, end of second), bid_px_0..N-1, bid_sz_0..N-1, ask_px_0.., ask_sz_0..

Checks printed per day (the rebuild is only trusted if these hold):
  trade_throughs  fills that skipped a better-priced order which was resting before the
                  checkpoint and untouched during it (a phantom order in our book); want 0
  fills_unknown   share of fills on orders we never saw (placed before the warm-up)
  cancels_match   share of cancels whose tracked size equals the indexer's remaining_quantity
  crossed_s       share of seconds with best bid >= best ask
"""

from __future__ import annotations

import bisect
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from flowcast.config import PROCESSED, RAW

DEPTH = 10
WARMUP_H = 6
EPS = 1e-6
KIND = {"Placed": 0, "Modified": 1, "fill": 2, "Canceled": 3, "Expired": 3}


class Book:
    def __init__(self):
        self.orders: dict[str, tuple[bool, float, float]] = {}  # id -> (is_bid, price, size)
        self.levels = {True: {}, False: {}}  # is_bid -> {price: size}
        self.prices = {True: [], False: []}  # is_bid -> sorted prices with size > 0
        self.qf: dict[str, list[float]] = {}  # id -> [total, filled per trades, filled per Modified]

    def _add(self, is_bid: bool, price: float, qty: float) -> None:
        lv = self.levels[is_bid]
        new = lv.get(price, 0.0) + qty
        if new > EPS:
            if price not in lv:
                bisect.insort(self.prices[is_bid], price)
            lv[price] = new
        elif price in lv:
            del lv[price]
            px = self.prices[is_bid]
            del px[bisect.bisect_left(px, price)]

    def set_size(self, oid: str, is_bid: bool, price: float, size: float) -> None:
        old = self.orders.get(oid)
        if old:
            self._add(old[0], old[1], -old[2])
        if size > EPS:
            self.orders[oid] = (is_bid, price, size)
            self._add(is_bid, price, size)
        else:
            self.orders.pop(oid, None)

    def refresh(self, oid: str, is_bid: bool, price: float) -> None:
        """Re-derive an order's open size from its total and filled amounts."""
        q = self.qf[oid]
        size = q[0] - max(q[1], q[2])
        self.set_size(oid, is_bid, price, size)
        if size <= EPS:
            del self.qf[oid]

    def best(self, is_bid: bool) -> float | None:
        px = self.prices[is_bid]
        return (px[-1] if is_bid else px[0]) if px else None

    def top(self, depth: int = DEPTH) -> list[float]:
        bids, asks = self.prices[True][::-1][:depth], self.prices[False][:depth]
        pad = lambda xs: xs + [np.nan] * (depth - len(xs))  # noqa: E731
        return (pad(bids) + pad([self.levels[True][p] for p in bids])
                + pad(asks) + pad([self.levels[False][p] for p in asks]))


def load_events(pool: str, day: date) -> pd.DataFrame:
    """One day of orders + fills as one timeline: ts, kind, oid, is_bid, price, qty, filled."""
    def read(stream):
        files = sorted((RAW / stream / f"pool={pool}" / f"date={day:%Y-%m-%d}").glob("hour=*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()

    o, t = read("orders"), read("trades")
    parts = []
    if len(o):
        parts.append(pd.DataFrame({"ts": o.timestamp, "kind": o.status.map(KIND), "oid": o.order_id,
                                   "is_bid": o["type"].eq("buy"), "price": o.price,
                                   "qty": o.remaining_quantity, "filled": o.filled_quantity}))
    if len(t):  # the maker sits on the side opposite the taker
        parts.append(pd.DataFrame({"ts": t.timestamp, "kind": KIND["fill"], "oid": t.maker_order_id,
                                   "is_bid": ~t.taker_is_bid.astype(bool), "price": t.price,
                                   "qty": t.base_volume, "filled": 0.0}))
    if not parts:
        return pd.DataFrame(columns=["ts", "kind", "oid", "is_bid", "price", "qty", "filled"])
    ev = pd.concat(parts, ignore_index=True)
    return ev.sort_values(["ts", "kind"], kind="stable").reset_index(drop=True)


def replay(book: Book, ev: pd.DataFrame, t0: int, t1: int, depth: int = DEPTH):
    """Apply events in [t0, t1) ms; return per-second snapshots and check counters."""
    rows, stats = [], dict(fills=0, trade_throughs=0, fills_unknown=0, cancels=0, cancels_match=0)
    next_s = t0 + 1000
    cur_ts, fills, touched = None, [], {True: set(), False: set()}

    def close_checkpoint():
        # a better level nobody touched this checkpoint was resting all along: a fill behind it is wrong
        for is_bid, price in fills:
            px = book.prices[is_bid]
            better = px[bisect.bisect_right(px, price):] if is_bid else px[:bisect.bisect_left(px, price)]
            stats["trade_throughs"] += any(p not in touched[is_bid] for p in better)
        fills.clear()
        touched[True].clear()
        touched[False].clear()

    for ts, kind, oid, is_bid, price, qty, filled in zip(
            ev.ts.values, ev.kind.values, ev.oid.values, ev.is_bid.values, ev.price.values,
            ev.qty.values, ev.filled.values, strict=True):
        if ts < t0 or ts >= t1:
            continue
        if ts != cur_ts:
            close_checkpoint()
            cur_ts = ts
        while ts >= next_s:
            rows.append([next_s, *book.top(depth)])
            next_s += 1000
        is_bid = bool(is_bid)
        o = book.orders.get(oid)
        if o:
            touched[o[0]].add(o[1])
        touched[is_bid].add(price)
        if kind == 0:  # Placed
            book.qf[oid] = [qty, 0.0, filled]
            book.refresh(oid, is_bid, price)
        elif kind == 1:  # Modified: new total (incl. filled) + cumulative filled
            q = book.qf.setdefault(oid, [qty, 0.0, 0.0])
            q[0], q[2] = qty, max(q[2], filled)
            book.refresh(oid, is_bid, price)
        elif kind == 2:  # fill
            stats["fills"] += 1
            fills.append((is_bid, price))
            if o is None:
                stats["fills_unknown"] += 1
            else:
                book.qf[oid][1] += qty
                book.refresh(oid, o[0], o[1])
        else:  # Canceled / Expired
            if o is not None:
                stats["cancels"] += 1
                stats["cancels_match"] += abs(o[2] - qty) <= EPS * max(1.0, qty)
            book.qf.pop(oid, None)
            book.set_size(oid, is_bid, price, 0.0)
    close_checkpoint()
    while next_s <= t1:
        rows.append([next_s, *book.top(depth)])
        next_s += 1000
    cols = (["ts"] + [f"bid_px_{i}" for i in range(depth)] + [f"bid_sz_{i}" for i in range(depth)]
            + [f"ask_px_{i}" for i in range(depth)] + [f"ask_sz_{i}" for i in range(depth)])
    return pd.DataFrame(rows, columns=cols).astype({"ts": "int64"}), stats


def _ms(d: datetime) -> int:
    return int(d.timestamp() * 1000)


def rebuild(pool: str, start: str, end: str, depth: int = DEPTH) -> list[dict]:
    """Rebuild days in [start, end) and write one parquet per day. Returns per-day checks."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    t_start = datetime(first.year, first.month, first.day, tzinfo=UTC)
    out_dir = PROCESSED / "book" / f"pool={pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    book = Book()

    warm_from = t_start - timedelta(hours=WARMUP_H)
    replay(book, load_events(pool, warm_from.date()), _ms(warm_from), _ms(t_start), depth)

    report = []
    for i in range((last - first).days):
        d0 = t_start + timedelta(days=i)
        snaps, s = replay(book, load_events(pool, d0.date()), _ms(d0), _ms(d0 + timedelta(days=1)), depth)
        snaps.to_parquet(out_dir / f"date={d0:%Y-%m-%d}.parquet", index=False)
        crossed = (snaps.bid_px_0 >= snaps.ask_px_0).mean()
        r = {"date": f"{d0:%Y-%m-%d}", "fills": s["fills"], "trade_throughs": s["trade_throughs"],
             "fills_unknown": s["fills_unknown"] / max(s["fills"], 1),
             "cancels_match": s["cancels_match"] / max(s["cancels"], 1),
             "crossed_s": float(crossed), "median_spread": float((snaps.ask_px_0 - snaps.bid_px_0).median())}
        report.append(r)
        print(f"[book] {pool} {r['date']}  fills={r['fills']:6}  trade_throughs={r['trade_throughs']}  "
              f"unknown={r['fills_unknown']:.4f}  cancels_match={r['cancels_match']:.4f}  "
              f"crossed_s={r['crossed_s']:.4f}  spread={r['median_spread']:.5f}")
    return report
