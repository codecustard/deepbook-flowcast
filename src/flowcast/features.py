"""Features: one row per second (same `ts` grid as the books and labels). Everything uses
data at or before the prediction time only; rolling windows look backwards.

  book state   imb_1/5/10 (bid vs ask size over top k levels), spread_bps, log depth per side
               (top 5), micro_bps (size-weighted microprice minus mid)
  price        past_ret_1/10/60s (bps), vol_60s/300s (std of 1s returns, bps)
  order flow   ofi_1/5/30s (order flow imbalance at the top of book, per-second snapshots,
               scaled by 5-min mean top size); place_imb / cancel_imb over 5s/30s;
               log_events_5s; modify_share_30s
  trades       trade_imb_5/30s (taker buy vs sell volume), log_trades_30s, secs_since_trade
  market maker top balance manager of the PREVIOUS day (by order events): mm_share_30s of
               events, mm_active_10s, mm_cancel_share_30s; NaN on a period's first day
  time         hour_sin, hour_cos, dow
  cross-venue  bn_* -- Binance as of BINANCE_DELAY_S before now: bn_gap_bps (Binance price
               vs mid minus its trailing 5-min median, i.e. basis removed), bn_ret_1/5/30s,
               bn_trade_imb_5s, bn_log_trades_5s

DeepBook-only models use every column except ts and bn_*; cross-venue models add bn_*.
1-second grid, so the 0.5s Binance window from the plan is not built.

Output: data/processed/features/pool={POOL}/date=YYYY-MM-DD.parquet
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flowcast.config import PROCESSED, RAW
from flowcast.labels import PERIODS, days

BINANCE = {"SUI_USDC": ("spot", "SUIUSDC"), "DEEP_USDC": ("um", "DEEPUSDT")}
BINANCE_DELAY_S = 1
HISTORY_S = 600  # previous-day seconds carried in so windows work across midnight
OUT = PROCESSED / "features"


def _end_of_second(ms: pd.Series) -> pd.Series:
    """Bucket a ms timestamp to the end of its second, the grid's `ts` convention."""
    return (ms // 1000 + 1) * 1000


def _per_second(ts_end: pd.Series, grid: pd.Series, **cols: pd.Series) -> pd.DataFrame:
    """Sum each column per second-end bucket, on the full grid (missing seconds = 0)."""
    df = pd.DataFrame({"ts": ts_end.to_numpy(), **{k: v.to_numpy() for k, v in cols.items()}})
    return df.groupby("ts").sum().reindex(grid.to_numpy(), fill_value=0).reset_index(drop=True)


def _read_day(stream: str, pool: str, day, columns: list[str]) -> pd.DataFrame:
    """One day of a raw stream; empty hours (stored without columns) contribute no rows."""
    files = sorted((RAW / stream / f"pool={pool}" / f"date={day}").glob("*.parquet"))
    return pd.concat([pd.read_parquet(f).reindex(columns=columns) for f in files], ignore_index=True)


def day_inputs(pool: str, day, mm_id: str | None) -> tuple[pd.DataFrame, str | None]:
    """Raw per-second inputs for one day, and that day's most active balance manager."""
    book = pd.read_parquet(PROCESSED / "book" / f"pool={pool}" / f"date={day}.parquet")
    grid = book.ts
    df = book[["ts"] + [c for c in book.columns if c.endswith(("_0", "_1", "_2", "_3", "_4",
                                                                "_5", "_6", "_7", "_8", "_9"))]].copy()

    o = _read_day("orders", pool, day, ["timestamp", "status", "type", "balance_manager_id"])
    bid = o["type"].eq("buy")
    placed, cancel = o.status.eq("Placed"), o.status.isin(["Canceled", "Expired"])
    is_mm = o.balance_manager_id.eq(mm_id) if mm_id else pd.Series(False, index=o.index)
    agg = _per_second(_end_of_second(o.timestamp), grid,
                      place_bid=placed & bid, place_ask=placed & ~bid,
                      cancel_bid=cancel & bid, cancel_ask=cancel & ~bid,
                      modified=o.status.eq("Modified"), events=pd.Series(1, index=o.index),
                      mm_events=is_mm, mm_place=is_mm & placed, mm_cancel=is_mm & cancel)
    if not mm_id:
        agg[["mm_events", "mm_place", "mm_cancel"]] = np.nan
    top_mm = o.balance_manager_id.value_counts().index[0] if len(o) else None

    t = _read_day("trades", pool, day, ["timestamp", "taker_is_bid", "base_volume"])
    buy = t.taker_is_bid.astype(bool)
    agg = agg.join(_per_second(_end_of_second(t.timestamp), grid, buy_vol=t.base_volume.where(buy, 0.0),
                               sell_vol=t.base_volume.where(~buy, 0.0), trades=pd.Series(1, index=t.index)))

    market, sym = BINANCE[pool]
    b = pd.read_parquet(RAW / "binance" / market / f"symbol={sym}" / f"date={day}.parquet",
                        columns=["ts_us", "price", "qty", "is_buyer_maker"])
    b_end = _end_of_second(b.ts_us // 1000)
    agg = agg.join(_per_second(b_end, grid, bn_buy_vol=b.qty.where(~b.is_buyer_maker, 0.0),
                               bn_sell_vol=b.qty.where(b.is_buyer_maker, 0.0),
                               bn_trades=pd.Series(1, index=b.index)))
    last = b.assign(ts=b_end.to_numpy()).groupby("ts").price.last()
    agg["bn_last"] = last.reindex(grid.to_numpy()).to_numpy()  # NaN where no Binance trade
    return pd.concat([df.reset_index(drop=True), agg], axis=1), top_mm


def _imb(a: pd.Series, b: pd.Series) -> pd.Series:
    """(a - b) / (a + b); 0 (neutral) when there is nothing on either side."""
    return ((a - b) / (a + b).replace(0, np.nan)).fillna(0.0).where(a.notna() & b.notna())


def compute(x: pd.DataFrame) -> pd.DataFrame:
    """Features from per-second inputs (contiguous grid). Row i uses rows <= i only."""
    f = pd.DataFrame({"ts": x.ts})
    roll = lambda c, w: x[c].rolling(w, min_periods=1).sum()  # noqa: E731

    # book state
    for k in (1, 5, 10):
        f[f"imb_{k}"] = _imb(x[[f"bid_sz_{i}" for i in range(k)]].sum(axis=1),
                             x[[f"ask_sz_{i}" for i in range(k)]].sum(axis=1))
    mid = (x.bid_px_0 + x.ask_px_0) / 2
    f["spread_bps"] = 1e4 * (x.ask_px_0 - x.bid_px_0) / mid
    f["log_depth_bid_5"] = np.log1p(x[[f"bid_sz_{i}" for i in range(5)]].sum(axis=1))
    f["log_depth_ask_5"] = np.log1p(x[[f"ask_sz_{i}" for i in range(5)]].sum(axis=1))
    micro = (x.ask_px_0 * x.bid_sz_0 + x.bid_px_0 * x.ask_sz_0) / (x.bid_sz_0 + x.ask_sz_0)
    f["micro_bps"] = 1e4 * (micro - mid) / mid

    # price
    logmid = np.log(mid)
    for k in (1, 10, 60):
        f[f"past_ret_{k}s"] = 1e4 * (logmid - logmid.shift(k))
    r1 = 1e4 * logmid.diff()
    for w in (60, 300):
        f[f"vol_{w}s"] = r1.rolling(w, min_periods=w // 2).std()

    # order flow: OFI at the top of book from consecutive 1s snapshots
    bp, bs, ap, as_ = x.bid_px_0, x.bid_sz_0, x.ask_px_0, x.ask_sz_0
    e_bid = np.where(bp > bp.shift(), bs, np.where(bp == bp.shift(), bs - bs.shift(), -bs.shift()))
    e_ask = np.where(ap < ap.shift(), as_, np.where(ap == ap.shift(), as_ - as_.shift(), -as_.shift()))
    ofi = pd.Series(np.nan_to_num(e_bid - e_ask))
    scale = (bs + as_).rolling(300, min_periods=1).mean().replace(0, np.nan)
    for w in (1, 5, 30):
        f[f"ofi_{w}s"] = ofi.rolling(w, min_periods=1).sum() / scale
    for w in (5, 30):
        f[f"place_imb_{w}s"] = _imb(roll("place_bid", w), roll("place_ask", w))
        f[f"cancel_imb_{w}s"] = _imb(roll("cancel_bid", w), roll("cancel_ask", w))
    f["log_events_5s"] = np.log1p(roll("events", 5))
    f["modify_share_30s"] = roll("modified", 30) / roll("events", 30).replace(0, np.nan)

    # trades
    for w in (5, 30):
        f[f"trade_imb_{w}s"] = _imb(roll("buy_vol", w), roll("sell_vol", w))
    f["log_trades_30s"] = np.log1p(roll("trades", 30))
    i = pd.Series(np.arange(len(x)))
    last_trade = i.where(x.trades > 0).ffill()
    f["secs_since_trade"] = (i - last_trade).clip(upper=3600).fillna(3600)

    # market maker (NaN when the previous day's top manager is unknown)
    f["mm_share_30s"] = roll("mm_events", 30) / roll("events", 30).replace(0, np.nan)
    f["mm_active_10s"] = (roll("mm_events", 10) > 0).astype(float).where(x.mm_events.notna())
    mm_c, mm_p = roll("mm_cancel", 30), roll("mm_place", 30)
    f["mm_cancel_share_30s"] = mm_c / (mm_c + mm_p).replace(0, np.nan)

    # time (ts is the end of the second)
    sec = ((x.ts - 1000) // 1000) % 86400
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * sec / 86400), np.cos(2 * np.pi * sec / 86400)
    f["dow"] = pd.to_datetime(x.ts - 1000, unit="ms").dt.dayofweek

    # Binance, delayed: row i sees Binance only up to the end of second i - D
    d = BINANCE_DELAY_S
    px = x.bn_last.ffill().shift(d)
    gap = 1e4 * np.log(px / mid)
    f["bn_gap_bps"] = gap - gap.rolling(300, min_periods=60).median().shift(1)
    lpx = np.log(px)
    for k in (1, 5, 30):
        f[f"bn_ret_{k}s"] = 1e4 * (lpx - lpx.shift(k))
    f["bn_trade_imb_5s"] = _imb(roll("bn_buy_vol", 5), roll("bn_sell_vol", 5)).shift(d)
    f["bn_log_trades_5s"] = np.log1p(roll("bn_trades", 5)).shift(d)
    return f


def top_mm(pool: str, day) -> str | None:
    """The day's most active balance manager by order events."""
    ids = _read_day("orders", pool, day, ["balance_manager_id"]).balance_manager_id
    return ids.value_counts().index[0] if len(ids) else None


def build(pool: str, redo: bool = False) -> None:
    """Resumable: days already on disk are skipped. A day after a skipped one reloads its
    previous day (history + market maker), so results match an uninterrupted run."""
    out_dir = OUT / f"pool={pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for start, end in PERIODS:
        ds = days(start, end)
        prev, mm_id = None, None
        for i, day in enumerate(ds):
            path = out_dir / f"date={day}.parquet"
            if path.exists() and not redo:
                prev = None
                continue
            if prev is None and i > 0:  # resuming: rebuild yesterday's inputs
                prev, mm_id = day_inputs(pool, ds[i - 1], top_mm(pool, ds[i - 2]) if i > 1 else None)
            cur, top = day_inputs(pool, day, mm_id)
            both = cur if prev is None else pd.concat([prev.tail(HISTORY_S), cur], ignore_index=True)
            f = compute(both).tail(len(cur)).reset_index(drop=True)
            tmp = path.with_suffix(".parquet.tmp")
            f.to_parquet(tmp, index=False)
            tmp.replace(path)  # atomic: a half-written day never looks done
            print(f"[features] {pool} {day}  rows={len(f)}  cols={f.shape[1] - 1}  "
                  f"mostly-missing={[c for c in f.columns if f[c].isna().mean() > 0.2]}", flush=True)
            prev, mm_id = cur, top
