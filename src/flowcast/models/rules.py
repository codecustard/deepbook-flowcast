"""Baselines -- the rungs every model must beat. Each predicts up (1) / flat (0) / down (-1).

  flat        always 0
  imbalance   x = (bid size - ask size) / (bid size + ask size) over the top k levels;
              up if x > a, down if x < -a (k in 1/2/3/5, fit with a)
  lead-lag    x = Binance last price (as of >= 1s before now) vs DeepBook mid, in bps, minus
              its trailing 5-minute median (removes the USDT/USDC and perp basis);
              up if x > a (Binance is above: DeepBook should follow), down if x < -a

Each threshold `a` is fit on the train month (macro F1) per pool and horizon, then frozen.
Scored on the validation month; October 2025 and 2026 stay untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flowcast.config import RAW
from flowcast.evaluate import macro_f1
from flowcast.labels import HORIZONS, TRAIN, days, load_book, load_labels

VALID = ("2025-09-01", "2025-10-01")
BINANCE = {"SUI_USDC": ("spot", "SUIUSDC"), "DEEP_USDC": ("um", "DEEPUSDT")}
LEVELS = (1, 2, 3, 5)
BINANCE_DELAY_S = 1
BASIS_WINDOW_S = 300
GRIDS = {"imbalance": np.round(np.linspace(0, 0.9, 19), 2),
         "lead_lag": np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 5, 8, 12, 20])}


def by_threshold(x: np.ndarray, a: float) -> np.ndarray:
    return np.select([x > a, x < -a], [1, -1], 0)


def imbalance(book: pd.DataFrame, k: int) -> np.ndarray:
    b = book[[f"bid_sz_{i}" for i in range(k)]].sum(axis=1)
    a = book[[f"ask_sz_{i}" for i in range(k)]].sum(axis=1)
    return ((b - a) / (b + a)).to_numpy()


def binance_last(pool: str, start: str, end: str, ts: pd.Series) -> np.ndarray:
    """Binance last trade price as of BINANCE_DELAY_S before each DeepBook second-end ts."""
    market, sym = BINANCE[pool]
    t = pd.concat([pd.read_parquet(RAW / "binance" / market / f"symbol={sym}" / f"date={d}.parquet",
                                   columns=["ts_us", "price"]) for d in days(start, end)])
    t = t.sort_values("ts_us", kind="stable")
    cutoff_us = (ts.to_numpy() - BINANCE_DELAY_S * 1000) * 1000
    i = np.searchsorted(t.ts_us.to_numpy(), cutoff_us, side="left") - 1  # trades strictly before cutoff
    px = t.price.to_numpy()
    return np.where(i >= 0, px[np.clip(i, 0, None)], np.nan)


def lead_lag(pool: str, start: str, end: str, book: pd.DataFrame) -> np.ndarray:
    m = (book.bid_px_0 + book.ask_px_0) / 2
    gap = pd.Series(1e4 * np.log(binance_last(pool, start, end, book.ts) / m))
    basis = gap.rolling(BASIS_WINDOW_S, min_periods=60).median().shift(1)  # past only
    return (gap - basis).to_numpy()


def signals(pool: str, start: str, end: str) -> pd.DataFrame:
    cols = ["bid_px_0", "ask_px_0"] + [f"{s}_sz_{i}" for s in ("bid", "ask") for i in range(max(LEVELS))]
    book = load_book(pool, start, end, cols)
    lab = load_labels(pool, start, end)
    df = lab[["ts"] + [f"label_{h}s" for h in HORIZONS]].copy()
    for k in LEVELS:
        df[f"imbalance_{k}"] = imbalance(book, k)
    df["lead_lag"] = lead_lag(pool, start, end, book)
    return df


def run(pool: str) -> pd.DataFrame:
    train, valid = signals(pool, *TRAIN), signals(pool, *VALID)
    rows = []
    for h in HORIZONS:
        y = f"label_{h}s"
        tr, va = train.dropna(subset=[y]), valid.dropna(subset=[y])
        ytr, yva = tr[y].to_numpy(int), va[y].to_numpy(int)
        rows.append({"horizon": f"{h}s", "rule": "flat", "signal": "", "a": None,
                     "macro_f1": macro_f1(yva, np.zeros_like(yva)), "accuracy": (yva == 0).mean()})
        for rule, grid in GRIDS.items():
            cols = [f"imbalance_{k}" for k in LEVELS] if rule == "imbalance" else [rule]
            x = {c: np.nan_to_num(tr[c].to_numpy()) for c in cols}
            col, a = max(((c, g) for c in cols for g in grid),
                         key=lambda cg: macro_f1(ytr, by_threshold(x[cg[0]], cg[1])))
            p = by_threshold(np.nan_to_num(va[col].to_numpy()), a)
            rows.append({"horizon": f"{h}s", "rule": rule, "signal": col, "a": float(a),
                         "macro_f1": macro_f1(yva, p), "accuracy": (yva == p).mean()})
    out = pd.DataFrame(rows)
    print(f"\n[baselines] {pool}: thresholds fit on {TRAIN[0]}..{TRAIN[1]}, scored on {VALID[0]}..{VALID[1]}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return out
