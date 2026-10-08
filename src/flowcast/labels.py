"""Labels: mid-price direction N seconds ahead -- up (1), flat (0) or down (-1).

  ret_Ns     log return of the mid over the next N seconds, in basis points (bps), so one
             threshold means the same thing at $3.5 (2025) and $1 (2026)
  label_Ns   up if ret > th, down if ret < -th, else flat; missing when either mid is
             missing (one side of the book empty)

th = the 1/3 quantile of |ret| on the TRAIN month only, per pool and horizon, so the classes
come out roughly balanced. At short horizons most seconds have no move at all, so th is 0
and "flat" stays above 1/3 (exactly-zero moves); macro F1 keeps that from flattering a
model that always says flat.

Output: data/processed/labels/pool={POOL}/date=YYYY-MM-DD.parquet  (ts, mid, ret_*, label_*)
        data/processed/labels/thresholds.json
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from flowcast.config import PROCESSED

HORIZONS = (1, 10, 60)
TRAIN = ("2025-08-01", "2025-09-01")
PERIODS = (("2025-08-01", "2025-11-01"), ("2026-07-01", "2026-10-01"))
OUT = PROCESSED / "labels"


def days(start: str, end: str) -> list[date]:
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    return [a + timedelta(i) for i in range((b - a).days)]


def load_book(pool: str, start: str, end: str, columns: list[str]) -> pd.DataFrame:
    d = PROCESSED / "book" / f"pool={pool}"
    return pd.concat([pd.read_parquet(d / f"date={x}.parquet", columns=["ts", *columns])
                      for x in days(start, end)], ignore_index=True)


def mid(book: pd.DataFrame) -> pd.Series:
    return (book.bid_px_0 + book.ask_px_0) / 2  # NaN when either side is empty


def future_returns(m: pd.Series, h: int) -> pd.Series:
    """bps return from now to h seconds ahead; rows are a contiguous 1-second grid."""
    return 1e4 * np.log(m.shift(-h) / m)


def label(ret: pd.Series, th: float) -> pd.Series:
    out = pd.Series(np.select([ret > th, ret < -th], [1, -1], 0), index=ret.index).astype("Int8")
    return out.mask(ret.isna())


def fit_thresholds(pool: str) -> dict[int, float]:
    m = mid(load_book(pool, *TRAIN, ["bid_px_0", "ask_px_0"]))
    return {h: float(future_returns(m, h).abs().quantile(1 / 3)) for h in HORIZONS}


def make_labels(pool: str) -> dict[int, float]:
    th = fit_thresholds(pool)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "thresholds.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    saved[pool] = {str(h): v for h, v in th.items()}
    path.write_text(json.dumps(saved, indent=2))

    out_dir = OUT / f"pool={pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for start, end in PERIODS:
        b = load_book(pool, start, end, ["bid_px_0", "ask_px_0"])
        df = pd.DataFrame({"ts": b.ts, "mid": mid(b)})
        for h in HORIZONS:
            df[f"ret_{h}s"] = future_returns(df.mid, h)
            df[f"label_{h}s"] = label(df[f"ret_{h}s"], th[h])
        day = pd.to_datetime(df.ts - 1, unit="ms").dt.strftime("%Y-%m-%d")  # ts = end of second
        for d, g in df.groupby(day):
            g.to_parquet(out_dir / f"date={d}.parquet", index=False)
        print(f"[labels] {pool} {start} -> {end}: {len(df):,} seconds; class shares (down/flat/up):",
              "  ".join(f"{h}s " + "/".join(f"{df[f'label_{h}s'].value_counts(normalize=True).get(k, 0):.2f}"
                                             for k in (-1, 0, 1)) for h in HORIZONS))
    print(f"[labels] {pool} thresholds (bps, fit on {TRAIN[0]}..{TRAIN[1]}): "
          + ", ".join(f"{h}s={v:.3f}" for h, v in th.items()))
    return th


def load_labels(pool: str, start: str, end: str) -> pd.DataFrame:
    d = OUT / f"pool={pool}"
    return pd.concat([pd.read_parquet(d / f"date={x}.parquet") for x in days(start, end)], ignore_index=True)
