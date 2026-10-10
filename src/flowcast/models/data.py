"""Training tables: features (X) joined to labels (y) on `ts`.

  train   Aug 2-31 2025 (Aug 1 dropped: warm-up day, no market-maker features)
  valid   Sep 2025
  test    Oct 2025 and Jul-Sep 2026 -- not loaded here; touched once, at the end

Columns dropped per notebooks/01_eda.ipynb: modify_share_30s (constant 0 in Aug),
cancel_imb_30s (r=0.95 with place_imb_30s), mm_active_10s (r=0.84 with mm_share_30s),
dow (only 4-5 of each weekday in a month).
"""

from __future__ import annotations

import pandas as pd

from flowcast.config import PROCESSED
from flowcast.labels import days, load_labels

TRAIN = ("2025-08-02", "2025-09-01")
VALID = ("2025-09-01", "2025-10-01")
DROP = ["modify_share_30s", "cancel_imb_30s", "mm_active_10s", "dow"]
FEATURE_SETS = ("deepbook", "all")  # deepbook = no Binance (bn_*) columns


def load_table(pool: str, start: str, end: str) -> pd.DataFrame:
    """Features + labels for [start, end), one row per second, float32 features."""
    d = PROCESSED / "features" / f"pool={pool}"
    f = pd.concat([pd.read_parquet(d / f"date={x}.parquet") for x in days(start, end)], ignore_index=True)
    lab = load_labels(pool, start, end)
    lab = lab[["ts"] + [c for c in lab.columns if c.startswith("label_")]]
    cols = [c for c in f.columns if c != "ts"]
    f[cols] = f[cols].astype("float32")
    return f.merge(lab, on="ts", how="inner")


def feature_columns(table: pd.DataFrame, feature_set: str = "all") -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    cols = [c for c in table.columns if c != "ts" and not c.startswith("label_") and c not in DROP]
    return cols if feature_set == "all" else [c for c in cols if not c.startswith("bn_")]


def xy(table: pd.DataFrame, horizon: str, feature_set: str = "all", step: int = 1):
    """X, y for one horizon ('1s', '10s', '60s'); rows without a label dropped; every `step`-th row."""
    t = table.dropna(subset=[f"label_{horizon}"]).iloc[::step]
    return t[feature_columns(t, feature_set)], t[f"label_{horizon}"].astype(int)


def default_step(horizon: str) -> int:
    """Training rows: neighbouring seconds are near-duplicates, so thin to non-overlapping
    windows (every 10th row at 10s, 60th at 60s); at 1s, every 5th row."""
    return max(int(horizon.rstrip("s")), 5)


def load_split(pool: str, horizon: str = "10s", feature_set: str = "all", step: int | None = None):
    """X_tr, y_tr, X_va, y_va. Training rows are thinned; validation keeps every second."""
    tr, va = load_table(pool, *TRAIN), load_table(pool, *VALID)
    return (*xy(tr, horizon, feature_set, step or default_step(horizon)), *xy(va, horizon, feature_set))
