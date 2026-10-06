"""Raw event storage: one parquet file per (stream, pool, hour), plus a manifest.

  data/raw/{stream}/pool={POOL}/date=YYYY-MM-DD/hour=HH.parquet
  data/raw/_manifest.jsonl     one line per written file: rows, truncated, fetched_at

A file's existence means that hour is done, which is what makes collection resumable.
Rows are stored as the indexer returned them (plus `pool`), deduplicated and sorted
by timestamp. Nothing derived lives here.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from flowcast.config import RAW

STREAMS: dict[str, list[str]] = {
    "orders": ["order_id", "status", "timestamp"],  # /order_updates: Placed, Canceled, Modified, Expired
    "trades": ["trade_id"],                         # /trades
}

NUMERIC = {
    "price", "original_quantity", "remaining_quantity", "filled_quantity",
    "base_volume", "quote_volume", "taker_fee", "maker_fee",
}

MANIFEST = RAW / "_manifest.jsonl"
_lock = threading.Lock()


def hour_path(stream: str, pool: str, hour: datetime) -> Path:
    return RAW / stream / f"pool={pool}" / f"date={hour:%Y-%m-%d}" / f"hour={hour:%H}.parquet"


def is_done(stream: str, pool: str, hour: datetime) -> bool:
    return hour_path(stream, pool, hour).exists()


def normalize(rows: list[dict], pool: str, stream: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({"timestamp": pd.Series([], dtype="int64"), "pool": pd.Series([], dtype="str")})
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif c == "timestamp":
            df[c] = df[c].astype("int64")
        elif df[c].dtype == object:
            df[c] = df[c].astype("string")
    df["pool"] = pool
    keys = [k for k in STREAMS[stream] if k in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys)
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def write_hour(stream: str, pool: str, hour: datetime, rows: list[dict], truncated: bool) -> int:
    df = normalize(rows, pool, stream)
    path = hour_path(stream, pool, hour)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)  # atomic: a half-written hour never looks done
    with _lock:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open("a") as f:
            f.write(json.dumps({
                "stream": stream, "pool": pool, "hour": hour.isoformat(), "rows": len(df),
                "truncated": truncated, "fetched_at": datetime.now(UTC).isoformat(),
            }) + "\n")
    return len(df)


def manifest() -> pd.DataFrame:
    if not MANIFEST.exists():
        return pd.DataFrame(columns=["stream", "pool", "hour", "rows", "truncated", "fetched_at"])
    df = pd.read_json(MANIFEST, lines=True)
    # a redone hour appears twice; the latest write wins
    return df.sort_values("fetched_at").drop_duplicates(["stream", "pool", "hour"], keep="last")
