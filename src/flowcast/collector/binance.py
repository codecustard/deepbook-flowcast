"""Binance reference prices for the lead-lag angle: free daily aggTrades files.

Source: https://data.binance.vision (public, no key). One file per (symbol, day):

  data/raw/binance/{market}/symbol={SYMBOL}/date=YYYY-MM-DD.parquet
  data/raw/binance/_manifest.jsonl   one line per day: rows, missing, fetched_at

Symbols are "SUIUSDC" (spot, the default) or "um:DEEPUSDT" (USD-M futures; DEEP has no
Binance spot market). A file's existence means the day is done, so runs are resumable.
A day Binance has no file for (404) is written as an empty file and flagged `missing`.

Columns: agg_id, price, qty, first_id, last_id, ts_us (UTC microseconds), is_buyer_maker.
Spot files carry microsecond timestamps from 2025 on and no header; futures files carry
milliseconds and a header. Both are normalized to ts_us here. Checksums are verified.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm

from flowcast.config import RAW, settings

BASE = "https://data.binance.vision/data"
ROOT = RAW / "binance"
MANIFEST = ROOT / "_manifest.jsonl"
MARKETS = {"spot": "spot", "um": "futures/um"}
COLUMNS = ["agg_id", "price", "qty", "first_id", "last_id", "ts", "is_buyer_maker"]
_lock = threading.Lock()


def parse_symbol(s: str) -> tuple[str, str]:
    """'SUIUSDC' -> ('spot', 'SUIUSDC'); 'um:DEEPUSDT' -> ('um', 'DEEPUSDT')."""
    market, _, sym = s.strip().rpartition(":")
    market = market or "spot"
    if market not in MARKETS:
        raise ValueError(f"unknown market {market!r} in {s!r}; use spot or um")
    return market, sym.upper()


def day_path(market: str, symbol: str, day: date) -> Path:
    return ROOT / market / f"symbol={symbol}" / f"date={day:%Y-%m-%d}.parquet"


def url(market: str, symbol: str, day: date) -> str:
    return f"{BASE}/{MARKETS[market]}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day:%Y-%m-%d}.zip"


def parse_csv(raw: bytes) -> pd.DataFrame:
    """Parse one aggTrades CSV (with or without header) into normalized columns."""
    first = raw.split(b"\n", 1)[0]
    has_header = not first[:1].isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None, usecols=range(7))
    df.columns = COLUMNS
    ts = df["ts"].astype("int64")
    # ms timestamps are ~1.7e12; us are ~1.7e15
    df["ts_us"] = ts * 1000 if len(ts) and ts.iloc[0] < 10**14 else ts
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().eq("true")
    return df.drop(columns="ts")[["agg_id", "price", "qty", "first_id", "last_id", "ts_us", "is_buyer_maker"]]


def _empty() -> pd.DataFrame:
    return parse_csv(b"agg_id,price,qty,first_id,last_id,ts,is_buyer_maker\n")


def fetch_day(client: httpx.Client, market: str, symbol: str, day: date) -> pd.DataFrame | None:
    """The day's trades, or None if Binance has no file for it."""
    u = url(market, symbol, day)
    r = client.get(u)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    c = client.get(u + ".CHECKSUM")
    if c.status_code == 200:
        want = c.text.split()[0]
        got = hashlib.sha256(r.content).hexdigest()
        if want != got:
            raise ValueError(f"checksum mismatch for {u}")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        return parse_csv(z.read(z.namelist()[0]))


def write_day(market: str, symbol: str, day: date, df: pd.DataFrame | None) -> int:
    missing = df is None
    df = _empty() if missing else df
    path = day_path(market, symbol, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    with _lock:
        with MANIFEST.open("a") as f:
            f.write(json.dumps({"market": market, "symbol": symbol, "date": f"{day:%Y-%m-%d}",
                                "rows": len(df), "missing": missing,
                                "fetched_at": datetime.now(UTC).isoformat()}) + "\n")
    return len(df)


def days(start: date, end: date) -> list[date]:
    """Days in [start, end), stopping before today (Binance publishes the next day)."""
    last = min(end, datetime.now(UTC).date())
    return [start + timedelta(d) for d in range((last - start).days)]


def run(symbols: list[str], start: date, end: date, workers: int | None = None, redo: bool = False) -> None:
    syms = [parse_symbol(s) for s in symbols]
    ROOT.mkdir(parents=True, exist_ok=True)
    todo = [(m, s, d) for m, s in syms for d in days(start, end) if redo or not day_path(m, s, d).exists()]
    total = len(syms) * len(days(start, end))
    print(f"[binance] {len(todo)} of {total} (symbol, day) files to fetch; {total - len(todo)} on disk")
    if not todo:
        return
    rows = missing = fails = 0
    with httpx.Client(timeout=settings.request_timeout, follow_redirects=True) as client, \
            ThreadPoolExecutor(max_workers=workers or settings.workers) as ex:
        futs = {ex.submit(fetch_day, client, *u): u for u in todo}
        with tqdm(total=len(futs), desc="binance", unit="day") as bar:
            for f in as_completed(futs):
                m, s, d = futs[f]
                try:
                    df = f.result()
                    missing += df is None
                    rows += write_day(m, s, d, df)
                except Exception as e:  # noqa: BLE001 - day stays missing, retried next run
                    fails += 1
                    tqdm.write(f"[binance] failed {m}:{s} {d}: {e}")
                bar.update(1)
                bar.set_postfix(rows=rows, failed=fails)
    print(f"[binance] done: {rows:,} trades, {missing} days with no Binance file, {fails} failed")


def status() -> None:
    if not MANIFEST.exists():
        print("binance: nothing collected yet")
        return
    m = pd.read_json(MANIFEST, lines=True, convert_dates=False, dtype={"date": str}).sort_values("fetched_at")
    m = m.drop_duplicates(["market", "symbol", "date"], keep="last")
    for (mk, s), g in m.groupby(["market", "symbol"]):
        print(f"binance {mk}:{s:10} {len(g):4} days  {g['date'].min()} -> {g['date'].max()}  "
              f"missing={int(g['missing'].sum())}  rows={int(g['rows'].sum()):>12,}")
