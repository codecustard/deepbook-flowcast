# deepbook-flowcast

**When does machine learning beat a simple rule at predicting short-term price moves on DeepBook, Sui's fully on-chain order book?**

DeepBook publishes every order placed, modified, cancelled and filled, with trader IDs, going back over a year. Most markets hide that. This project uses it to compare the classic **order book imbalance rule** against a ladder of models (logistic regression, gradient boosting, a CNN/LSTM) and to find *where* ML adds value: by horizon, pool liquidity, volatility, and market-maker behavior. It also tests how much of DeepBook's predictability is just lag behind Binance's SUI price. The main period is DeepBook's Aug–Oct 2025 peak; the quieter, bot-heavy 2026 market is the regime-shift test.

## Status

Week 1: DeepBook collector and Binance downloader working. Book reconstruction, features, and models are stubs. See [ROADMAP.md](ROADMAP.md).

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env                       # optional overrides

uv run flowcast pools                      # list DeepBook pools
uv run flowcast collect --start 2025-08-01 --end 2025-11-01   # DeepBook backfill (resumable)
uv run flowcast binance --start 2025-08-01 --end 2025-11-01   # Binance aggTrades, same days
uv run flowcast status                     # what's on disk
uv run flowcast snapshots --pools SUI_USDC # record live books (runs until Ctrl-C)
uv run pytest -q
```

## Layout

```
src/flowcast/
  collector/        raw data only; imports nothing from the ML side (tests/test_boundary.py)
    indexer_api.py  client for the public DeepBook indexer
    fetch.py        adaptive windows under the 10,000-row response cap
    store.py        hourly parquet in data/raw/ + manifest
    collect.py      resumable backfill, one unit = (stream, pool, hour)
    snapshots.py    live order book recorder -> data/snapshots/
    binance.py      Binance daily aggTrades -> data/raw/binance/ (lead-lag reference)
  book.py           rebuild the order book from raw events         (week 2)
  labels.py         up / flat / down, N seconds ahead               (week 2)
  features.py       imbalance, order flow, trades, vol, trader-aware (week 3)
  models/           the model ladder                                 (weeks 2-4)
  evaluate.py       walk-forward, bootstrap intervals, regime slices (weeks 3-4)
  cli.py            `flowcast` command
data/raw/           {orders,trades}/pool=X/date=Y/hour=HH.parquet
data/raw/binance/   {spot,um}/symbol=X/date=Y.parquet
data/snapshots/     pool=X/date=Y/part-*.parquet
```

## Data source

Public [DeepBookV3 indexer](https://docs.sui.io/standards/deepbookv3-indexer) by Mysten Labs (free, no key). If it ever goes away, it can be self-hosted from [MystenLabs/deepbookv3](https://github.com/MystenLabs/deepbookv3).

Binance reference trades come from [Binance public data](https://data.binance.vision) (free, no key): SUIUSDC and SUIUSDT spot, DEEPUSDT USD-M perp.

## License

MIT
