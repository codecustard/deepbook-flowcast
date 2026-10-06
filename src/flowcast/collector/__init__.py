"""The collector: DeepBook indexer in, raw events on disk out.

  indexer_api.py  thin client for the public DeepBook indexer
  fetch.py        adaptive windowed fetching (the indexer caps responses at 10,000 rows)
  store.py        hourly parquet files under data/raw/, plus a manifest
  collect.py      resumable backfill: one unit of work = one (stream, pool, hour)
  snapshots.py    live order book recorder (the indexer keeps no historical snapshots)
  binance.py      Binance daily aggTrades (reference price for the lead-lag angle)

Boundary: nothing here imports the ML side (book, features, labels, models, train,
evaluate, serve). The ML side reads data/raw and data/snapshots and never writes there.
Enforced by tests/test_boundary.py.
"""
