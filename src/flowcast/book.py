"""Order book reconstruction (week 2). ML side: reads data/raw, writes data/processed.

Replay the raw event stream (Placed, Modified, Canceled, Expired orders + trade fills)
into a limit order book, and emit the top-N levels per event and per second.

The indexer keeps no historical snapshots, so each rebuilt window starts empty and needs a
warm-up period before its book is trusted. Validate against data/snapshots.
"""


def rebuild(pool: str, start: str, end: str) -> None:
    raise NotImplementedError("week 2: book reconstruction")
