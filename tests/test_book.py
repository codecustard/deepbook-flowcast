"""Book replay: Modified carries total size incl. filled; trade-throughs are caught."""

import pandas as pd

from flowcast.book import KIND, Book, replay


def _ev(rows):
    return pd.DataFrame(rows, columns=["ts", "kind", "oid", "is_bid", "price", "qty", "filled"]).assign(
        kind=lambda d: d.kind.map(KIND))


def test_modified_after_fills_and_full_fill_leaves_nothing():
    # the real 2025-10-30 sequence: place 1210, two fills, two Modifieds (total incl. filled), last fill
    ev = _ev([(1000, "Placed", "a", False, 2.0, 1210.0, 0.0),
              (1000, "Modified", "a", False, 2.0, 933.1, 595.4),
              (1000, "fill", "a", False, 2.0, 374.7, 0.0),
              (1000, "fill", "a", False, 2.0, 220.7, 0.0),
              (1500, "Modified", "a", False, 2.0, 815.1, 595.4),
              (1800, "fill", "a", False, 2.0, 219.7, 0.0)])
    book = Book()
    _, s = replay(book, ev, 0, 2000)
    assert book.orders == {} and book.prices[False] == [] and s["trade_throughs"] == 0


def test_fill_behind_untouched_better_order_is_a_trade_through():
    ev = _ev([(1000, "Placed", "a", False, 2.0, 10.0, 0.0),   # resting best ask
              (1000, "Placed", "b", False, 2.1, 10.0, 0.0),
              (2500, "fill", "b", False, 2.1, 5.0, 0.0)])    # a buyer skipped 2.0: impossible
    _, s = replay(Book(), ev, 0, 3000)
    assert s["trade_throughs"] == 1
