"""Dedupe must not drop distinct fills that share a trade_id."""

from flowcast.collector.store import normalize


def test_fills_sharing_trade_id_are_kept():
    rows = [
        {"trade_id": "7", "event_digest": "a0", "timestamp": 1, "price": "1.0"},
        {"trade_id": "7", "event_digest": "b0", "timestamp": 2, "price": "1.1"},
        {"trade_id": "7", "event_digest": "b0", "timestamp": 2, "price": "1.1"},  # true duplicate
    ]
    assert len(normalize(rows, "P", "trades")) == 2


def test_same_ms_modifications_are_kept():
    base = {"order_id": "9", "status": "Modified", "timestamp": 1, "price": "0.7"}
    rows = [{**base, "remaining_quantity": "15756"}, {**base, "remaining_quantity": "15462.9"},
            {**base, "remaining_quantity": "15462.9"}]  # last one is a true duplicate
    assert len(normalize(rows, "P", "orders")) == 2
