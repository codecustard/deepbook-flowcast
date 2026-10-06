"""Binance aggTrades parsing: spot (no header, microseconds) and futures (header, milliseconds)."""

from datetime import date

from flowcast.collector import binance as b

SPOT = b"48557761,3.68770000,500.00000000,64203917,64203917,1757894400260792,False,True\n" \
       b"48557762,3.68770000,100.00000000,64203918,64203918,1757894400374224,True,True\n"
UM = b"agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n" \
     b"1,0.1500,20,1,1,1757894400260,true\n"


def test_spot_no_header_microseconds():
    df = b.parse_csv(SPOT)
    assert len(df) == 2
    assert df["ts_us"].iloc[0] == 1757894400260792
    assert df["is_buyer_maker"].tolist() == [False, True]


def test_futures_header_milliseconds_normalized():
    df = b.parse_csv(UM)
    assert len(df) == 1 and df["ts_us"].iloc[0] == 1757894400260000 and df["is_buyer_maker"].iloc[0]


def test_symbols_and_paths():
    assert b.parse_symbol("SUIUSDC") == ("spot", "SUIUSDC")
    assert b.parse_symbol("um:deepusdt") == ("um", "DEEPUSDT")
    assert b.url("um", "DEEPUSDT", date(2025, 9, 15)).endswith(
        "futures/um/daily/aggTrades/DEEPUSDT/DEEPUSDT-aggTrades-2025-09-15.zip")


def test_empty_frame_has_columns():
    cols = ["agg_id", "price", "qty", "first_id", "last_id", "ts_us", "is_buyer_maker"]
    assert list(b._empty().columns) == cols
