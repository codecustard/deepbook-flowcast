"""Features (week 3), from rebuilt books and raw events. Everything uses data at or
before the prediction time only.

  book state   imbalance at levels 1/5/10, spread, depth, microprice - mid
  order flow   order flow imbalance (OFI); placements and cancels per side over 1s/5s/30s
  trades       signed volume, buy/sell count ratio, time since last trade
  volatility   realized vol over 1m/5m
  trader-aware top market maker's share of the book, whether it quoted recently, its cancel rate
  time         hour of day, day of week
  cross-venue  Binance last price minus DeepBook mid; Binance return and signed volume over
               the last 0.5s/1s/5s. Clock = DeepBook checkpoint time; main test only uses
               Binance data >= 1s older than the prediction time (0s as a sensitivity check)
"""


def build(pool: str) -> None:
    raise NotImplementedError("week 3: features")
