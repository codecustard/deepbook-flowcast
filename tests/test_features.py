"""Features never look ahead: changing the future must not change any past row."""

import numpy as np
import pandas as pd

from flowcast.features import compute


def _inputs(n: int, seed: int) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    x = pd.DataFrame({"ts": (np.arange(n) + 1_757_980_801) * 1000})
    mid = 3.5 + np.cumsum(r.normal(0, 1e-4, n))
    for i in range(10):
        x[f"bid_px_{i}"], x[f"ask_px_{i}"] = mid - 1e-4 * (i + 1), mid + 1e-4 * (i + 1)
        x[f"bid_sz_{i}"], x[f"ask_sz_{i}"] = r.uniform(1, 1e4, n), r.uniform(1, 1e4, n)
    for c in ["place_bid", "place_ask", "cancel_bid", "cancel_ask", "modified", "events", "mm_events",
              "mm_place", "mm_cancel", "trades", "bn_trades"]:
        x[c] = r.integers(0, 5, n)
    for c in ["buy_vol", "sell_vol", "bn_buy_vol", "bn_sell_vol"]:
        x[c] = r.uniform(0, 100, n)
    x["bn_last"] = np.where(r.random(n) < 0.7, mid + r.normal(0, 2e-4, n), np.nan)
    return x


def test_no_lookahead():
    n, cut = 900, 600
    x, y = _inputs(n, 1), _inputs(n, 2)
    z = pd.concat([x.iloc[:cut], y.iloc[cut:]], ignore_index=True)  # same past, different future
    a, b = compute(x).iloc[:cut], compute(z).iloc[:cut]
    pd.testing.assert_frame_equal(a, b)
