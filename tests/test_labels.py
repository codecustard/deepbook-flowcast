"""Labels and the scorer."""

import numpy as np
import pandas as pd

from flowcast.evaluate import macro_f1
from flowcast.labels import future_returns, label


def test_labels_up_flat_down_and_missing_at_the_end():
    mid = pd.Series([100.0, 100.0, 100.02, 100.0, np.nan])
    ret = future_returns(mid, 1)  # bps: 0, +2, -2, nan, nan
    assert label(ret, th=1.0).tolist() == [0, 1, -1, pd.NA, pd.NA]


def test_always_flat_scores_low_on_macro_f1_despite_high_accuracy():
    y = np.array([0] * 8 + [1, -1])
    flat = np.zeros_like(y)
    assert (y == flat).mean() == 0.8
    assert abs(macro_f1(y, flat) - (2 * 0.8 * 1.0 / 1.8) / 3) < 1e-12
    assert macro_f1(y, y) == 1.0
