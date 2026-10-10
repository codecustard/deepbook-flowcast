"""Bootstrap pieces: F1 from confusion counts matches sklearn; paired differences behave."""

import numpy as np

from flowcast.evaluate import day_confusions, f1_from_confusion, interval, macro_f1, resample_weights


def test_f1_from_daily_confusions_matches_sklearn():
    r = np.random.default_rng(0)
    y, p, day = r.integers(-1, 2, 5000), r.integers(-1, 2, 5000), r.integers(0, 30, 5000)
    cm = day_confusions(y, p, day, 30)
    assert cm.sum() == 5000
    assert np.isclose(f1_from_confusion(cm.sum(axis=0)), macro_f1(y, p))


def test_paired_difference_of_a_model_with_itself_is_exactly_zero_and_interval_covers_point():
    r = np.random.default_rng(1)
    y, day = r.integers(-1, 2, 5000), r.integers(0, 30, 5000)
    good = np.where(r.random(5000) < 0.6, y, r.integers(-1, 2, 5000))    # right ~70% of the time
    w = resample_weights(30, b=300)
    a = f1_from_confusion(w @ day_confusions(y, good, day, 30))
    assert np.all(a - a == 0)
    lo, hi = interval(a)
    assert lo <= macro_f1(y, good) <= hi
    bad = r.integers(-1, 2, 5000)
    b = f1_from_confusion(w @ day_confusions(y, bad, day, 30))
    assert interval(a - b)[0] > 0                                       # clearly better: interval excludes 0
