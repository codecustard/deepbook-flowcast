"""Model pipeline: preprocessing is learned from training data only, and the plumbing works."""

import numpy as np
import pandas as pd

from flowcast.evaluate import macro_f1
from flowcast.models.pipeline import make_pipeline, preprocess


def test_validation_extremes_are_clipped_to_training_range():
    r = np.random.default_rng(0)
    train = pd.DataFrame({"spread_bps": r.uniform(0, 5, 1000), "imb_1": r.uniform(-1, 1, 1000)})
    pre = preprocess(list(train.columns)).fit(train)
    valid = pd.DataFrame({"spread_bps": [1e6], "imb_1": [-50.0]})  # far outside anything seen in training
    t, v = pre.transform(train), pre.transform(valid)
    assert (v.max() <= t.max() + 1e-9).all() and (v.min() >= t.min() - 1e-9).all()


def test_logreg_learns_a_planted_signal_with_missing_values():
    r = np.random.default_rng(1)
    x = r.normal(size=3000)
    y = np.select([x > 0.5, x < -0.5], [1, -1], 0)
    X = pd.DataFrame({"signal": x, "noise": r.normal(size=3000), "spread_bps": r.uniform(0, 3, 3000)})
    X.loc[::50, "noise"] = np.nan
    m = make_pipeline("logreg", list(X.columns)).fit(X, y)
    assert macro_f1(y, m.predict(X)) > 0.9


def test_lgbm_learns_a_planted_signal_with_missing_values():
    r = np.random.default_rng(2)
    x = r.normal(size=3000)
    y = np.select([x > 0.5, x < -0.5], [1, -1], 0)
    X = pd.DataFrame({"signal": x, "noise": r.normal(size=3000)})
    X.loc[::50, "noise"] = np.nan
    m = make_pipeline("lgbm", list(X.columns), n_estimators=50).fit(X, y)
    assert macro_f1(y, m.predict(X)) > 0.9
