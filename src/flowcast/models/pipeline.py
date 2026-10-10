"""Preprocessing + model as one scikit-learn Pipeline, so every step is fit on training data
only (per notebooks/01_eda.ipynb):

  1. log1p on long-tailed, non-negative columns
  2. impute missing with the training median, plus a was-missing flag per affected column
  3. clip at the training 1st / 99th percentiles
  4. standardize (mean 0, std 1)
  5. model

Tree models (lgbm) skip 1-4: splits only compare values, so scale and tails don't matter,
and LightGBM routes missing values itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

LOG_COLS = ["spread_bps", "vol_60s", "vol_300s", "secs_since_trade"]


class Clip(BaseEstimator, TransformerMixin):
    """Clip each column to its training [lo, hi] percentiles."""

    def __init__(self, lo: float = 1, hi: float = 99):
        self.lo, self.hi = lo, hi

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        a = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = np.nanpercentile(a, self.lo, axis=0), np.nanpercentile(a, self.hi, axis=0)
        self.n_features_in_ = a.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_ if input_features is None else np.asarray(input_features)


def preprocess(columns: list[str]) -> Pipeline:
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    logs = [c for c in LOG_COLS if c in columns]
    return Pipeline([
        ("log", ColumnTransformer([("log1p", log1p, logs)], remainder="passthrough",
                                  verbose_feature_names_out=False)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("clip", Clip()),
        ("scale", StandardScaler()),
    ]).set_output(transform="pandas")


def make_pipeline(kind: str, columns: list[str], **params) -> Pipeline:
    if kind == "logreg":
        model = LogisticRegression(class_weight="balanced", max_iter=1000, **params)
        return Pipeline([("pre", preprocess(columns)), ("model", model)])
    if kind == "lgbm":
        # n_jobs=1: the hyperparameter search already runs one fit per core
        model = LGBMClassifier(learning_rate=0.05, min_child_samples=200, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, n_jobs=1, verbose=-1, **params)
        return Pipeline([("model", model)])
    raise ValueError(f"unknown model kind {kind!r}")


def importances(pipe: Pipeline) -> pd.Series:
    """LightGBM feature importance: total loss reduction (gain) from splits on each feature."""
    b = pipe.named_steps["model"].booster_
    return pd.Series(b.feature_importance("gain"), index=b.feature_name()).sort_values(ascending=False)


def shap_values(pipe: Pipeline, X: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """LightGBM SHAP values per class: {"down"|"flat"|"up": rows of X x (features + "bias")}.

    Each row adds up to the model's raw score (log-odds) for that class: bias + sum of the
    feature contributions. Positive = that feature pushed this prediction toward the class.
    """
    m = pipe.named_steps["model"]
    raw = m.predict(X, pred_contrib=True)
    k, cols = X.shape[1] + 1, [*X.columns, "bias"]
    names = {-1: "down", 0: "flat", 1: "up"}
    return {names[c]: pd.DataFrame(raw[:, i * k:(i + 1) * k], columns=cols, index=X.index)
            for i, c in enumerate(m.classes_)}


def coefficients(pipe: Pipeline) -> pd.DataFrame:
    """Logistic regression weights: one row per (preprocessed) feature, one column per class."""
    m = pipe.named_steps["model"]
    names = pipe.named_steps["pre"].get_feature_names_out()
    classes = [{-1: "down", 0: "flat", 1: "up"}[c] for c in m.classes_]
    return pd.DataFrame(m.coef_.T, index=names, columns=classes)
