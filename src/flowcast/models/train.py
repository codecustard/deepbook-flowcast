"""Train and score models: fit on Aug 2025 (hyperparameters tuned with TimeSeriesSplit inside
Aug), scored once on Sep 2025, next to the best baseline rule.

DEEP_USDC is the thin, test-only pool: the SUI_USDC-trained model is scored on it as-is
("does it carry over?"), and a DEEP-trained model is scored alongside.

Every run gets its own folder, so nothing is overwritten:
  runs/{YYYYmmdd-HHMMSS}_{kind}/results.csv    the results table (saved after each model)
  runs/{YYYYmmdd-HHMMSS}_{kind}/config.json    grid, dates, cv, git commit: what was tried
  runs/{YYYYmmdd-HHMMSS}_{kind}/cv_results.csv every setting tried and its score on each Aug fold
  runs/{YYYYmmdd-HHMMSS}_{kind}/models/{kind}_{pool}_{horizon}_{features}.joblib
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, ParameterGrid, TimeSeriesSplit
from tqdm import tqdm

from flowcast.config import ROOT, RUNS
from flowcast.evaluate import macro_f1
from flowcast.models import rules
from flowcast.models.data import FEATURE_SETS, TRAIN, VALID, default_step, load_table, xy
from flowcast.models.pipeline import make_pipeline

GRIDS = {
    "logreg": {"model__C": [1e-4, 1e-3, 1e-2, 1e-1], "model__class_weight": ["balanced", None]},
    "lgbm": {"model__n_estimators": [150, 400], "model__num_leaves": [15, 63],
             "model__class_weight": ["balanced", None]},
}
HORIZONS = ("1s", "10s", "60s")
SCORER = make_scorer(macro_f1)
CV_FOLDS = 3


def git_state() -> dict:
    """The commit this run's code came from, and whether there were uncommitted changes."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "src"))}


def latest_run(kind: str) -> Path:
    runs = sorted(RUNS.glob(f"*_{kind}"))
    if not runs:
        raise FileNotFoundError(f"no {kind} runs under {RUNS}")
    return runs[-1]


def load_model(kind: str, pool: str, horizon: str, features: str = "all", run: str | None = None):
    """A trained model from `run` (folder name), or from the latest run of that kind."""
    d = RUNS / run if run else latest_run(kind)
    return joblib.load(d / "models" / f"{kind}_{pool}_{horizon}_{features}.joblib")


def run(kind: str = "logreg", pools=("SUI_USDC", "DEEP_USDC"), horizons=HORIZONS,
        feature_sets=FEATURE_SETS) -> pd.DataFrame:
    out_dir = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_{kind}"
    (out_dir / "models").mkdir(parents=True)
    (out_dir / "config.json").write_text(json.dumps({
        "kind": kind, "grid": GRIDS[kind], "pools": list(pools), "horizons": list(horizons),
        "feature_sets": list(feature_sets), "train": TRAIN, "valid": VALID,
        "cv": f"TimeSeriesSplit({CV_FOLDS})", "train_step": {h: default_step(h) for h in horizons},
        "git": git_state(),
    }, indent=2))
    print(f"[train] run folder: {out_dir}")
    path = lambda p, h, fs: out_dir / "models" / f"{kind}_{p}_{h}_{fs}.joblib"  # noqa: E731

    rows, cv_rows = [], []
    bar = tqdm(total=len(pools) * len(horizons) * len(feature_sets), desc="train", unit="model")
    for pool in pools:
        tqdm.write(f"[train] {pool}: loading Aug + Sep, scoring baselines...")
        base = rules.run(pool).query("rule != 'flat'").groupby("horizon").macro_f1.max()
        tr, va = load_table(pool, *TRAIN), load_table(pool, *VALID)
        for h in horizons:
            for fs in feature_sets:
                X, y = xy(tr, h, fs, default_step(h))
                Xv, yv = xy(va, h, fs)
                tag = f"{kind} {pool} {h:>3} {fs:8}"
                n = len(ParameterGrid(GRIDS[kind]))
                tqdm.write(f"[train] {tag}  tuning on {len(X):,} Aug rows ({n} settings x {CV_FOLDS} folds)")
                t0 = time.time()
                search = GridSearchCV(make_pipeline(kind, list(X.columns)), GRIDS[kind], scoring=SCORER,
                                      cv=TimeSeriesSplit(CV_FOLDS), n_jobs=-1, verbose=2).fit(X, y)
                model = search.best_estimator_
                cv = pd.DataFrame(search.cv_results_)
                keep = [c for c in cv.columns if c.startswith(("param_", "split")) or c in
                        ("mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time")]
                cv_rows.append(cv[keep].assign(pool=pool, horizon=h, features=fs))
                pd.concat(cv_rows).to_csv(out_dir / "cv_results.csv", index=False)
                joblib.dump(model, path(pool, h, fs))
                r = {"model": kind, "pool": pool, "horizon": h, "features": fs, "trained_on": pool,
                     **{k.split("__")[-1]: v for k, v in search.best_params_.items()},
                     "cv_f1": search.best_score_, "train_f1": macro_f1(y, model.predict(X)),
                     "sep_f1": macro_f1(yv, model.predict(Xv)), "best_rule_f1": base[h],
                     "fit_s": round(time.time() - t0)}
                rows.append(r)
                tqdm.write(f"[train] {tag}  {search.best_params_}  fit {r['fit_s']}s  train F1="
                           f"{r['train_f1']:.3f}  Sep F1={r['sep_f1']:.3f}  (best rule {base[h]:.3f})")
                if pool != "SUI_USDC" and "SUI_USDC" in pools:  # carry-over: SUI model, as-is
                    sui = joblib.load(path("SUI_USDC", h, fs))
                    sui_row = next(x for x in rows if x["pool"] == "SUI_USDC" and x["horizon"] == h
                                   and x["features"] == fs)
                    rows.append({**sui_row, "pool": pool, "best_rule_f1": base[h], "cv_f1": None,
                                 "train_f1": None, "sep_f1": macro_f1(yv, sui.predict(Xv)), "fit_s": None})
                    tqdm.write(f"[train] {tag}  SUI-trained model on {pool}: Sep F1={rows[-1]['sep_f1']:.3f}")
                bar.update(1)
                pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)  # saved as we go
    bar.close()
    out = pd.DataFrame(rows)
    print(f"\n[train] {kind}: trained on {TRAIN[0]}..{TRAIN[1]}, scored on {VALID[0]}..{VALID[1]} (macro F1)")
    print(out.drop(columns=["model"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"[train] saved to {out_dir}")
    return out
