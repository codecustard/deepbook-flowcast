"""Held-out tests, run once each: October 2025, then Jul-Sep 2026 (regime shift).

Choices fixed BEFORE looking, from the walk-forward run (runs/20261110-173006_walkforward):
  primary model    SUI_USDC 1s: LightGBM retrained on Aug 2 - Sep 30 (retraining helped, +0.048)
                   everything else: the Aug-only model (retraining hurt SUI 10s/60s, no effect on DEEP)
  alternative      the other variant, reported alongside
  best rule        imbalance and lead-lag refit on Aug 2 - Sep 30; the better on that window is used
Settings for retrained models: frozen from the Aug grid search (walkforward.BASE_RUN).

  flowcast final-train                       train + save the Aug-Sep models and rule thresholds
  flowcast holdout oct2025                   score them on October 2025 (trains first if needed)
  flowcast holdout 2026 --final-run <name>   same frozen models on Jul-Sep 2026

Each period refuses to run twice (that would turn the test into validation); --force overrides
and is recorded in config.json.

Output: runs/{YYYYmmdd-HHMMSS}_holdout_{period}/{scores,comparisons,confidence}.csv + config.json
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from flowcast.evaluate import (
    TAKER_FEE_BPS,
    confidence,
    day_confusions,
    f1_from_confusion,
    interval,
    macro_f1,
    resample_weights,
)
from flowcast.labels import load_labels
from flowcast.models import rules
from flowcast.models.data import TRAIN, default_step, load_table, xy
from flowcast.models.pipeline import make_pipeline
from flowcast.models.walkforward import BASE_RUN, FEATURES, HORIZONS, frozen_params, in_train

FINAL_TRAIN = (TRAIN[0], "2025-10-01")
PERIODS = {"oct2025": ("2025-10-01", "2025-11-01"), "2026": ("2026-07-01", "2026-10-01")}
POOLS = ("SUI_USDC", "DEEP_USDC")
RETRAINED_PRIMARY = {("SUI_USDC", "1s")}
DAY_MS = 86_400_000


def primary_variant(pool: str, h: str) -> str:
    return "retrained" if (pool, h) in RETRAINED_PRIMARY else "fixed"


def train_final(pools=POOLS, horizons=HORIZONS) -> str:
    """Retrain LightGBM on Aug 2 - Sep 30 (frozen settings) and refit the rules; save both."""
    from flowcast.config import RUNS
    from flowcast.models.train import git_state

    out_dir = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_final"
    (out_dir / "models").mkdir(parents=True)
    chosen_rules = {}
    for pool in pools:
        t0 = time.time()
        print(f"[final] {pool}: loading {FINAL_TRAIN[0]}..{FINAL_TRAIN[1]}...")
        table = load_table(pool, *FINAL_TRAIN)
        sig = rules.signals(pool, *FINAL_TRAIN)
        params = frozen_params(pool)
        for h in horizons:
            h_s = int(h.rstrip("s"))
            t1 = time.time()
            tr = table[in_train(table.ts, FINAL_TRAIN[0], FINAL_TRAIN[1], h_s)]
            X, y = xy(tr, h, FEATURES, default_step(h))
            m = make_pipeline("lgbm", list(X.columns), **params[h]).set_params(model__n_jobs=-1).fit(X, y)
            joblib.dump(m, out_dir / "models" / f"lgbm_{pool}_{h}_{FEATURES}.joblib")
            s_tr = sig[in_train(sig.ts, FINAL_TRAIN[0], FINAL_TRAIN[1], h_s)]
            fitted = rules.fit(pool, train=s_tr)
            yl = s_tr.dropna(subset=[f"label_{h}"])
            scored = {
                r: macro_f1(yl[f"label_{h}"].to_numpy(int), rules.predict(yl, *fitted[(h_s, r)]))
                for r in rules.GRIDS
            }
            best = max(scored, key=scored.get)
            col, a = fitted[(h_s, best)]
            chosen_rules[f"{pool} {h}"] = {"rule": best, "signal": col, "a": a, "train_f1": scored}
            print(f"[final] {pool} {h:>3}  lgbm fit on {len(X):,} rows  rule {best} ({col}, a={a:g})  "
                  f"{time.time() - t1:.0f}s")
        print(f"[final] {pool}: done in {time.time() - t0:.0f}s")
        del table, sig
    (out_dir / "config.json").write_text(json.dumps({
        "train": FINAL_TRAIN, "features": FEATURES, "params_from": BASE_RUN,
        "rules": chosen_rules, "git": git_state()}, indent=2))
    print(f"[final] saved to {out_dir}")
    return out_dir.name


def run(period: str, final_run: str | None = None, pools=POOLS, horizons=HORIZONS, force: bool = False):
    from flowcast.config import RUNS
    from flowcast.models.train import git_state, load_model

    if period not in PERIODS:
        raise ValueError(f"period must be one of {list(PERIODS)}")
    done = sorted(RUNS.glob(f"*_holdout_{period}"))
    if done and not force:
        raise SystemExit(f"[holdout] {period} was already tested ({done[-1].name}); it is a one-time test. "
                         "Use --force only if that run was broken, and say so in docs/LOG.md.")
    final_run = final_run or train_final(pools, horizons)
    final_cfg = json.loads((RUNS / final_run / "config.json").read_text())
    start, end = PERIODS[period]

    out_dir = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_holdout_{period}"
    out_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(json.dumps({
        "period": PERIODS[period], "final_run": final_run, "fixed_run": BASE_RUN,
        "primary": {f"{p} {h}": primary_variant(p, h) for p in pools for h in horizons},
        "rules": final_cfg["rules"], "forced": force, "taker_fee_bps": TAKER_FEE_BPS,
        "git": git_state()}, indent=2))
    print(f"[holdout] {period} {start}..{end}; models from {final_run} + {BASE_RUN}; output -> {out_dir}")

    scores, comps, conf = [], [], []
    for pool in pools:
        t0 = time.time()
        te = load_table(pool, start, end)
        sig = rules.signals(pool, start, end).set_index("ts")
        rets = load_labels(pool, start, end).set_index("ts")
        print(f"[holdout] {pool}: loaded {len(te):,} rows in {time.time() - t0:.0f}s")
        for h in horizons:
            rows = te.dropna(subset=[f"label_{h}"])
            ts = rows.ts.to_numpy()
            X, y = xy(te, h, FEATURES)
            day = (ts - 1) // DAY_MS
            d = {"y": y.to_numpy(int), "day": day - day.min(), "spread": rows.spread_bps.to_numpy(float),
                 "ret": rets.loc[ts, f"ret_{h}"].to_numpy(), "pred": {}}
            models = {"fixed": load_model("lgbm", pool, h, FEATURES, run=BASE_RUN),
                      "retrained": load_model("lgbm", pool, h, FEATURES, run=final_run)}
            prim = primary_variant(pool, h)
            for variant, m in models.items():
                m.named_steps["model"].set_params(n_jobs=-1)
                proba, cls = m.predict_proba(X), list(m.classes_)
                name = "primary" if variant == prim else "alternative"
                d["pred"][name] = (np.asarray(cls)[proba.argmax(axis=1)],
                                   proba[:, cls.index(1)] - proba[:, cls.index(-1)])
            r = final_cfg["rules"][f"{pool} {h}"]
            d["pred"]["best rule"] = (rules.predict(sig.loc[ts], r["signal"], r["a"]), None)
            d["pred"]["always flat"] = (np.zeros(len(ts), dtype=int), None)

            n_days = int(d["day"].max()) + 1
            w = resample_weights(n_days)
            cms = {k: day_confusions(d["y"], p, d["day"], n_days) for k, (p, _) in d["pred"].items()}
            point = {k: float(f1_from_confusion(cm.sum(axis=0))) for k, cm in cms.items()}
            boot = {k: f1_from_confusion(w @ cm) for k, cm in cms.items()}
            variant_of = {"primary": prim, "alternative": "retrained" if prim == "fixed" else "fixed"}
            for k in d["pred"]:
                lo, hi = interval(boot[k])
                scores.append({"pool": pool, "horizon": h, "model": k,
                               "detail": variant_of.get(k, f"{r['rule']} ({r['signal']}, a={r['a']:g})"
                                                        if k == "best rule" else ""),
                               "macro_f1": point[k], "lo": lo, "hi": hi})
            for label, a, b in (("primary vs rule", "primary", "best rule"),
                                ("alternative vs rule", "alternative", "best rule"),
                                ("primary vs alternative", "primary", "alternative")):
                lo, hi = interval(boot[a] - boot[b])
                comps.append({"pool": pool, "horizon": h, "comparison": label, "diff": point[a] - point[b],
                              "lo": lo, "hi": hi,
                              "verdict": "real" if lo > 0 else ("worse" if hi < 0 else "not proven")})
            conf += confidence(pool, h, {**d, "pred": {"lgbm all": d["pred"]["primary"]}}, w)
            print(f"[holdout] {pool} {h:>3}  primary ({prim}) {point['primary']:.3f}  "
                  f"alternative {point['alternative']:.3f}  rule {point['best rule']:.3f}")
            for name, t in (("scores", scores), ("comparisons", comps), ("confidence", conf)):
                pd.DataFrame(t).to_csv(out_dir / f"{name}.csv", index=False)
        del te, sig, rets

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    print(f"\n[holdout] {period}: macro F1 with 95% intervals (bootstrap over days)")
    print(pd.DataFrame(scores).to_string(index=False, float_format=fmt))
    print("\n[holdout] paired comparisons: 'real' = interval excludes 0")
    print(pd.DataFrame(comps).to_string(index=False, float_format=fmt))
    print(f"\n[holdout] confident primary-model calls vs cost (spread + 2 x {TAKER_FEE_BPS} bp)")
    print(pd.DataFrame(conf).to_string(index=False, float_format=fmt))
    print(f"\n[holdout] saved to {out_dir}")
