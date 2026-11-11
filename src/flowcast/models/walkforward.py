"""Walk-forward evaluation inside Aug-Sep 2025: is September's lift steady week to week?

Folds: September split into weeks. For each week, train on Aug 2 .. week start (expanding),
dropping training rows whose label looks past the week start (purge = horizon), then score
the week. October 2025 stays untouched.

Per fold and horizon, three predictors on the same rows:
  lgbm retrained   LightGBM refit on the fold's training window, settings frozen from the
                   Aug tuning run (no grid search)
  lgbm fixed       the Aug-only model from that run, as-is (does an older model fade?)
  best rule        imbalance and lead-lag thresholds refit on the fold's training window;
                   the better of the two on that window is used (chosen on train, not test)

Features: "all" (with Binance). Pools: SUI_USDC, DEEP_USDC (own-pool models).

Summary: macro F1 per week, plus paired differences pooled over all September days with a
bootstrap over days (same method as `flowcast evaluate`).

Output: runs/{YYYYmmdd-HHMMSS}_walkforward/{weeks,summary}.csv + config.json
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from flowcast.evaluate import day_confusions, f1_from_confusion, interval, macro_f1, resample_weights
from flowcast.models import rules
from flowcast.models.data import TRAIN, VALID, default_step, load_table, xy
from flowcast.models.pipeline import make_pipeline

DAY_MS = 86_400_000
FOLDS = (("2025-09-01", "2025-09-08"), ("2025-09-08", "2025-09-15"),
         ("2025-09-15", "2025-09-22"), ("2025-09-22", "2025-10-01"))
HORIZONS = ("1s", "10s", "60s")
FEATURES = "all"
BASE_RUN = "20261009-191600_lgbm"
PARAMS = ("n_estimators", "num_leaves", "class_weight")


def ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)


def in_train(ts: pd.Series, train_start: str, test_start: str, horizon_s: int) -> pd.Series:
    """Rows usable for training before `test_start`: ts is the END of the second, and the label
    at ts covers (ts, ts + horizon], so it must end at or before the test start."""
    return (ts > ms(train_start)) & (ts + horizon_s * 1000 <= ms(test_start))


def in_test(ts: pd.Series, start: str, end: str) -> pd.Series:
    return (ts > ms(start)) & (ts <= ms(end))


def frozen_params(pool: str, run: str = BASE_RUN) -> dict[str, dict]:
    """{horizon: LightGBM settings} chosen by the Aug grid search in `run`."""
    from flowcast.config import RUNS

    r = pd.read_csv(RUNS / run / "results.csv")
    r = r[(r.pool == pool) & (r.trained_on == pool) & (r.features == FEATURES)]
    out = {}
    for _, row in r.iterrows():
        p = {k: row[k] for k in PARAMS}
        p["class_weight"] = None if pd.isna(p["class_weight"]) else p["class_weight"]
        p["n_estimators"], p["num_leaves"] = int(p["n_estimators"]), int(p["num_leaves"])
        out[row.horizon] = p
    return out


def best_rule(sig_tr: pd.DataFrame, sig_te: pd.DataFrame, pool: str, h_s: int) -> tuple[str, np.ndarray]:
    """Fit both rules on the training window; predict the test rows with whichever scored
    better on training."""
    chosen = rules.fit(pool, train=sig_tr)
    ytr = sig_tr.dropna(subset=[f"label_{h_s}s"])
    best, best_f1 = None, -1.0
    for rule in rules.GRIDS:
        col, a = chosen[(h_s, rule)]
        f1 = macro_f1(ytr[f"label_{h_s}s"].to_numpy(int), rules.predict(ytr, col, a))
        if f1 > best_f1:
            best, best_f1 = (rule, col, a), f1
    rule, col, a = best
    return f"{rule} ({col}, a={a:g})", rules.predict(sig_te, col, a)


def run(pools=("SUI_USDC", "DEEP_USDC"), horizons=HORIZONS, run_name: str = BASE_RUN) -> pd.DataFrame:
    from flowcast.config import RUNS
    from flowcast.models.train import git_state, load_model

    out_dir = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_walkforward"
    out_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(json.dumps({
        "folds": FOLDS, "train_start": TRAIN[0], "horizons": list(horizons), "features": FEATURES,
        "params_from": run_name, "purge": "label end <= test start", "git": git_state()}, indent=2))
    print(f"[walkforward] output -> {out_dir}")

    weeks, summary = [], []
    for pool in pools:
        t0 = time.time()
        print(f"[walkforward] {pool}: loading Aug 2 .. Oct 1 features + rule signals...")
        table = load_table(pool, TRAIN[0], VALID[1])
        sig = rules.signals(pool, TRAIN[0], VALID[1])
        params = frozen_params(pool, run_name)
        print(f"[walkforward] {pool}: loaded in {time.time() - t0:.0f}s")
        for h in horizons:
            h_s = int(h.rstrip("s"))
            fixed = load_model("lgbm", pool, h, FEATURES, run=run_name)
            ys, ps, days = [], {"lgbm retrained": [], "lgbm fixed": [], "best rule": []}, []
            for k, (start, end) in enumerate(FOLDS, 1):
                t1 = time.time()
                tr = table[in_train(table.ts, TRAIN[0], start, h_s)]
                te = table[in_test(table.ts, start, end)]
                X, y = xy(tr, h, FEATURES, default_step(h))
                Xt, yt = xy(te, h, FEATURES)
                model = make_pipeline("lgbm", list(X.columns), **params[h])
                model.set_params(model__n_jobs=-1).fit(X, y)
                p_re, p_fx = model.predict(Xt), fixed.predict(Xt)

                s_tr = sig[in_train(sig.ts, TRAIN[0], start, h_s)]
                s_te = sig[in_test(sig.ts, start, end)]
                s_te = s_te.set_index("ts").loc[te.dropna(subset=[f"label_{h}"]).ts].reset_index()
                rule_name, p_rule = best_rule(s_tr, s_te, pool, h_s)
                assert (s_te[f"label_{h}"].to_numpy(int) == yt.to_numpy()).all(), "rule rows misaligned"

                row = {"pool": pool, "horizon": h, "fold": k, "test": f"{start}..{end}",
                       "train_rows": len(X), "test_rows": len(Xt),
                       "lgbm_retrained": macro_f1(yt, p_re), "lgbm_fixed": macro_f1(yt, p_fx),
                       "best_rule": macro_f1(yt, p_rule), "rule": rule_name, "fit_s": round(time.time() - t1)}
                row["lift_vs_rule"] = row["lgbm_retrained"] - row["best_rule"]
                weeks.append(row)
                pd.DataFrame(weeks).to_csv(out_dir / "weeks.csv", index=False)
                print(f"[walkforward] {pool} {h:>3} week {k} ({start}..{end})  "
                      f"retrained {row['lgbm_retrained']:.3f}  fixed {row['lgbm_fixed']:.3f}  "
                      f"rule {row['best_rule']:.3f}  [{rule_name}]  {row['fit_s']}s")
                ys.append(yt.to_numpy())
                days.append((te.dropna(subset=[f"label_{h}"]).ts.to_numpy() - 1) // DAY_MS)
                for name, p in (("lgbm retrained", p_re), ("lgbm fixed", p_fx), ("best rule", p_rule)):
                    ps[name].append(np.asarray(p, dtype=int))

            y_all = np.concatenate(ys)
            d_all = np.concatenate(days)
            d_all = d_all - d_all.min()
            n_days = int(d_all.max()) + 1
            w = resample_weights(n_days)
            cms = {k: day_confusions(y_all, np.concatenate(v), d_all, n_days) for k, v in ps.items()}
            boot = {k: f1_from_confusion(w @ cm) for k, cm in cms.items()}
            point = {k: float(f1_from_confusion(cm.sum(axis=0))) for k, cm in cms.items()}
            for label, a, b in (("retrained vs rule", "lgbm retrained", "best rule"),
                                ("fixed vs rule", "lgbm fixed", "best rule"),
                                ("retrained vs fixed", "lgbm retrained", "lgbm fixed")):
                lo, hi = interval(boot[a] - boot[b])
                summary.append({"pool": pool, "horizon": h, "comparison": label,
                                "a_f1": point[a], "b_f1": point[b], "diff": point[a] - point[b],
                                "lo": lo, "hi": hi,
                                "verdict": "real" if lo > 0 else ("worse" if hi < 0 else "not proven")})
            pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
        del table, sig

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    w_df, s_df = pd.DataFrame(weeks), pd.DataFrame(summary)
    print("\n[walkforward] macro F1 per September week (expanding train from Aug 2)")
    print(w_df.drop(columns=["rule"]).to_string(index=False, float_format=fmt))
    print("\n[walkforward] pooled over September, paired bootstrap over days")
    print(s_df.to_string(index=False, float_format=fmt))
    print(f"\n[walkforward] saved to {out_dir}")
    return w_df
