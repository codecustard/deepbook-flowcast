"""Evaluation.

Built (`flowcast evaluate`, on the validation month, Sep 2025):
  significance  block bootstrap over DAYS (neighbouring seconds are not independent):
                macro F1 with 95% intervals for every model and rule, and PAIRED differences
                (both sides scored on the same resampled days); a lift counts only if its
                interval excludes 0
  confidence    the most confident LightGBM calls: directional accuracy, average move in the
                predicted direction, and net of cost (entry spread + 2 x taker fee), with intervals
  lead-lag      correlation of DeepBook 1s mid returns with Binance 1s returns k seconds earlier

Still to do: walk-forward within Aug-Oct 2025; the untouched October test and the frozen
2025 models on Jul-Sep 2026 (regime shift); slices by volatility tercile and market-maker state.

Output: runs/{YYYYmmdd-HHMMSS}_eval/{scores,comparisons,confidence,leadlag}.csv + config.json
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

B = 1000                # bootstrap resamples
SEED = 0
TAKER_FEE_BPS = 1.0     # measured from trades: ~1 bp per side when paid in DEEP; makers pay 0
TOP_SHARES = (1.0, 0.2, 0.05, 0.01)
LAGS = range(-5, 11)


def macro_f1(y, p, classes=(-1, 0, 1)) -> float:
    """Mean F1 over the classes, so always predicting the biggest class scores poorly."""
    return float(f1_score(y, p, average="macro", labels=list(classes), zero_division=0))


# ------------------------------------------------------------ bootstrap over days

def day_confusions(y: np.ndarray, p: np.ndarray, day: np.ndarray, n_days: int) -> np.ndarray:
    """(n_days, 9) counts of (true, predicted) class pairs per day; classes -1/0/1."""
    out = np.zeros(n_days * 9, dtype=np.int64)
    np.add.at(out, day * 9 + (y + 1) * 3 + (p + 1), 1)
    return out.reshape(n_days, 9)


def f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Macro F1 from (..., 9) confusion counts (rows = true class)."""
    cm = cm.reshape(*cm.shape[:-1], 3, 3).astype(float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    pred, true = cm.sum(axis=-2), cm.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(pred + true > 0, 2 * tp / (pred + true), 0.0)
    return f1.mean(axis=-1)


def resample_weights(n_days: int, b: int = B, seed: int = SEED) -> np.ndarray:
    """(b, n_days): how many times each day is drawn in each resample (draws with replacement)."""
    draws = np.random.default_rng(seed).integers(0, n_days, (b, n_days))
    return np.stack([np.bincount(d, minlength=n_days) for d in draws])


def interval(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------ predictions on the validation month

def _predictions(pool: str, runs: dict[str, str], log) -> dict:
    """Per horizon: y, day index, spread, future return, and {name: (pred, lean)} for every model/rule."""
    from flowcast.labels import load_labels
    from flowcast.models import rules
    from flowcast.models.data import VALID, load_table, xy
    from flowcast.models.train import load_model

    va = load_table(pool, *VALID)
    rets = load_labels(pool, *VALID).set_index("ts")
    sig = rules.signals(pool, *VALID).set_index("ts")
    chosen = rules.fit(pool)
    out = {}
    for h in ("1s", "10s", "60s"):
        rows = va.dropna(subset=[f"label_{h}"])
        ts = rows.ts.to_numpy()
        d = {"y": rows[f"label_{h}"].to_numpy(int), "day": (ts - 1) // 86_400_000,
             "spread": rows.spread_bps.to_numpy(float), "ret": rets.loc[ts, f"ret_{h}"].to_numpy(),
             "pred": {}}
        d["day"] = d["day"] - d["day"].min()
        s = sig.loc[ts]
        d["pred"]["rule: flat"] = (np.zeros(len(ts), dtype=int), None)
        for rule in ("imbalance", "lead_lag"):
            col, a = chosen[(int(h[:-1]), rule)]
            d["pred"][f"rule: {rule}"] = (rules.predict(s, col, a), None)
        trained_on = [pool] + (["SUI_USDC"] if pool != "SUI_USDC" else [])
        for kind, run in runs.items():
            for fs in ("deepbook", "all"):
                X, _ = xy(va, h, fs)
                for src in trained_on:
                    m = load_model(kind, src, h, fs, run=run)
                    if kind == "lgbm":
                        m.named_steps["model"].set_params(n_jobs=-1)
                    proba = m.predict_proba(X)
                    classes = m.classes_
                    pred = classes[proba.argmax(axis=1)]
                    lean = proba[:, list(classes).index(1)] - proba[:, list(classes).index(-1)]
                    name = f"{kind} {fs}" + ("" if src == pool else " (SUI model)")
                    d["pred"][name] = (pred, lean)
                    log(f"[evaluate] {pool} {h:>3}  predicted {name}")
        out[h] = d
    return out


# ------------------------------------------------------------ the three analyses

COMPARISONS = [  # (label, a, b): is a better than b?
    ("lgbm vs best rule", "lgbm all", "best rule"),
    ("logreg vs best rule", "logreg all", "best rule"),
    ("lgbm vs logreg", "lgbm all", "logreg all"),
    ("Binance adds (lgbm)", "lgbm all", "lgbm deepbook"),
    ("Binance adds (logreg)", "logreg all", "logreg deepbook"),
    ("own model vs SUI model (lgbm)", "lgbm all", "lgbm all (SUI model)"),
]


def significance(pool: str, h: str, d: dict, w: np.ndarray) -> tuple[list[dict], list[dict]]:
    n_days = int(d["day"].max()) + 1
    cms = {k: day_confusions(d["y"], p, d["day"], n_days) for k, (p, _) in d["pred"].items()}
    point = {k: float(f1_from_confusion(cm.sum(axis=0))) for k, cm in cms.items()}
    boot = {k: f1_from_confusion(w @ cm) for k, cm in cms.items()}
    best = max(("rule: imbalance", "rule: lead_lag"), key=point.get)  # chosen on Sep: conservative
    point["best rule"], boot["best rule"] = point[best], boot[best]
    scores = [{"pool": pool, "horizon": h, "model": k, "macro_f1": point[k],
               **dict(zip(("lo", "hi"), interval(boot[k]), strict=True))} for k in d["pred"]]
    comps = []
    for label, a, b in COMPARISONS:
        if a in point and b in point:
            lo, hi = interval(boot[a] - boot[b])
            comps.append({"pool": pool, "horizon": h, "comparison": label, "a": a,
                          "b": best if b == "best rule" else b,
                          "diff": point[a] - point[b], "lo": lo, "hi": hi,
                          "verdict": "real" if lo > 0 else ("worse" if hi < 0 else "not proven")})
    return scores, comps


def confidence(pool: str, h: str, d: dict, w: np.ndarray) -> list[dict]:
    """Most confident LightGBM calls: does the move in the predicted direction beat the cost?"""
    _, lean = d["pred"]["lgbm all"]
    n_days, rows = int(d["day"].max()) + 1, []
    for q in TOP_SHARES:
        m = np.abs(lean) >= np.quantile(np.abs(lean), 1 - q)
        g = np.sign(lean[m]) * d["ret"][m]                       # move in the predicted direction, bps
        cost = d["spread"][m] + 2 * TAKER_FEE_BPS
        ok = ~np.isnan(g)
        g, cost, day = g[ok], cost[ok], d["day"][m][ok]
        sums = (g, g - cost, np.ones_like(g))  # per day: move, net, count
        per_day = np.stack([np.bincount(day, weights=x, minlength=n_days) for x in sums])
        boot = (w @ per_day.T)                                   # (B, 3): resampled sums
        net_lo, net_hi = interval(boot[:, 1] / boot[:, 2])
        mv_lo, mv_hi = interval(boot[:, 0] / boot[:, 2])
        rows.append({"pool": pool, "horizon": h, "top_share": q, "calls": int(ok.sum()),
                     "directional_accuracy": float(np.mean(g > 0) / max(np.mean(g != 0), 1e-12)),
                     "avg_move_bps": float(g.mean()), "move_lo": mv_lo, "move_hi": mv_hi,
                     "avg_cost_bps": float(cost.mean()), "avg_net_bps": float((g - cost).mean()),
                     "net_lo": net_lo, "net_hi": net_hi})
    return rows


def lead_lag(pool: str) -> list[dict]:
    """corr(DeepBook 1s mid return at t, Binance 1s return at t - k); peak at k > 0 = Binance leads."""
    from flowcast.config import RAW
    from flowcast.features import BINANCE
    from flowcast.labels import days, load_labels
    from flowcast.models.data import VALID

    lab = load_labels(pool, *VALID)
    grid = lab.ts.to_numpy()
    market, sym = BINANCE[pool]
    b = pd.concat([pd.read_parquet(RAW / "binance" / market / f"symbol={sym}" / f"date={x}.parquet",
                                   columns=["ts_us", "price"]) for x in days(*VALID)])
    b_end = (b.ts_us // 1_000_000 + 1) * 1000                   # end of second, like the grid
    px = b.assign(ts=b_end.to_numpy()).groupby("ts").price.last().reindex(grid).ffill()
    db = pd.Series(1e4 * np.log(lab.mid.to_numpy())).diff()
    bn = pd.Series(1e4 * np.log(px.to_numpy())).diff()
    return [{"pool": pool, "lag_s": k, "corr": float(db.corr(bn.shift(k)))} for k in LAGS]


def run(pools=("SUI_USDC", "DEEP_USDC")) -> None:
    from flowcast.config import RUNS
    from flowcast.models.train import git_state, latest_run

    runs = {k: latest_run(k).name for k in ("logreg", "lgbm")}
    out_dir = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_eval"
    out_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(json.dumps({
        "model_runs": runs, "pools": list(pools), "bootstrap": {"resamples": B, "seed": SEED, "unit": "day"},
        "taker_fee_bps": TAKER_FEE_BPS, "git": git_state()}, indent=2))
    print(f"[evaluate] models from {runs}; output -> {out_dir}")

    scores, comps, conf, ll = [], [], [], []
    for pool in pools:
        preds = _predictions(pool, runs, print)
        for h, d in preds.items():
            w = resample_weights(int(d["day"].max()) + 1)
            s, c = significance(pool, h, d, w)
            scores += s
            comps += c
            conf += confidence(pool, h, d, w)
        ll += lead_lag(pool)
        del preds

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    tables = {"scores": pd.DataFrame(scores), "comparisons": pd.DataFrame(comps),
              "confidence": pd.DataFrame(conf), "leadlag": pd.DataFrame(ll)}
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    print("\n[evaluate] macro F1 on Sep 2025 with 95% intervals (bootstrap over days)")
    print(tables["scores"].to_string(index=False, float_format=fmt))
    print("\n[evaluate] paired comparisons: 'real' = interval excludes 0")
    print(tables["comparisons"].to_string(index=False, float_format=fmt))
    print(f"\n[evaluate] confident LightGBM calls: move in predicted direction vs cost "
          f"(spread + 2 x {TAKER_FEE_BPS} bp)")
    print(tables["confidence"].to_string(index=False, float_format=fmt))
    ll_t = tables["leadlag"].pivot(index="lag_s", columns="pool", values="corr")
    print("\n[evaluate] lead-lag: corr(DeepBook return at t, Binance return at t - lag)")
    print(ll_t.to_string(float_format=fmt))
    print(f"\n[evaluate] saved to {out_dir}")
