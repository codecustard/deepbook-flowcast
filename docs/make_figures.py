"""Figures for the README, from saved run results only (no data or models needed).

    uv run python docs/make_figures.py      # matplotlib comes with the dev extra
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS, OUT = ROOT / "runs", ROOT / "docs" / "img"
WALK, OCT, Y2026, EVAL = ("20261110-173006_walkforward", "20261110-193241_holdout_oct2025",
                          "20261111-184908_holdout_2026", "20261009-232053_eval")
PRIMARY_WALK = {"1s": "retrained vs rule", "10s": "fixed vs rule", "60s": "fixed vs rule"}
H = ["1s", "10s", "60s"]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 (validated all-pairs)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def lift_chart():
    w = pd.read_csv(RUNS / WALK / "summary.csv").query("pool == 'SUI_USDC'")
    w = pd.concat([w[(w.horizon == h) & (w.comparison == PRIMARY_WALK[h])] for h in H])
    o = pd.read_csv(RUNS / OCT / "comparisons.csv").query("pool == 'SUI_USDC' and comparison == 'primary vs rule'")
    y = pd.read_csv(RUNS / Y2026 / "comparisons.csv").query("pool == 'SUI_USDC' and comparison == 'primary vs rule'")
    periods = [("Sep 2025 (walk-forward)", w, S1), ("Oct 2025 (held out)", o, S2),
               ("Jul–Sep 2026 (held out)", y, S3)]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x, width = np.arange(len(H)), 0.26
    for i, (name, d, c) in enumerate(periods):
        d = d.set_index("horizon").loc[H]
        xs = x + (i - 1) * (width + 0.02)
        ax.bar(xs, d["diff"], width, color=c, label=name, zorder=2)
        ax.errorbar(xs, d["diff"], yerr=[d["diff"] - d.lo, d.hi - d["diff"]], fmt="none",
                    ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
        for xi, v, hi in zip(xs, d["diff"], d.hi, strict=True):
            ax.text(xi, hi + 0.004, f"+{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(x, [f"{h} ahead" for h in H])
    ax.set_ylabel("macro F1, model minus best rule")
    ax.set_title("SUI/USDC: LightGBM beats the best rule at every horizon, in every test period",
                 loc="left", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
    fig.text(0.01, 0.02, "Bars: gain over the better of the order book imbalance rule and the Binance lead-lag rule.\n"
             "Whiskers: 95% interval, bootstrap over days.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "lift_by_period.png", dpi=200)


def cost_chart():
    sources = [("Sep 2025", pd.read_csv(RUNS / EVAL / "confidence.csv")),
               ("Oct 2025", pd.read_csv(RUNS / OCT / "confidence.csv")),
               ("Jul–Sep 2026", pd.read_csv(RUNS / Y2026 / "confidence.csv"))]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6))
    for ax, (name, c) in zip(axes, sources, strict=True):
        d = c.query("pool == 'SUI_USDC' and horizon == '10s'").sort_values("top_share", ascending=False)
        x, width = np.arange(len(d)), 0.38
        ax.bar(x - 0.2, d.avg_move_bps, width, color=S1, label="avg move in predicted direction", zorder=2)
        ax.bar(x + 0.2, d.avg_cost_bps, width, color=S2, label="avg cost (spread + fees)", zorder=2)
        ax.set_xticks(x, [f"top {s:.0%}" if s < 1 else "all" for s in d.top_share], fontsize=8)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(0.1, None)
    axes[0].set_ylabel("basis points (log scale)")
    fig.suptitle("SUI/USDC, 10s: even the most confident calls move less than the cost of trading them",
                 x=0.01, ha="left", fontsize=11)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(OUT / "cost_vs_move.png", dpi=200)


def leadlag_chart():
    ll = pd.read_csv(RUNS / EVAL / "leadlag.csv").query("pool == 'SUI_USDC'").sort_values("lag_s")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(ll.lag_s, ll["corr"], color=S1, linewidth=2, marker="o", markersize=5, zorder=2)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.axvline(0, color=GRID, linewidth=1)
    for lag in (0, 1, 2):
        v = ll.set_index("lag_s").loc[lag, "corr"]
        ax.annotate(f"{v:.2f}", (lag, v), textcoords="offset points", xytext=(8, 2), fontsize=8, color=INK2)
    ax.set_xlabel("Binance return this many seconds earlier (negative = DeepBook first)")
    ax.set_ylabel("correlation")
    ax.set_title("SUI/USDC, Sep 2025: DeepBook moves with Binance, or up to ~2 s after", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "leadlag.png", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    lift_chart()
    cost_chart()
    leadlag_chart()
    print(f"saved to {OUT}")
