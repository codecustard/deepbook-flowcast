"""Evaluation (weeks 3-4).

  splits        walk-forward within Aug-Oct 2025 (Oct = untouched test); then 2025 models
                scored frozen on Jul-Sep 2026 (the regime-shift test)
  metrics       macro F1, per-class accuracy, lift over the rule in points, calibration
  significance  block bootstrap over days; a lift counts only if its interval excludes 0
  slices        horizon x pool x volatility tercile x market-maker state
  lead-lag      cross-correlate Binance and DeepBook returns at lags 0-10s; accuracy vs
                horizon per model, with DeepBook's ~0.5s settlement time marked
  thin pool     DEEP_USDC is test-only: SUI_USDC-trained models scored as-is, then retrained once
"""

from sklearn.metrics import f1_score


def macro_f1(y, p, classes=(-1, 0, 1)) -> float:
    """Mean F1 over the classes, so always predicting the biggest class scores poorly."""
    return float(f1_score(y, p, average="macro", labels=list(classes), zero_division=0))


def run() -> None:
    raise NotImplementedError("weeks 3-4: evaluation")
