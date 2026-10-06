"""The model ladder (weeks 2-4), each step reported against the one below it:

  1. always "flat"
  2. imbalance rule, threshold tuned on train     -- the heuristic under test
  2b. lead-lag rule: DeepBook mid moves toward Binance's last price
  3. logistic regression on ~10 features
  4. gradient boosting (LightGBM) on all tabular features
  5. CNN + LSTM (DeepLOB-style) in PyTorch on the raw book sequence

Models 3-5 are each trained twice: DeepBook-only features, and with the Binance
cross-venue features added, so the lift from each source is reported separately.
"""
