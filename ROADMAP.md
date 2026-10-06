# Roadmap

Full plan: "DeepBook Order-Flow Study: Project Plan" (Claude doc). Summary below.

## Research questions

1. **Lift:** does each model step beat the imbalance rule out of sample, and by how much?
2. **Lead-lag:** how long does DeepBook's mid trail Binance's SUI price, and do Binance features beat DeepBook-only models? If Binance dominates, own-book signals add little.
3. **Horizon:** 1s vs 10s vs 60s, against DeepBook's ~0.5s settlement time.
4. **Thin pool:** do SUI/USDC-trained models carry over to DEEP/USDC (test-only)?
5. **Regime:** volatile periods; the dominant market maker quoting vs absent (trader IDs).
6. **Market shift:** do 2025-trained models keep their edge on the thin, bot-heavy 2026 market?
7. **Economics:** does any gain survive the spread and fees?

## Data

- **DeepBook:** SUI_USDC (main), DEEP_USDC (test-only), from the public Mysten indexer.
- **Binance:** daily aggTrades from [data.binance.vision](https://data.binance.vision): SUIUSDC and SUIUSDT spot, DEEPUSDT USD-M perp (DEEP has no Binance spot market). Microsecond timestamps on spot. No historical best quotes, so the reference is the last trade price; the live predictor adds Binance's websocket quotes.

## Periods

| Period | SUI/USDC per day | Role |
|---|---|---|
| Aug–Oct 2025 | $46–50M, ~11x today's active takers | Main: train, validate, test (Oct untouched) |
| Jul–Sep 2026 | $6–9M, bot-heavy | Regime-shift test, live deployment |

## Timeline (ends 2026-10-31)

- [ ] **Week of Sep 28 (slipped into Oct 5):** collector + raw store; Binance downloader; live snapshot recorder on the VPS; pull Aug–Oct 2025 and Jul–Sep 2026 (DeepBook + Binance)
- [ ] **Week of Oct 5:** book reconstruction + validation; labels; baselines (always-flat, imbalance rule, Binance lead-lag rule)
- [ ] **Week of Oct 12:** features incl. Binance cross-venue; logistic regression + gradient boosting, with and without Binance; walk-forward eval with bootstrap intervals; lead-lag cross-correlation
- [ ] **Week of Oct 19:** CNN/LSTM; slices by horizon, pool, regime, bot state; SHAP
- [ ] **Week of Oct 26:** live predictor, dashboard (model vs rule scoreboard), API, Docker; README write-up

## Cut order if behind

Docker → weekly report → DEEP_USDC test pool → 60s horizon. Never cut: the rule baseline, walk-forward eval, confidence intervals, the README.

## Stretch

- Hyperliquid comparison (record its live book by websocket from week 1)
- Spoofing detection on the same data
- Sui Overflow / DeepBook grant pitch
