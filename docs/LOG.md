# Project log

One entry per step: what was run, on which code, where the output is, what we learned, and what we decided.
Newest at the bottom. Commands run from the repo root. Notebooks show the results of each stage:
`00_data_pipeline` → `01_eda` → `02_training` → `03_models` → `04_evaluation`.

---

## 2026-10-05 — Setup and a 3-day test pull
- **Ran:** `make setup`; `uv run pytest -q` (8 passed); `flowcast collect` + `flowcast binance` for 2025-09-15 → 09-18.
- **Result:** 4.44M DeepBook rows in 2m41s, 0 truncated, 0 gaps; Binance 9 files, 1.33M trades, 0 missing; 89 MB.
- **Bug found:** trades were deduplicated on `trade_id`, which repeats across the fills of one taker order → ~16% of
  SUI trades dropped in busy hours. **Fix:** dedupe on `event_digest` (`ea4fd40`); test trades re-pulled (+11.8%).
- **Noticed:** SUI orders were 2–6k/h until 09:00 on Sep 15, then ~60k/h: the dominant market maker (one balance
  manager, 93–97% of events) was absent before that. A ready-made "MM absent" sample.
- **Decided:** run the full pulls locally (MacBook) instead of the VPS; skip the live snapshot recorder for now
  (validate the book against trades instead).

## 2026-10-05 → 10-07 — Full backfill: Aug–Oct 2025 and Jul–Sep 2026
- **Ran:** `caffeinate -i bash -c 'for i in 1 2 3; do make collect-2025 collect-2026; done'` (resumable; stopped and
  resumed several times; the indexer slowed to 7–40 s per request at times).
- **Bug found (orders):** dedupe key (`order_id`, `status`, `timestamp`) collapsed real events: one order can be
  **Modified** several times in the same millisecond with different sizes. **Fix:** drop only exact duplicate rows
  (`5386db0`). **Repair:** re-fetched only the 395 hours of 2025 that contained any `Modified` row (+18,814 rows,
  0.05% of those hours) and the 69 hours of 2026 saved before the fix (+1,513 rows).
- **Speed fix (same commit):** keep a full 10,000-row page and only ask for the older remainder instead of discarding
  it and halving the window: 7 requests instead of 15 for a busy 2026 hour, identical rows.
- **Bug found:** the indexer's `end_time` is inclusive at exactly `xx:00:00.000` → 66 rows saved in two hour files.
  **Fix:** trim to `[start, end)` (`9e5c744`); 6 files cleaned.
- **Result:** every hour of both periods for both pools, 0 truncated hours, 0 duplicates; Binance 184 days × 3 symbols,
  0 missing; 6.3 GB. 7 empty DEEP_USDC trade hours are genuine (indexer confirms 0 trades).

## 2026-10-07 — Order book reconstruction
- **Ran:** `flowcast book` on five test days, then all 368 pool-days (`cc0e191`, output `data/processed/book/`,
  log `book_rebuild.log`).
- **Learned:** fills appear only in trades; `Canceled` reports the open size cancelled (100% match with our replay);
  Sui stamps a whole checkpoint (~0.25 s) with one millisecond, so the order within a millisecond is unknown — the
  "fill at the best price" check was unfair, replaced by **trade-throughs** (a fill skipping an untouched better order).
- **Bug found:** `Modified.remaining_quantity` is the order's new **total** size, including what already filled.
  Treating it as the open size left phantom orders (2025-10-30: book crossed 9.5% of seconds, 314 trade-throughs).
  **Fix:** track total and filled per order; open = total − filled.
- **Result:** all 368 days: 0 trade-throughs, 100% cancel match, 0 crossed seconds; ≤ 0.84% fills on orders placed
  before the warm-up. 1.3 GB.

## 2026-10-08 — Labels and baselines
- **Ran:** `flowcast labels`; `flowcast baselines` (`319155b`).
- **Labels:** mid return over 1/10/60 s in bps; flat band = 1/3 quantile of |return| on Aug only (SUI: 0 / 0.95 / 3.43
  bps; DEEP: 0 / 0 / 1.20). At 1 s FLAT is 55–85% (most seconds don't move at all).
- **Baselines (Sep macro F1):** SUI best rule 0.361 / 0.357 / 0.359 (Binance lead-lag rule best); DEEP 0.424
  (imbalance) / 0.396 / 0.393. The imbalance rule fits its depth (1/2/3/5 levels) on Aug, which made it stronger.

## 2026-10-08 — Features
- **Ran:** `flowcast features` (`dfad203`, resumable; output `data/processed/features/`, log `features_build.log`).
- **Result:** 37 backward-looking features per second (book, price, order flow, trades, market maker, time, Binance ≥ 1 s
  old); a test perturbs the future and checks no past row changes. Crashed once on an empty DEEP trade hour → fixed.
- **Noticed:** DEEP_USDC in 2026 often has **no order events for 30+ s** (thin regime).

## 2026-10-09 — EDA (notebook 01, by hand)
- **Decisions:** macro F1 not accuracy; drop Aug 1, `modify_share_30s` (constant in Aug), `cancel_imb_30s`,
  `mm_active_10s` (redundant), `dow`; `log1p` + clip + scale for linear models; split by time.

## 2026-10-09 — Logistic regression, runs 1 and 2
- **Ran:** `flowcast train logreg` twice (≈ `d52a73a`) → `runs/20261009-180800_logreg/`, `runs/20261009-182404_logreg/`.
- **Run 1** (C ∈ {0.01, 0.1, 1}, balanced): beats the rules on SUI by ~+0.09 at 1/10 s; C = 0.01 won everywhere (grid
  edge); DEEP 1 s loses to the rule.
- **Run 2** (C ∈ {1e-4 … 0.1} × class weights): scores moved ≤ ±0.005 → linear ceiling. Class weights were not the
  DEEP 1 s problem (my guess was wrong).
- **Decided:** stop tuning logistic regression; try gradient boosting.

## 2026-10-09 — LightGBM
- **Setup:** `brew install libomp` (LightGBM needs OpenMP on macOS). `shap` dropped (no Python 3.13 build of llvmlite);
  LightGBM computes SHAP values itself.
- **Ran:** `flowcast train lgbm` (`a36386a`, clean) → `runs/20261009-191600_lgbm/` (~85 min; I had estimated 25).
- **Result (Sep, + Binance):** SUI 0.479 / 0.463 / 0.404; DEEP 0.502 / 0.486 / 0.463. Beats logistic regression
  everywhere; **fixes DEEP 1 s** (thresholds). Grid edges again → early stopping next time.

## 2026-10-09 — Model comparison, SHAP, costs (notebook 03, then numbered 02)
- **SHAP:** `imb_1` is a staircase (no effect until ~+0.55, then a jump); `vol_60s` saturates near 1; `bn_gap_bps`
  S-shaped — the nonlinearity logistic regression can't express.
- **Directional accuracy** on UP/DOWN calls that moved: ~59% at 10 s, ~62% at 1 s.
- **Correction:** I first said taker fees were ~7 bps; the fee is paid in **DEEP**, ~1 bp per side (makers 0).

## 2026-10-09 — Evaluation: bootstrap and lead-lag
- **Ran:** `flowcast evaluate` (`ff89a6d`) → `runs/20261009-232053_eval/`.
- **Result:** 32 of 33 paired comparisons **real** (interval excludes 0), including "Binance adds +0.005" at SUI 10 s
  [+0.004, +0.006]; logistic regression on DEEP 1 s is reliably **worse** than the rule.
- **Lead-lag:** Binance leads SUI_USDC by ~0–2 s (corr 0.29 same second, 0.24 at 1 s, 0.09 at 2 s); DeepBook rarely
  leads. DEEP shows ~no 1 s correlation with the DEEP perp.
- **Correction:** notebook 02 had called the top 1% most confident calls "roughly break-even" using the **median**
  spread; expected profit needs the **average** cost, which is ~2.5× higher there (confident calls cluster in wide
  spreads). **No confidence level is profitable as a taker** (best: SUI 60 s top 1%, −2.0 bps [−3.7, −0.7]).
  Fixed in `21a4215`. The maker angle is untested.

## 2026-10-10 — Tracking our steps
- **Added:** this log; notebooks `00_data_pipeline`, `02_training`, `04_evaluation`; `02_models` renamed to
  `03_models`; run results (`config.json`, `results.csv`, evaluation tables) now tracked in git (models stay local:
  `runs/*/models/` is ignored); every training run now also saves `cv_results.csv` (every setting × fold score).
- **Next:** walk-forward evaluation, then the October test (once).

## 2026-11-10 — Walk-forward over September weeks
- **Ran:** `flowcast walkforward` (`d940b95`, clean) → `runs/20261110-173006_walkforward/` (~90 min, log
  `walkforward.log`). 4 Sep weeks, expanding train from Aug 2, purge = horizon; LightGBM retrained per week with the
  Aug-tuned settings vs the fixed Aug model vs the best rule (refit per week, chosen on train).
- **Sanity check:** week 1 retrained = fixed (same data, same settings), as expected.
- **Result:** model beats the rule in 23 of 24 weeks; all 12 pooled model-vs-rule comparisons real. Pooled lift:
  SUI 1s +0.166 (retrained) / +0.118 (fixed); SUI 10s +0.070 / +0.088; SUI 60s +0.025 / +0.038; DEEP +0.07–0.09 either.
  Only loss: SUI 60s week 2 (−0.009). The single-split +0.105 at SUI 10s was optimistic; walk-forward says +0.07–0.09.
- **Surprise:** retraining helps SUI 1s (+0.048 vs fixed) but reliably hurts SUI 10s/60s (−0.018 / −0.014); no effect
  on DEEP. At SUI 10s the refit rule switches from lead-lag to imbalance and improves (0.33 → 0.39) as Sep enters the
  training window: SUI changed during September. Not explained yet (class-weight guess undercut: DEEP has none either
  and doesn't degrade). Check: predicted class mix per week.
- **Week 3** (market maker absent until Sep 15 09:00): no dip anywhere.
- **Next:** October test, once. Choose which model per pool/horizon from this run before looking.
