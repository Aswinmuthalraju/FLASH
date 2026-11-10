# LEAKAGE_AUDIT

Tool: `preprocessing/leakage_audit.py` (stage `audit` for real data, `audit-mock` for mock).
Every check is **executed on data**. Tests in `tests/test_leakage_audit.py` show each check also **fails on a
planted leak** (positive controls), so a pass means something.

**Executed on:** HF-shaped mock data (all 8 pass) and small synthetic sets in the tests.
**Not yet executed on real data** (HF unreachable). Run `bash preprocessing/run_pipeline.sh audit` after download.

## Results

| # | Check | How it is tested | Mock result | Positive control |
|---|---|---|---|---|
| L1 | Splits: whole days, chronological, disjoint, 60/20/20 by day | every day maps to one split; split order is monotone in time | pass (12/4/4 days) | shuffled split → fails |
| L2 | Purge across split boundaries | count labelled rows whose `t+h` leaves the session | 0 for h = 5, 15, 30 | label planted at 15:28 → fails |
| L3 | Label construction | independent naive triple-barrier loop vs FLASH, 3,000 rows × 3 horizons | 9,000 checked, 0 mismatches | rotated labels → fails |
| L4 | Feature causality | recompute row *t* from the session prefix `[0..t]`, compare to full-day computation | max diff 0.0 | feature that looks 5 bars ahead → fails |
| L5 | Back-fill look-ahead | valid feature rows before a day's first real trade | 0 | late-open day through FLASH loader → 15 leaky rows found |
| L6 | News published before prediction | every attached headline's timestamp < bar timestamp | 0 violations / 66,868 rows | (FLASH's own `test_lookahead` covers the leaky-aligner control) |
| L7 | Event memory across split boundaries | corrupt every outcome not yet known at the val/test boundary → features of earlier events | max change 0.0 | — |
| L8 | PCA/HMM fitted on train only | corrupt all test-split regime features, refit | identical components, train posteriors diff 0.0 | — |

## Timestamp alignment

- Bars: HF timestamps are UTC; converted to IST (03:45Z = 09:15 IST, tested). The bar label is the bar **open**.
  A feature row at bar *t* uses the close of *t*, so the prediction is made at *t*'s close.
- News: minute-resolution IST. FLASH attaches a headline only to bars **strictly after** its timestamp. A
  headline stamped `10:00` (truly published 10:00:00–10:00:59) first affects the bar opening 10:01. So
  minute truncation can never move news earlier than it was knowable.
- Overnight and pre-open news ages from the open (FLASH design). That uses past information only.

## Label construction

Triple barrier per (symbol, session): UP/DOWN when `log(close[t+j]/close[t])` first crosses `±k·σ·√h`
(k = 1, σ = trailing 30-bar std at *t*), NEUTRAL otherwise. Rows whose window would pass 15:29 get no label.
σ uses past bars only (L4). Labels were independently re-derived (L3).

## Chronological splitting and purge

Split by trading **day**, 60/20/20, never shuffled. Because labels end at the session close and splits are
whole days, **no label window can cross a split boundary**. The purge is structural (L2 counts 0), so no
extra gap is needed. All features and HMM filtering reset at each session open.

Information that *does* flow across a boundary, all of it past-only and allowed:
- news decay carrying from the last day of one split into the next;
- event-memory outcomes that were already known (L7);
- val-trained components: PCA component count, early stopping, calibration.

The last item makes **validation metrics optimistic**, which is a known FLASH limitation. Test stays untouched.

## Leakage risks found and resolutions

| Risk | Status |
|---|---|
| **Back-fill look-ahead in `regularize_bars`** (`close.ffill().bfill()`). If a stock-day's first trade is after 09:45, rows 09:45 → first trade are "valid" but contain the first trade's (future) close. Shown concretely in `test_backfill_leak_exists_in_flash_loader_and_is_caught`. | **Resolved in preprocessing**: such symbol-days are dropped and logged. FLASH's loader is unchanged. A first trade exactly at 09:45 is safe (tested). The team may prefer fixing the loader itself; that edit is not made without approval. |
| Article ids used to infer dates | **Removed** (feature deleted); undated rows kept aside |
| Test set touched during development | **Locked**: `evaluation/run_baselines.py --split test` needs `--unlock-test` **and** `FLASH_TEST_UNLOCK=I_HAVE_APPROVAL` (tested) |
| Baselines using future data | Fitted on train rows only (corrupting val/test changes nothing, tested); momentum uses `close[t]`, `close[t−h]` of the same session and `vol30[t]` (future-perturbation test) |

## Residual risks this data cannot rule out (need an ablation or a decision)

1. **Edited headlines.** Scraped titles are the *final* versions. Moneycontrol often updates live-market
   titles ("rises 4%" → "surges 10%") while the timestamp may be the original publication. That would leak
   the outcome into the text. Recommended ablation: re-score D on validation after removing headlines that
   match price-move patterns (`shares (rise|fall|jump|slide|surge)… \d+%`). If most of D's gain disappears,
   the news signal is not trustworthy.
2. **Published vs updated time.** If the timestamp is the *update* time, news arrives late. That is
   conservative (no leakage) but understates the effect.
3. **Contemporaneous market return (`mret`).** This is the equal-weight average of the 10 stocks at the same
   minute. It's known at the bar close, so it's legitimate. But live inference needs all 10 feeds before
   predicting, which is a latency consideration rather than leakage.
