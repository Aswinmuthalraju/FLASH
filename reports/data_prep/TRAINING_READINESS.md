# TRAINING_READINESS

**Verdict: NOT READY for real-data training.** The code path is ready and tested. The *data* has not been
obtained, so items 1–8 below cannot be answered with real numbers yet. No training, real or mock-as-real, was run.

## Status of the 12 items

| # | Item | Status |
|---|---|---|
| 1 | Dataset quality | ⛔ pending download; checks implemented (`verify`) and passing on mock |
| 2 | Dataset coverage | ⛔ pending; coverage-gap metrics implemented |
| 3 | Stock-wise sample counts | ⛔ pending; `per_symbol`, `per_symbol_valid_labelled_h15` |
| 4 | News coverage per stock | ⛔ pending; `stock_specific_per_symbol` |
| 5 | Train/val/test sizes | ⛔ pending; split is 60/20/20 by trading day (L1) |
| 6 | Feature availability | ✅ code: 42 features (14 market + 4 regime + 14 news + derived) build with 0 NaN/inf on mock; real ⛔ |
| 7 | Class distribution | ⛔ pending; reported per split × horizon |
| 8 | Leakage audit | ✅ 8 executable checks with positive controls; 1 real leak found and fixed in preprocessing; ⛔ not run on real data |
| 9 | Baselines | ✅ majority, class-prior, momentum + per-class P/R/F1, confusion matrix, AUC, Brier, log-loss, ECE, reliability bins, day-block bootstrap CIs (incl. CI of difference vs prior), test-split lock |
| 10 | Compute requirements | see below |
| 11 | Expected limitations | see below |
| 12 | Initial configuration | see below |

## Baseline results — SYNTHETIC data only (functional demonstration, not evidence)

Repo's committed synthetic bars, **validation split** (12 days), 200 day-block bootstrap resamples.

| h | baseline | macro-F1 | AUC | Brier [95% CI] | ECE |
|---|---|---|---|---|---|
| 15 | majority | 0.200 | 0.500 | 1.141 [1.075, 1.195] | 0.570 |
| 15 | prior | 0.200 | 0.500 | 0.659 [0.650, 0.668] | 0.028 |
| 15 | momentum | 0.302 | 0.541 | 0.655 [0.648, 0.662] | 0.032 |

Momentum − prior Brier, 95% CI: h5 [−0.005, −0.001], **h15 [−0.012, +0.004]**, h30 [−0.025, −0.001].
So at h = 15 the momentum baseline is *not* reliably better than the prior. That is the point of the CIs:
FLASH's gains over these baselines must be judged by the CI of the difference, not by point estimates.

Two caveats. The repo's published FLASH numbers are on **test**, so they must not be compared with these
validation numbers. And the momentum baseline is a train-count lookup table, not a trainable model. It
doesn't count against the five-model rule, but I've flagged it in case you see it differently.

## Recommended initial experiment (small; awaiting approval)

After `data` → `verify` → `audit` pass and the team reviews them:

1. `train-light`: first ~6 weeks of the news window, `configs/real.yaml` (model settings **unchanged** from
   `config/config.yaml`; only output paths differ).
2. Score A, B, C, D **on validation only** against prior and momentum, with day-block bootstrap CIs of the
   difference.
3. Ablations on validation: news off; history off; **price-move headlines removed** (LEAKAGE_AUDIT residual
   risk 1).
4. Go/no-go before any full run: does D beat the prior's Brier with a CI on the difference that excludes 0,
   and does the gain survive removing price-move headlines?

Scalable configuration: the full news window, same settings; re-tune only `num_leaves`, `min_child_samples`,
`k` and `gate_threshold`, on validation only. Test is scored once, after everything is frozen.

FLASH's training is classical, so the deep-learning vocabulary maps as follows:
- **Optimiser:** gradient boosting for LightGBM, EM for the GMM/HMM, L-BFGS for the logistic leg.
- **"Epochs":** boosting rounds, capped at 3,000, with early stopping after 50 rounds without validation
  improvement. Learning rate 0.03.
- **Regularisation:** leaves 7, min_child 300, λ 10, subsample 0.8, colsample 0.5, C 0.05.
- **Calibration:** temperature scaling fitted on validation.
- FinBERT stays frozen. No fine-tuning and no extra models without approval.

## Computational requirements

| Resource | Measured / estimated |
|---|---|
| Mock (20 days × 10 symbols): `train.py` with stub encoder | **measured** ≈ 17 s, CPU |
| Mock: test suite (58) | **measured** ≈ 23 s |
| Real window (≈80–140 trading days) | estimate: minutes per train run after the FinBERT cache exists |
| FinBERT pass | estimate 5–15 min CPU for up to ~25k headlines, once (cached by hash); GPU optional |
| RAM | estimate 8 GB comfortable |
| Disk | ~20 GB transient for the minute-shard read; <200 MB processed |
| **torch install** | plain `pip install torch` on Linux pulls the **CUDA build** (observed here: a single 519 MB cuDNN wheel, several GB in total; still downloading at the end of the session). For CPU-only use `pip install torch --index-url https://download.pytorch.org/whl/cpu`. `requirements.txt` was left unchanged |

## Expected limitations

- Small effective sample: a few months of bars; possibly few stock-specific headlines.
- News coverage gaps make "no news" ambiguous on many days.
- Linker false positives, and over-broad market-wide event rules on real headlines.
- Possible outcome leakage via edited headlines (ablation planned).
- Validation reuse (PCA, early stopping, calibration) makes validation metrics optimistic.
- Hyper-parameters were tuned on synthetic data. They're unbiased with respect to real test, but probably not optimal.
- 1-minute markets are close to unpredictable: expect AUCs near 0.5–0.6. Baseline CIs decide whether anything is real.

## Remaining blockers

1. **Hugging Face access** (DNS blocked in this sandbox). Run the `data` stage where huggingface.co resolves.
2. **torch** not available here, so the real FinBERT pass can't run. `audit` does not need it; `verify` does.
3. Team review of `verify` and `audit` output on real data.
4. Decisions:
   - fix the back-fill in FLASH's loader, or keep the preprocessing drop;
   - whether to use a news-coverage day filter;
   - approve the momentum baseline;
   - approve the price-move-headline ablation;
   - licence acceptance;
   - commit approval.
