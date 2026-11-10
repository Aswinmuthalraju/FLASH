# Stage D5 — Validation-only evaluation: execution report

**Status: COMPLETE.** Exit code 0. All gates G1–G8 passed. Only validation rows were predicted or scored.
**Test evaluation, calibration and D6 have NOT started. The test split remains locked and unexamined.**

All numbers below were read from the D5 output files (`d5_validation_metrics.json`, `d5_run_metadata.json`, `d5_val_probabilities.npz`), the run log, or the post-run verification commands. Where a statement is an interpretation or estimate rather than a measured fact, it says so.

---

## 0. Interpretation restrictions (read first)

1. **Probabilities are uncalibrated.** The D4 identity calibrator was used (temperature, map [1.0]). No calibration, threshold tuning or gate tuning was performed.
2. **Validation was already used** for PCA `n_components` selection (= 4) and for LightGBM early stopping (model D, best_iter 174 / 121 / 107).
3. **The D3 baselines were also evaluated on this same validation split.**
4. Therefore **D5 is not an independent generalisation assessment.** The results are optimistic, especially for D, whose stopping point was chosen on these rows.
5. **D differs from A in several ways at once:**
   - model family (LGBM + logistic blend vs. logistic regression)
   - feature count (42 vs. 8)
   - feature composition (news, regime and PCA features vs. price only)
   - blending (0.75·LGBM + 0.25·logit)
6. **A D-over-A improvement cannot establish that news has value.**
7. Isolating the news contribution would require a **separately authorised ablation.**
8. **The test split remains locked and unexamined.**

---

## 1. Run summary

| Item | Value |
|---|---|
| Command | `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe reports/data_prep/stage_d5_validation_evaluation/d5_validation_evaluation.py` |
| Script SHA-256 | `0cf317ef82aaf92a3d13dee99d9d23d9687588d10029b57a5ae92d1ecaeec0db` (unchanged; equals the reviewed version) |
| Start (wrapper) | 2026-11-09T22:10:11+05:30 |
| Start (script, `started_utc`) | 2026-11-09T16:40:14Z = 22:10:14 IST |
| Completion | 2026-11-10T00:07:19+05:30 |
| Runtime | 7,028 s wall (wrapper); 7,024.2 s self-reported. The pre-run estimate was ~3 h, so it ran faster than estimated. |
| Exit code | **0** |
| `status` (metrics and metadata) | `complete` / `complete` |
| `d5_gate_failure.json` | not created |
| Runtime warnings captured | 0 |
| Versions | Python 3.11.18, LightGBM 4.7.0, scikit-learn 1.9.1, NumPy 2.4.6 |
| Bootstrap | B = 1000, seed 0, α = 0.05, block = trading day (27 days) |

### Time per phase (from the log)

| Phase | Time |
|---|---|
| G2 / G3 checks | 3 s |
| `prepare()` + G4 | ~3 min 50 s |
| G5–G7, all three horizons | 3 s |
| G8 | 9 s |
| npz written | 22:14:17 |
| Bootstrap plus metrics, per model/horizon | 1382 / 1320 / 842 / 1218 / 862 / 1158 s (h5 A, h5 D, h15 A, h15 D, h30 A, h30 D) |

---

## 2. Gate results

| Gate | Result | Evidence |
|---|---|---|
| G1 — no test unlock; no existing outputs | **PASS** | `FLASH_TEST_UNLOCK` unset (also rechecked after the run); all outputs were absent at start |
| G2 — D4 artifact hashes (start **and** end) | **PASS** | All 5 files in `models_real/stage_d4/` match the D4 inventory, both before and after the run |
| G3 — bundle structure | **PASS** | See details below |
| G3b — feature list for A | **PASS** | 8 features: ret1, ret5, ret15, mean20, var20, vol30, range, close_pos |
| G4 — PCA/HMM reproducibility | **PASS** | See details below |
| G5 — validation rows | **PASS** | See §3 |
| G6 — finiteness and probability validity | **PASS** | See details below |
| G7 — reproduction of D3 baselines | **PASS** | See details below |
| G8 — streaming vs. batch cross-check (**blocking**) | **PASS** | See details below |
| FinBERT cache unchanged at end | **PASS** | `finbert_cache_unchanged_end = true`; sha `43447937…f1373` |

**G3 details:**
- cfg equals `real.yaml`; 42 columns.
- Feature names match D4.
- The LightGBM model string equals the `.txt` file.
- Booster iterations equal best_iter (174 / 121 / 107).
- Logit W is 3×42 and b is 3; blend weight w = 0.25 = cfg.
- A has W 3×8.
- Calibrators are identity.

**G4 details:**
- pca_mean, pca_scale, P and c, plus HMM startprob, transmat, means and covars: max |diff| = **0.0** for all (tolerance rtol 1e-9, atol 1e-12).
- clip, col_names, groups and HMM names are equal.

**G6 details:**
- 0 non-finite feature rows in A or D, at every horizon.
- P_A, P_lgbm, P_logit and P_D all valid: no NaN, inf, negative or >1 values; max |row sum − 1| = 2.2e-16.

**G7 details:**
- majority, prior and momentum reproduce D3 exactly: all abs diffs **0.0**.
- per-class, confusion matrix and reliability results are equal at every horizon.

**G8 details:**
- Replay of day_idx 82 (2024-10-15): 3,750 bars, 5,544 events available.
- Per-horizon results:

| Mode | Horizon | Rows compared | Coverage | Max err | Median err | Frac > 1e-6 |
|---|---|---|---|---|---|---|
| A | h5 | 3,400 | 1.0 | 2.2e-16 | 2.8e-17 | 0 |
| A | h15 | 3,300 | 1.0 | 2.2e-16 | 2.8e-17 | 0 |
| A | h30 | 3,150 | 1.0 | 2.2e-16 | 5.6e-17 | 0 |
| D | h5 | 3,400 | 1.0 | 3.3e-16 | 2.8e-17 | 0 |
| D | h15 | 3,300 | 1.0 | 2.2e-16 | 2.8e-17 | 0 |
| D | h30 | 3,150 | 1.0 | 2.2e-16 | 2.8e-17 | 0 |

- No non-finite live rows.
- The thresholds were D median < 1e-9, D fraction > 1e-6 < 0.002, A max < 1e-8, and coverage ≥ 0.9. All were met by many orders of magnitude.

---

## 3. Validation rows and class distributions (G5, verified)

| h | n | DOWN | NEUTRAL | UP | Days | day_idx range | all split = val | all valid_x | labels ≥ 0 |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 91,800 | 17,581 | 56,517 | 17,702 | 27 | 82–108 | ✔ | ✔ | ✔ |
| 15 | 89,100 | 19,697 | 49,554 | 19,849 | 27 | 82–108 | ✔ | ✔ | ✔ |
| 30 | 85,050 | 20,006 | 45,035 | 20,009 | 27 | 82–108 | ✔ | ✔ | ✔ |

All values equal the expected counts and supports exactly.

Training priors used by the baselines, h5, fit on 278,800 train rows: [0.1881, 0.6248, 0.1871].

---

## 4. Results by model and horizon (point estimates on validation)

All six model/horizon combinations are present (A and D × h5, h15, h30), alongside the three baselines.

| h | Model | Accuracy | Macro-F1 | ROC-AUC (OvR) | Brier | Log loss | ECE |
|---|---|---|---|---|---|---|---|
| 5 | majority | 0.6157 | 0.2540 | 0.5000 | 0.7687 | 10.6199 | 0.3843 |
| 5 | prior | 0.6157 | 0.2540 | 0.5000 | 0.5472 | 0.9327 | 0.0092 |
| 5 | momentum | 0.6157 | 0.2540 | 0.5187 | 0.5466 | 0.9319 | 0.0087 |
| 5 | **A** | 0.6156 | 0.2554 | 0.5798 | 0.5382 | 0.9198 | 0.0159 |
| 5 | **D** | 0.6156 | 0.2541 | **0.5886** | **0.5356** | **0.9162** | 0.0158 |
| 15 | majority | 0.5562 | 0.2383 | 0.5000 | 0.8877 | 12.2637 | 0.4438 |
| 15 | prior | 0.5562 | 0.2383 | 0.5000 | 0.5926 | 0.9951 | 0.0171 |
| 15 | momentum | 0.5562 | 0.2383 | 0.5079 | 0.5925 | 0.9949 | 0.0168 |
| 15 | **A** | 0.5561 | 0.2389 | 0.5867 | 0.5807 | 0.9786 | 0.0290 |
| 15 | **D** | 0.5562 | 0.2394 | **0.5916** | **0.5787** | **0.9762** | 0.0283 |
| 30 | majority | 0.5295 | 0.2308 | 0.5000 | 0.9410 | 13.0001 | 0.4705 |
| 30 | prior | 0.5295 | 0.2308 | 0.5000 | 0.6092 | 1.0178 | 0.0125 |
| 30 | momentum | 0.5295 | 0.2308 | 0.5118 | 0.6089 | 1.0176 | 0.0122 |
| 30 | **A** | 0.5296 | 0.2324 | **0.6023** | 0.5918 | 0.9942 | 0.0289 |
| 30 | **D** | **0.5324** | **0.2477** | 0.6018 | **0.5904** | **0.9928** | **0.0259** |

### 4.1 Per-class precision / recall / F1

| h | Model | DOWN P / R / F1 | NEUTRAL P / R / F1 | UP P / R / F1 |
|---|---|---|---|---|
| 5 | A | 0.333 / 0.0006 / 0.0012 | 0.616 / 0.9992 / 0.7620 | 0.351 / 0.0015 / 0.0030 |
| 5 | D | 0.000 / 0.0000 / 0.0000 | 0.616 / 1.0000 / 0.7621 | 0.083 / 0.0001 / 0.0001 |
| 15 | A | 0.209 / 0.0005 / 0.0009 | 0.556 / 0.9994 / 0.7148 | 0.367 / 0.0006 / 0.0011 |
| 15 | D | 0.339 / 0.0011 / 0.0021 | 0.557 / 0.9994 / 0.7150 | 0.190 / 0.0006 / 0.0011 |
| 30 | A | 0.274 / 0.0018 / 0.0037 | 0.530 / 0.9991 / 0.6926 | 0.444 / 0.0004 / 0.0008 |
| 30 | D | 0.388 / 0.0210 / 0.0399 | 0.535 / 0.9945 / 0.6959 | 0.270 / 0.0037 / 0.0074 |

### 4.2 Confusion matrices

Rows are true labels, columns are predictions, both ordered DOWN, NEUTRAL, UP.

| h | A | D |
|---|---|---|
| 5 | [[11, 17553, 17], [14, 56470, 33], [8, 17667, 27]] | [[0, 17572, 9], [0, 56515, 2], [0, 17701, 1]] |
| 15 | [[9, 19679, 9], [18, 49526, 10], [16, 19822, 11]] | [[21, 19652, 24], [5, 49526, 23], [36, 19802, 11]] |
| 30 | [[37, 19964, 5], [35, 44995, 5], [63, 19938, 8]] | [[421, 19449, 136], [180, 44788, 67], [483, 19451, 75]] |

### 4.3 Prediction distributions

| h | Model | argmax DOWN / NEUTRAL / UP | NEUTRAL share | Mean prob DOWN / NEUTRAL / UP |
|---|---|---|---|---|
| 5 | A | 33 / 91,690 / 77 | 99.88 % | 0.1846 / 0.6313 / 0.1841 |
| 5 | D | 0 / 91,788 / 12 | 99.99 % | 0.1840 / 0.6310 / 0.1850 |
| 15 | A | 43 / 89,027 / 30 | 99.92 % | 0.2074 / 0.5850 / 0.2076 |
| 15 | D | 62 / 88,980 / 58 | 99.87 % | 0.2069 / 0.5839 / 0.2092 |
| 30 | A | 135 / 84,897 / 18 | 99.82 % | 0.2202 / 0.5584 / 0.2213 |
| 30 | D | 1,084 / 83,688 / 278 | 98.40 % | 0.2213 / 0.5572 / 0.2215 |

Reliability bins for each class are stored in `d5_validation_metrics.json` under `horizons.<h>.models.<m>.reliability`.

### 4.4 95 % day-block bootstrap CIs for each model (B = 1000)

| h | Model | Accuracy | Macro-F1 | AUC | Brier | Log loss | ECE |
|---|---|---|---|---|---|---|---|
| 5 | A | [0.6111, 0.6205] | [0.2539, 0.2572] | [0.5745, 0.5854] | [0.5337, 0.5426] | [0.9134, 0.9260] | [0.0093, 0.0231] |
| 5 | D | [0.6112, 0.6205] | [0.2529, 0.2553] | [0.5813, 0.5954] | [0.5311, 0.5401] | [0.9099, 0.9226] | [0.0101, 0.0219] |
| 15 | A | [0.5467, 0.5654] | [0.2363, 0.2416] | [0.5782, 0.5955] | [0.5731, 0.5886] | [0.9681, 0.9896] | [0.0190, 0.0407] |
| 15 | D | [0.5469, 0.5654] | [0.2366, 0.2424] | [0.5817, 0.6014] | [0.5716, 0.5857] | [0.9661, 0.9861] | [0.0193, 0.0393] |
| 30 | A | [0.5165, 0.5432] | [0.2285, 0.2362] | [0.5925, 0.6115] | [0.5829, 0.6006] | [0.9813, 1.0066] | [0.0193, 0.0437] |
| 30 | D | [0.5198, 0.5456] | [0.2419, 0.2542] | [0.5891, 0.6138] | [0.5818, 0.5988] | [0.9809, 1.0043] | [0.0146, 0.0402] |

---

## 5. Comparison against majority, prior and momentum (paired day-block bootstrap)

How to read this section:
- Differences are computed as **model − reference**. For Brier, log loss and ECE, negative is better. For AUC, F1 and accuracy, positive is better.
- Δ is the point difference and the bracket is the 95 % CI. A ✔ means the CI excludes 0.
- There are 21 paired comparisons × 6 metrics, with **no multiplicity correction** (the comparisons are exploratory).

### 5.1 Against **prior** (the strongest probabilistic baseline in D3, together with momentum)

| h | Model | ΔBrier | ΔLog loss | ΔAUC | ΔMacro-F1 CI | ΔAccuracy CI | ΔECE CI |
|---|---|---|---|---|---|---|---|
| 5 | A | −0.0090 [−0.0103, −0.0076] ✔ | −0.0129 [−0.0150, −0.0109] ✔ | +0.0798 [+0.0745, +0.0854] ✔ | [+0.0007, +0.0023] | [−0.0003, +0.0001] | [+0.0029, +0.0110] (worse) |
| 5 | D | −0.0116 [−0.0133, −0.0101] ✔ | −0.0165 [−0.0189, −0.0141] ✔ | +0.0886 [+0.0813, +0.0954] ✔ | [+0.0000, +0.0001] | [−0.0000, +0.0000] | [+0.0036, +0.0102] (worse) |
| 15 | A | −0.0119 [−0.0145, −0.0092] ✔ | −0.0165 [−0.0202, −0.0126] ✔ | +0.0867 [+0.0782, +0.0955] ✔ | [+0.0003, +0.0010] | [−0.0002, +0.0000] | [+0.0065, +0.0184] (worse) |
| 15 | D | −0.0139 [−0.0165, −0.0113] ✔ | −0.0189 [−0.0226, −0.0153] ✔ | +0.0916 [+0.0817, +0.1014] ✔ | [+0.0006, +0.0018] | [−0.0002, +0.0003] | [+0.0078, +0.0164] (worse) |
| 30 | A | −0.0173 [−0.0205, −0.0141] ✔ | −0.0236 [−0.0284, −0.0192] ✔ | +0.1023 [+0.0925, +0.1115] ✔ | [+0.0009, +0.0024] | [−0.0001, +0.0003] | [+0.0099, +0.0249] (worse) |
| 30 | D | −0.0187 [−0.0224, −0.0149] ✔ | −0.0251 [−0.0302, −0.0196] ✔ | +0.1018 [+0.0891, +0.1138] ✔ | [+0.0125, +0.0219] | [+0.0010, +0.0052] | [+0.0082, +0.0208] (worse) |

### 5.2 Against **momentum**

| h | Model | ΔBrier | ΔLog loss | ΔAUC | ΔMacro-F1 CI | ΔAccuracy CI | ΔECE CI |
|---|---|---|---|---|---|---|---|
| 5 | A | −0.0084 [−0.0097, −0.0070] ✔ | −0.0120 [−0.0140, −0.0099] ✔ | +0.0611 [+0.0542, +0.0675] ✔ | [+0.0007, +0.0023] | [−0.0003, +0.0001] | [+0.0035, +0.0115] (worse) |
| 5 | D | −0.0110 [−0.0127, −0.0093] ✔ | −0.0156 [−0.0180, −0.0131] ✔ | +0.0699 [+0.0610, +0.0777] ✔ | [+0.0000, +0.0001] | [−0.0000, +0.0000] | [+0.0042, +0.0107] (worse) |
| 15 | A | −0.0118 [−0.0144, −0.0090] ✔ | −0.0163 [−0.0202, −0.0123] ✔ | +0.0788 [+0.0681, +0.0894] ✔ | [+0.0003, +0.0010] | [−0.0002, +0.0000] | [+0.0068, +0.0186] (worse) |
| 15 | D | −0.0137 [−0.0164, −0.0111] ✔ | −0.0187 [−0.0225, −0.0150] ✔ | +0.0837 [+0.0712, +0.0957] ✔ | [+0.0006, +0.0018] | [−0.0002, +0.0003] | [+0.0082, +0.0166] (worse) |
| 30 | A | −0.0171 [−0.0203, −0.0139] ✔ | −0.0234 [−0.0281, −0.0187] ✔ | +0.0905 [+0.0801, +0.1021] ✔ | [+0.0009, +0.0024] | [−0.0001, +0.0003] | [+0.0102, +0.0252] (worse) |
| 30 | D | −0.0185 [−0.0222, −0.0145] ✔ | −0.0248 [−0.0301, −0.0193] ✔ | +0.0900 [+0.0767, +0.1038] ✔ | [+0.0125, +0.0219] | [+0.0010, +0.0052] | [+0.0085, +0.0210] (worse) |

### 5.3 Against **majority**

The majority baseline outputs one-hot probabilities, so its Brier and log loss are degenerate. Beating it is trivial.

| h | Model | ΔBrier | ΔLog loss | ΔAUC |
|---|---|---|---|---|
| 5 | A | −0.2304 [−0.2358, −0.2248] | −9.700 [−9.817, −9.571] | +0.0798 [+0.0745, +0.0854] |
| 5 | D | −0.2331 [−0.2383, −0.2275] | −9.704 [−9.822, −9.574] | +0.0886 [+0.0813, +0.0954] |
| 15 | A | −0.3070 [−0.3179, −0.2954] | −11.285 [−11.533, −11.040] | +0.0867 [+0.0782, +0.0955] |
| 15 | D | −0.3090 [−0.3211, −0.2970] | −11.288 [−11.536, −11.042] | +0.0916 [+0.0817, +0.1014] |
| 30 | A | −0.3491 [−0.3674, −0.3304] | −12.006 [−12.358, −11.639] | +0.1023 [+0.0925, +0.1115] |
| 30 | D | −0.3505 [−0.3702, −0.3302] | −12.007 [−12.356, −11.637] | +0.1018 [+0.0891, +0.1138] |

The accuracy and macro-F1 CIs against majority are identical to those against prior and momentum, because all three baselines predict all NEUTRAL.

### 5.4 D vs A (exploratory; see restrictions 5–7)

| h | ΔBrier | ΔLog loss | ΔAUC | ΔMacro-F1 CI | ΔAccuracy CI | ΔECE CI |
|---|---|---|---|---|---|---|
| 5 | −0.0027 [−0.0035, −0.0018] ✔ | −0.0036 [−0.0050, −0.0022] ✔ | +0.0088 [+0.0057, +0.0118] ✔ | [−0.0022, −0.0006] ✔ (D worse) | [−0.0001, +0.0003] | [−0.0023, +0.0034] |
| 15 | −0.0020 [−0.0035, −0.0004] ✔ | −0.0024 [−0.0048, +0.0001] | +0.0049 [−0.0010, +0.0103] | [−0.0002, +0.0013] | [−0.0001, +0.0004] | [−0.0041, +0.0046] |
| 30 | −0.0014 [−0.0036, +0.0009] | −0.0014 [−0.0048, +0.0022] | −0.0005 [−0.0086, +0.0076] | [+0.0114, +0.0196] ✔ | [+0.0010, +0.0050] ✔ | [−0.0083, +0.0029] |

---

## 6. Paired bootstrap findings (summary)

1. **Both models beat every baseline on proper scoring rules and ranking, at every horizon.**
   - This holds for Brier, log loss and AUC, and all 18 such CIs exclude 0.
   - The gains over prior and momentum are **small in absolute terms**: Brier −0.008 to −0.019 and log loss −0.012 to −0.025.
   - AUC rises from ~0.50–0.52 to ~0.58–0.60.
2. **Both models are worse calibrated than the prior and momentum baselines on ECE**, at every horizon (all 12 such CIs lie above 0). This is consistent with the probabilities being uncalibrated. ECE is still ≤ 0.029.
3. **Classification metrics barely move.**
   - Both models predict NEUTRAL for 98.4–99.99 % of rows, so accuracy and macro-F1 stay at the all-NEUTRAL level.
   - The one exception is **D at h30**: macro-F1 +0.017 and accuracy +0.003 vs. the baselines, both with CIs excluding 0. This comes from 1,084 DOWN calls (precision 0.388, recall 0.021).
4. **D vs A:**
   - At h5, D is better on Brier, log loss and AUC (CIs exclude 0) but has slightly lower macro-F1.
   - At h15 only the Brier gain excludes 0.
   - At h30 the proper-scoring and AUC differences are **not distinguishable from 0**. Only macro-F1 and accuracy favour D, through the DOWN calls.
   - Even where D is better, the improvement is about ¼ of A's gain over the baselines.
   - **None of this attributes value to news** (restrictions 5–7). D also had its early stopping chosen on these same rows (restriction 2), which favours D in this comparison.
5. There is no multiplicity correction. Of 126 CIs, a few marginal ones may be chance results. This is an interpretation.

---

## 7. Warnings, limitations and anomalies

| # | Type | Finding |
|---|---|---|
| 1 | Anomaly (naming; verified) | In `d5_val_probabilities.npz`, the `h*_timestamp_ns` arrays hold **microseconds**, not nanoseconds. `pd.DatetimeIndex(...).asi8` returns the native datetime64[us] resolution of this pandas build. Decoding as µs gives 2024-10-15 09:45 IST through 2024-11-25 15:24 IST, which is the first and last validation day as expected. Values are correct; only the key suffix is wrong. Not fixed, because the script must not be modified. |
| 2 | Limitation | The results are **not** an independent generalisation estimate (§0, items 2–4). Optimism is most likely for D. |
| 3 | Limitation | The probabilities are uncalibrated, and the ECE comparisons show the baselines are better calibrated. |
| 4 | Observation | Both models are almost pure NEUTRAL predictors under argmax. Macro-F1 ≈ 0.24–0.25 does not reflect useful directional classification, except the modest DOWN signal from D at h30. |
| 5 | Observation | Majority-baseline log loss (10.6–13.0) comes from one-hot probabilities clipped at eps (reproduced from D3 exactly). Comparisons against majority are uninformative for probabilistic metrics. |
| 6 | Limitation | 21 comparisons × 6 metrics with no multiplicity correction. |
| 7 | Limitation | 27 day-blocks is a small number of bootstrap clusters, so the CIs may be somewhat narrow. This is an estimate, not measured. |
| 8 | Limitation | Paired point differences were recorded only for Brier, log loss and AUC. For accuracy, macro-F1 and ECE, the point differences can be derived from the §4 table. |
| 9 | Scope note (accepted) | `prepare()` processes the full chronological dataset in memory, including test-period rows, as recorded in `data_scope`. Only validation rows were predicted or scored. |
| 10 | Process note | My first `sha256sum -c` attempt failed only because it ran from inside the D5 directory, while the file lists repo-relative paths. Run from the repo root, all 3 hashes verify. The file uses LF line endings. |
| 11 | Process note | Runtime was 1 h 57 min vs. the ~3 h pre-run estimate. |
| 12 | Carried erratum | The D4 report's "109 trading days" wording means train+val. The dataset has 136 days (see D5 preflight). |
| — | Runtime warnings | None. `warnings = []`, with 0 in the log. |

---

## 8. Test-isolation confirmation (verified)

- All npz arrays have `day_idx` in [82, 108] at every horizon (validation = 82–108; test starts at day 109).
- The maximum timestamp is 2024-11-25 15:24 IST, the last validation day.
- Row counts per horizon equal the expected validation counts exactly. G5 confirms `all_split_val = true`.
- The `_NoTestDataset.rows` and `BL.rows` runtime guards raise on any split other than train or val. No guard fired: exit 0 and no failure file.
- The G8 replay used validation day 82 only.
- `FLASH_TEST_UNLOCK` was unset before and after the run.
- **No test rows were predicted, calibrated or scored.**

---

## 9. Integrity rechecks after the run (verified)

| Check | Result |
|---|---|
| D4 artifacts: `d4_artifacts.sha256`, CR stripped | all OK |
| D4 artifacts: G2 at start and end of the run | all OK |
| D4 report files: driver `3d496a72…`, log `28247485…`, metadata `d14eddf8…`, inventory `2563cd14…`, report `4ff8beea…` | unchanged |
| D3 files: report `3fcc3fae…`, json `f74ad79f…`, log `6ded7eb0…` | unchanged |
| 40-file D3-era data checklist | 40 / 40 OK |
| 24-file D2 checklist | 24 / 24 OK |
| `models/` (original bundle and event memory) | 2 / 2 OK |
| News source `money_control_news.parquet` | `7213bcd4a42d8ea3ab95798461524bb0ca6479085ac1be2afe26e524fd42e65d` ✔ |
| FinBERT cache | unchanged (metadata flag) |
| D5 script / preflight / review | `0cf317ef…` / `e849e585…` / `cd7f3cf6…` — unchanged |

### Git status

| Item | Value |
|---|---|
| Branch | `data-pipeline` |
| HEAD | `ffa1cda72861d22ac9f2ec71e029858e582cc9d3` |
| `backup/pre-data-pipeline-20261109` | `ffa1cda…` (same) |
| Tracked modifications | `.gitignore` only (+15 lines; the pre-existing patch) |
| Staged | 0 |
| Untracked, collapsed view | 12 entries |
| Untracked, `-uall` | 60 files |
| New since launch | the five D5 outputs listed in §10, plus this report |
| `models_real/` | ignored (`.gitignore:50`) |

Nothing was committed, staged or pushed.

---

## 10. Output inventory

| File | Size (B) | SHA-256 | State |
|---|---|---|---|
| `d5_validation_metrics.json` | 70,648 | `29f3c9aa9ee4a8e2ebc71aac5dab4d2ca562fbc7527630fe37333f589c79bcb8` | complete; hash verified |
| `d5_run_metadata.json` | 21,897 | `4511a185edf8e908aef6cb9ec887e1a50d4441e5b99b23ad18ec104e96115f63` | complete; hash verified |
| `d5_val_probabilities.npz` | 13,601,657 | `ef8704cb32709837d2e610d88cb54abbc9885158d2f6cbaf144ba604d35f40fb` | complete; hash verified; 30 arrays (row_index, y, day_idx, symbol, timestamp, P_A, P_D, P_majority, P_prior, P_momentum × 3 horizons) |
| `d5_outputs.sha256` | 418 | (manifest of the three files above; LF) | complete |
| `d5_validation_evaluation.log` | 1,711 | not hashed by the script | complete |
| `d5_gate_failure.json` | — | — | absent (expected on success) |
| `STAGE_D5_VALIDATION_EVALUATION_REPORT.md` | — | this file | — |

---

## 11. Stage boundary

- D5 (validation-only evaluation) is **complete**.
- The following have **NOT** been started:
  - test-split evaluation
  - test unlock
  - calibration
  - threshold or gate tuning
  - news ablation
  - benchmark, demo or Streamlit runs
  - **D6**
- No model was trained or retrained. No D1–D4 artifact or source, test, config or dependency file was modified.
- Further work requires explicit approval.
