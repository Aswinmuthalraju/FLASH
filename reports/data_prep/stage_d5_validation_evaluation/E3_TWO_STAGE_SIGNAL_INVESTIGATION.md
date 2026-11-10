# E3 — Two-stage predictive signal investigation (diagnostic only)

**Status: COMPLETE.** The analysis ran as inline read-only Python from the D5 validation arrays only. Exit code 0.

**Not done:**
- No model was loaded, trained or calibrated.
- No threshold was tuned. No features or labels were rebuilt. `prepare()` was not called. No raw market data was read.
- No file other than this report was created. No test data was accessed.

**Tags:** **[M]** measured · **[I]** interpretation · **[H]** hypothesis.

---

## 1. Executive summary

1. **Movement prediction is clearly stronger than direction prediction.** [M]
   - Movement ROC-AUC is **0.594–0.633** across the six model/horizon combinations.
   - Conditional UP-vs-DOWN ROC-AUC is **0.491–0.532**.
   - The paired bootstrap gap (movement AUC − direction AUC) excludes 0 in all six combinations: lower CI bounds +0.050 to +0.114.
   - Movement AUC beats direction AUC on 25–27 of 27 days.
2. **Movement signal is consistent.** [M] Movement AUC is above 0.5 on **27 / 27 days** for every model and horizon. Leave-one-day-out changes the pooled movement AUC by at most 0.008.
3. **Direction is essentially chance level.** [M]
   - Model A: direction AUC 0.526–0.532, with 95 % CIs just above 0.5 (lower bounds 0.504–0.523).
   - Model D: 0.521 / 0.506 / **0.491** at h5 / h15 / h30. The h15 and h30 CIs include 0.5.
   - Conditional directional Brier is 0.2494–0.2516. A constant 0.5 forecast scores exactly 0.25.
   - Daily directional AUC exceeds 0.5 on only 10–24 of 27 days.
4. **A vs D, paired day-block bootstrap** [M]:
   - **Movement:** D is better than A. ROC-AUC +0.012 / +0.009 / +0.005; the CI excludes 0 at h5 and h15 but not h30. Movement Brier is lower for D at all horizons; at h30 the CI's upper bound is only −0.00006.
   - **Direction:** D is **worse** than A at every horizon. AUC −0.010 / −0.024 / −0.035 and directional Brier +0.0006 / +0.0014 / +0.0020; all CIs exclude 0.
5. **Practical significance** [I]:
   - The movement edge is modest: Brier skill against a constant validation-prevalence forecast is only **2.4–5.1 %**.
   - The directional edge is negligible.
   - An AUC above 0.5 does **not** by itself imply a useful or profitable trading signal. No trading evaluation was performed or authorised.

---

## 2. Data sources and integrity verification

### 2.1 Sources (read only)

`reports/data_prep/stage_d5_validation_evaluation/`:

| File | SHA-256 (before = after) |
|---|---|
| `d5_val_probabilities.npz` | `ef8704cb32709837d2e610d88cb54abbc9885158d2f6cbaf144ba604d35f40fb` |
| `d5_validation_metrics.json` | `29f3c9aa9ee4a8e2ebc71aac5dab4d2ca562fbc7527630fe37333f589c79bcb8` |
| `d5_run_metadata.json` | `4511a185edf8e908aef6cb9ec887e1a50d4441e5b99b23ad18ec104e96115f63` |

Also read:
- `STAGE_D5_VALIDATION_EVALUATION_REPORT.md` (`f047f1a0…`)
- `D5_1_NEUTRAL_BIAS_INVESTIGATION.md` (`6cf62b67…`)
- `evaluation/metrics_ext.py::block_bootstrap`, for the day-resampling scheme only.

### 2.2 Pre-execution checks [M] (all passed; the run asserts each one before any computation)

| Check | Result |
|---|---|
| All three D5 outputs exist; `d5_outputs.sha256` verifies | OK |
| NPZ contains exactly the 30 expected keys (10 per horizon) | OK |
| Shapes: every array has n rows; P_A and P_D are (n, 3); n = 91,800 / 89,100 / 85,050 | OK |
| Labels in {0, 1, 2}; supports equal D5 G5 exactly | OK |
| Probabilities finite, non-negative, rows sum to 1 within 1e-9 | OK |
| Alignment: 3-class Brier and log loss recomputed from the npz equal the D5 JSON values for all six combinations | difference 0.0 |
| `day_idx` ∈ [82, 108] with exactly 27 unique days (2024-10-15..2024-11-25) at every horizon | OK |
| Timestamps between 2024-10-15 and 2024-11-25 (IST); test begins at day 109 (2024-11-26) | OK — no test rows |
| `FLASH_TEST_UNLOCK` | unset |

### 2.3 Execution record [M]

| Item | Value |
|---|---|
| Start | 2026-11-10T08:10:10+05:30 |
| End | 2026-11-10T08:16:15+05:30 |
| Runtime | 365 s (360.6 s in Python) |
| Exit code | 0 |
| Earlier attempt (~08:08) | Failed with exit 2 during **bash** parsing of the inline heredoc, before Python started. Nothing was computed or written. The identical code was re-run by piping it to Python's stdin through PowerShell. |
| Files written | none apart from this report |

---

## 3. Methodology and definitions

Notation: P = (P_D, P_N, P_U) are the stored D5 validation probabilities, uncalibrated with the identity calibrator. y ∈ {0 = DOWN, 1 = NEUTRAL, 2 = UP}.

### 3.1 Stage 1 — movement vs neutral (all validation rows)

- Target: t₁ = 1 if y ∈ {DOWN, UP}, else 0.
- Score: P(MOVE) = P_D + P_U = 1 − P_N.
- **Ranking metrics** (no threshold):
  - ROC-AUC via `sklearn.metrics.roc_auc_score`.
  - **Average precision (AP)** via `sklearn.metrics.average_precision_score`, positive class = MOVEMENT. This is the step-wise sum Σₙ (Rₙ − Rₙ₋₁)·Pₙ, not trapezoidal interpolation. A random ranker scores AP equal to prevalence.
- **Probabilistic metrics:**
  - Binary Brier = mean (P(MOVE) − t₁)².
  - Binary log loss with probabilities clipped to [1e-15, 1 − 1e-15].
- **Threshold-dependent metrics** at a **fixed** 0.5 (binary argmax), predicting MOVEMENT iff P(MOVE) > 0.5: accuracy, precision, recall, F1 (positive class = MOVEMENT), confusion matrix.
- No row had P(MOVE) exactly 0.5.

### 3.2 Stage 2 — UP vs DOWN, conditional on an actual movement

- Rows: only those with y ∈ {DOWN, UP}; NEUTRAL rows are excluded.
- Target: t₂ = 1 for UP, 0 for DOWN.
- Score: P(UP | MOVE) = P_U / (P_D + P_U), and P(DOWN | MOVE) = 1 − P(UP | MOVE).
- **Zero-denominator rows: 0** in every combination [M]. The minimum P_D + P_U among moving rows is 0.0337 / 0.0243 / 0.0276 for A and 0.1658 / 0.1953 / 0.1849 for D (h5 / h15 / h30).
- **Ranking:** ROC-AUC, and AP with **positive class = UP**.
- **Probabilistic:** Brier = mean (P(UP | MOVE) − t₂)²; binary log loss as in Stage 1.
- **Threshold-dependent** at a fixed 0.5 (predict UP iff P(UP | MOVE) > 0.5; 0 ties): accuracy, UP and DOWN precision and recall, macro-F1, confusion matrix.
- **This is an oracle-conditioned diagnostic.** At prediction time it is unknown whether a move will occur. Stage 2 is **not** comparable to the three-class D5 evaluation.

### 3.3 Paired day-block bootstrap

- B = 1,000 replicates, seed 0, 95 % percentile intervals (2.5th and 97.5th percentiles of valid replicates).
- Each replicate draws 27 trading days with replacement from days 82–108 and keeps **all rows of each drawn day**.
- The draws reproduce `block_bootstrap` in `evaluation/metrics_ext.py`: `np.random.default_rng(0)` followed by 1,000 sequential calls of `rng.choice(days, 27, replace=True)`.
  - Draw-matrix fingerprint: SHA-256 prefix `b011829853e3b193`.
  - First draw begins [104, 99, 95, 89, 90, 83, 84, 82].
  - Mean distinct days per replicate: 17.31.
- **One draw sequence is shared** across Models A and D, Stages 1 and 2, and all three horizons. All three horizons have the same 27 days, so the same day draws apply everywhere.
- Validity rule: a replicate is invalid for an AUC or AP metric if its sample contains only one class. Invalid replicates are excluded, not replaced. Paired differences are valid only when both members are valid.
- Result: **1000 / 1000 valid replicates for every metric and every difference** [M].

### 3.4 Per-day consistency

- Metrics are computed separately on each of the 27 days.
- Pre-specified validity rule: a daily AUC is valid only if each class of that binary task has ≥ 30 rows that day.
- Minimum daily counts [M]: moving rows 1,195 / 1,304 / 1,286; UP rows 543 / 498 / 407; DOWN rows 501 / 471 / 344 (h5 / h15 / h30). **All 27 days were valid** for every metric.

### 3.5 Leave-one-day-out (LODO)

The pooled AUC is recomputed 27 times, each time with one day removed, to test whether any single day drives the results.

---

## 4. Stage 1 — movement vs neutral results [M]

### 4.1 Ranking and probabilistic metrics

| Model | h | n | Movement prevalence | ROC-AUC | AP (random = prevalence) | Brier | Brier of constant prevalence forecast¹ | Brier skill¹ | Log loss |
|---|---|---|---|---|---|---|---|---|---|
| A | 5 | 91,800 | 0.3843 | **0.5935** | 0.4642 | 0.23086 | 0.23663 | 2.4 % | 0.6539 |
| D | 5 | 91,800 | 0.3843 | **0.6054** | 0.4846 | 0.22897 | 0.23663 | 3.2 % | 0.6498 |
| A | 15 | 89,100 | 0.4438 | **0.6064** | 0.5389 | 0.23933 | 0.24684 | 3.0 % | 0.6715 |
| D | 15 | 89,100 | 0.4438 | **0.6158** | 0.5536 | 0.23767 | 0.24684 | 3.7 % | 0.6679 |
| A | 30 | 85,050 | 0.4705 | **0.6286** | 0.5906 | 0.23784 | 0.24913 | 4.5 % | 0.6684 |
| D | 30 | 85,050 | 0.4705 | **0.6334** | 0.6041 | 0.23636 | 0.24913 | 5.1 % | 0.6651 |

¹ Reference = constant forecast equal to the *validation* movement prevalence, p(1 − p). Skill = 1 − Brier / reference. This reference uses validation information and is shown only for scale.

### 4.2 Threshold-dependent metrics (fixed 0.5)

The positive class is MOVEMENT.

| Model | h | Predicted MOVE | Accuracy | Precision | Recall | F1 | NEUTRAL precision / recall | Confusion [[TN, FP], [FN, TP]] |
|---|---|---|---|---|---|---|---|---|
| A | 5 | 2,986 | 0.6195 | 0.5593 | 0.0473 | 0.0873 | 0.6215 / 0.9767 | [[55201, 1316], [33613, 1670]] |
| D | 5 | 5,498 | 0.6263 | 0.5886 | 0.0917 | 0.1587 | 0.6287 / 0.9600 | [[54255, 2262], [32047, 3236]] |
| A | 15 | 12,524 | 0.5823 | 0.5929 | 0.1878 | 0.2852 | 0.5805 / 0.8971 | [[44455, 5099], [32121, 7425]] |
| D | 15 | 13,623 | 0.5898 | 0.6101 | 0.2102 | 0.3126 | 0.5862 / 0.8928 | [[44242, 5312], [31235, 8311]] |
| A | 30 | 24,977 | 0.5922 | 0.6067 | 0.3787 | 0.4663 | 0.5861 / 0.7819 | [[35211, 9824], [24862, 15153]] |
| D | 30 | 23,844 | 0.5957 | 0.6180 | 0.3683 | 0.4615 | 0.5870 / 0.7978 | [[35927, 9108], [25279, 14736]] |

[I] Movement recall is low at a 0.5 threshold, especially at h5, because P(MOVE) rarely exceeds 0.5. The median P(MOVE) is 0.36–0.46. These threshold metrics describe the fixed 0.5 rule only. **No threshold was searched.**

---

## 5. Stage 2 — conditional UP vs DOWN results [M]

### 5.1 Ranking and probabilistic metrics

The positive class is UP. Brier of a constant 0.5 forecast = 0.25000.

| Model | h | n moving rows | UP share | DOWN share | ROC-AUC | AP (random ≈ UP share) | Brier | Log loss (constant 0.5 = 0.6931) |
|---|---|---|---|---|---|---|---|---|
| A | 5 | 35,283 | 0.5017 | 0.4983 | **0.5316** | 0.5286 | 0.24941 | 0.69196 |
| D | 5 | 35,283 | 0.5017 | 0.4983 | **0.5213** | 0.5190 | 0.24998 | 0.69312 |
| A | 15 | 39,546 | 0.5019 | 0.4981 | **0.5294** | 0.5256 | 0.24941 | 0.69196 |
| D | 15 | 39,546 | 0.5019 | 0.4981 | **0.5056** | 0.5018 | 0.25080 | 0.69476 |
| A | 30 | 40,015 | 0.5000 | 0.5000 | **0.5259** | 0.5228 | 0.24959 | 0.69233 |
| D | 30 | 40,015 | 0.5000 | 0.5000 | **0.4906** | 0.4932 | 0.25158 | 0.69632 |

[M] P(UP | MOVE) is tightly concentrated around 0.5. Its 5th–95th percentile range is about 0.44–0.56 for every combination.

### 5.2 Threshold-dependent metrics (fixed 0.5)

| Model | h | Accuracy | UP precision / recall | DOWN precision / recall | Macro-F1 | Confusion, rows = actual DOWN, UP [[DOWN→D, DOWN→U], [UP→D, UP→U]] |
|---|---|---|---|---|---|---|
| A | 5 | 0.5252 | 0.5275 / 0.5149 | 0.5231 / 0.5356 | 0.5252 | [[9417, 8164], [8587, 9115]] |
| D | 5 | 0.5140 | 0.5160 / 0.5059 | 0.5121 / 0.5222 | 0.5140 | [[9181, 8400], [8746, 8956]] |
| A | 15 | 0.5243 | 0.5269 / 0.5115 | 0.5218 / 0.5371 | 0.5242 | [[10580, 9117], [9696, 10153]] |
| D | 15 | 0.5029 | 0.5045 / 0.5354 | 0.5011 / 0.4701 | 0.5023 | [[9260, 10437], [9221, 10628]] |
| A | 30 | 0.5190 | 0.5194 / 0.5085 | 0.5185 / 0.5294 | 0.5189 | [[10592, 9414], [9835, 10174]] |
| D | 30 | 0.4911 | 0.4915 / 0.5119 | 0.4907 / 0.4704 | 0.4909 | [[9411, 10595], [9767, 10242]] |

[I] Conditional directional accuracy is 49.1–52.5 %, against a 50.0–50.2 % majority-class rate among moving rows.

---

## 6. Bootstrap 95 % confidence intervals [M]

All entries had 1000 / 1000 valid replicates. "Share > 0.5" is the fraction of replicates in which the AUC or AP exceeded 0.5.

| Model | h | Movement AUC | Movement AP | Movement Brier | Direction AUC (share > 0.5) | Direction AP (share > 0.5) | Direction Brier |
|---|---|---|---|---|---|---|---|
| A | 5 | [0.5865, 0.6008] | [0.4566, 0.4716] | [0.2293, 0.2324] | [0.5227, 0.5400] (1.000) | [0.5054, 0.5541] (0.991) | [0.2488, 0.2500] |
| D | 5 | [0.5972, 0.6139] | [0.4744, 0.4947] | [0.2275, 0.2306] | [0.5099, 0.5321] (1.000) | [0.4968, 0.5419] (0.961) | [0.2493, 0.2507] |
| A | 15 | [0.5952, 0.6177] | [0.5271, 0.5500] | [0.2369, 0.2418] | [0.5170, 0.5408] (1.000) | [0.4879, 0.5651] (0.891) | [0.2488, 0.2500] |
| D | 15 | [0.6038, 0.6276] | [0.5384, 0.5704] | [0.2354, 0.2397] | **[0.4868, 0.5227] (0.691)** | [0.4662, 0.5396] (0.536) | [0.2497, 0.2520] |
| A | 30 | [0.6157, 0.6402] | [0.5731, 0.6063] | [0.2354, 0.2404] | [0.5040, 0.5473] (0.990) | [0.4693, 0.5756] (0.790) | [0.2483, 0.2508] |
| D | 30 | [0.6174, 0.6482] | [0.5781, 0.6266] | [0.2339, 0.2388] | **[0.4640, 0.5177] (0.224)** | [0.4437, 0.5406] (0.384) | [0.2500, 0.2531] |

**Movement AUC − direction AUC, same model, same replicates:**

| Model | h5 | h15 | h30 |
|---|---|---|---|
| A | [+0.0498, +0.0746] | [+0.0585, +0.0970] | [+0.0760, +0.1299] |
| D | [+0.0703, +0.0981] | [+0.0904, +0.1313] | [+0.1142, +0.1717] |

All six intervals exclude 0, with 1000 / 1000 valid replicates each.

**Directional AP caveat:** for AP, "share > 0.5" is only a rough reference. Random AP equals the UP prevalence of each resample, which varies around 0.50.

---

## 7. Paired Model D − Model A comparisons [M]

Same day draws for both models. For AUC and AP, a positive difference favours D; for Brier, a negative difference favours D. **✔** marks a CI that excludes 0. Valid replicates were 1000 / 1000 for all.

| h | Metric | Point D − A | 95 % CI | Share of replicates where D > A | Reading |
|---|---|---|---|---|---|
| 5 | Movement AUC | +0.01193 | [+0.00890, +0.01506] ✔ | 1.000 | D better |
| 5 | Movement AP | +0.02040 | [+0.01541, +0.02545] ✔ | 1.000 | D better |
| 5 | Movement Brier | −0.00189 | [−0.00239, −0.00139] ✔ | 0.000 | D better |
| 5 | Direction AUC | −0.01033 | [−0.01790, −0.00190] ✔ | 0.005 | **D worse** |
| 5 | Direction AP | −0.00953 | [−0.01886, −0.00029] ✔ | 0.023 | **D worse** |
| 5 | Direction Brier | +0.00058 | [+0.00005, +0.00106] ✔ | 0.979 | **D worse** |
| 15 | Movement AUC | +0.00944 | [+0.00370, +0.01525] ✔ | 1.000 | D better |
| 15 | Movement AP | +0.01472 | [+0.00529, +0.02361] ✔ | 1.000 | D better |
| 15 | Movement Brier | −0.00166 | [−0.00251, −0.00079] ✔ | 0.000 | D better |
| 15 | Direction AUC | −0.02379 | [−0.04163, −0.00579] ✔ | 0.003 | **D worse** |
| 15 | Direction AP | −0.02380 | [−0.04179, −0.00330] ✔ | 0.009 | **D worse** |
| 15 | Direction Brier | +0.00139 | [+0.00030, +0.00253] ✔ | 0.991 | **D worse** |
| 30 | Movement AUC | +0.00485 | [−0.00405, +0.01388] | 0.858 | not distinguishable |
| 30 | Movement AP | +0.01348 | [+0.00248, +0.02479] ✔ | 0.989 | D better |
| 30 | Movement Brier | −0.00148 | [−0.00285, −0.00006] ✔ (marginal) | 0.023 | D better, marginally |
| 30 | Direction AUC | −0.03527 | [−0.06475, −0.00745] ✔ | 0.009 | **D worse** |
| 30 | Direction AP | −0.02962 | [−0.05477, −0.00484] ✔ | 0.006 | **D worse** |
| 30 | Direction Brier | +0.00198 | [+0.00039, +0.00365] ✔ | 0.996 | **D worse** |

[I] The D-over-A advantage seen in the three-class D5 Brier and log loss comes entirely from the movement component. On direction, D is consistently worse than A.

Caveats on this comparison:
- It is **not** evidence about news (D5 restrictions 5–7).
- D's early stopping was selected on this validation split, using the three-class log loss. That may favour D's movement component here. [H]

---

## 8. Trading-day consistency [M]

All 27 days were valid for every metric (each class ≥ 30 rows per day). "Days > 0.5" counts days with AUC above 0.5.

| Model | h | Movement AUC: median [IQR], min–max | Days > 0.5 | Direction AUC: median [IQR], min–max | Days > 0.5 | Days movement AUC > direction AUC | Median daily gap |
|---|---|---|---|---|---|---|---|
| A | 5 | 0.598 [0.583, 0.608], 0.555–0.632 | **27 / 27** | 0.526 [0.510, 0.550], 0.491–0.574 | 24 / 27 | 26 / 27 | +0.061 |
| D | 5 | 0.607 [0.594, 0.620], 0.553–0.658 | **27 / 27** | 0.523 [0.497, 0.543], 0.464–0.593 | 19 / 27 | 26 / 27 | +0.085 |
| A | 15 | 0.606 [0.589, 0.628], 0.562–0.668 | **27 / 27** | 0.523 [0.494, 0.541], 0.467–0.581 | 18 / 27 | 27 / 27 | +0.094 |
| D | 15 | 0.614 [0.602, 0.637], 0.563–0.703 | **27 / 27** | 0.495 [0.464, 0.533], 0.375–0.596 | 13 / 27 | 27 / 27 | +0.116 |
| A | 30 | 0.630 [0.612, 0.647], 0.534–0.680 | **27 / 27** | 0.511 [0.476, 0.536], 0.393–0.615 | 16 / 27 | 25 / 27 | +0.133 |
| D | 30 | 0.640 [0.631, 0.649], 0.512–0.715 | **27 / 27** | 0.482 [0.427, 0.517], 0.363–0.716 | 10 / 27 | 26 / 27 | +0.149 |

### 8.1 Daily prevalence

Prevalence depends only on labels, so it is the same for A and D.

| h | Movement prevalence: median [IQR], min–max | UP share among moving rows: median [IQR], min–max |
|---|---|---|
| 5 | 0.384 [0.375, 0.393], 0.352–0.405 | 0.496 [0.478, 0.518], 0.419–0.635 |
| 15 | 0.437 [0.428, 0.460], 0.395–0.495 | 0.485 [0.454, 0.538], 0.345–0.712 |
| 30 | 0.465 [0.445, 0.501], 0.408–0.532 | 0.483 [0.440, 0.543], 0.287–0.789 |

### 8.2 Extreme days

Lowest and highest daily direction AUC (date, value):

| Model | h | Lowest | Highest |
|---|---|---|---|
| A | 5 | 2024-10-17 0.491 | 2024-10-28 0.574 |
| D | 15 | 2024-11-18 0.375 | 2024-11-19 0.596 |
| A | 30 | 2024-11-21 0.393 | 2024-11-05 0.615 |
| D | 30 | 2024-11-11 0.363 | **2024-11-19 0.716** |

The lowest daily movement AUC was 2024-10-23: 0.534 (A h30) and 0.512 (D h30). The highest were on 2024-10-31 and 2024-11-25 (0.66–0.72).

### 8.3 Per-day D − A

| h | Movement AUC: days D > A | Median diff | Direction AUC: days D > A | Median diff |
|---|---|---|---|---|
| 5 | 25 / 27 | +0.009 | 8 / 27 | −0.010 |
| 15 | 20 / 27 | +0.006 | 9 / 27 | −0.030 |
| 30 | 14 / 27 | +0.000 | 9 / 27 | −0.015 |

### 8.4 Leave-one-day-out pooled AUC range [M]

| Model | h | Movement AUC range | Direction AUC range |
|---|---|---|---|
| A | 5 | 0.5920–0.5950 | 0.5300–0.5333 |
| D | 5 | 0.6034–0.6075 | 0.5186–0.5236 |
| A | 15 | 0.6039–0.6093 | 0.5265–0.5316 |
| D | 15 | 0.6126–0.6178 | 0.5007–0.5104 |
| A | 30 | 0.6264–0.6317 | 0.5194–0.5308 |
| D | 30 | 0.6301–0.6378 | **0.4819–0.4956** |

No single day moves any pooled AUC by more than 0.014. Removing 2024-10-31 lowers movement AUC the most for 5 of 6 combinations.

---

## 9. Horizon-wise comparison [M]

| | h5 | h15 | h30 |
|---|---|---|---|
| Movement AUC, A / D | 0.594 / 0.605 | 0.606 / 0.616 | **0.629 / 0.633** |
| Movement Brier skill vs prevalence, A / D | 2.4 % / 3.2 % | 3.0 % / 3.7 % | **4.5 % / 5.1 %** |
| Direction AUC, A / D | **0.532** / **0.521** | 0.529 / 0.506 | 0.526 / 0.491 |
| Direction Brier, A / D (0.25 = coin flip) | 0.2494 / 0.2500 | 0.2494 / 0.2508 | 0.2496 / 0.2516 |
| Days with direction AUC > 0.5, A / D | 24 / 19 | 18 / 13 | 16 / 10 |

- **Strongest movement signal: h30**, for both models. Movement AUC rises with horizon. AP also rises, partly because movement prevalence rises (0.38 → 0.47).
- **Strongest directional signal: h5**, for both models. A is roughly flat across horizons (0.532 → 0.526). D declines with horizon and falls below 0.5 at h30.

---

## 10. Interpretation and limitations

### 10.1 Answers to the nine questions

1. **Is movement prediction meaningfully stronger than direction prediction?**
   - [M] Yes, statistically. The movement-minus-direction AUC gap excludes 0 for all six combinations (+0.05 to +0.17). It holds on 25–27 of 27 days.
   - [I] It is also "meaningfully" stronger in a relative sense. In absolute terms, the movement signal itself is modest (AUC ≈ 0.6, Brier skill 2–5 %).
2. **Does D improve movement prediction compared with A?**
   - [M] Yes at h5 and h15 on AUC, AP and Brier. At h30, AP and Brier favour D but the AUC difference is not distinguishable from 0.
   - [I] The size is small: +0.005 to +0.012 AUC, −0.0015 to −0.0019 Brier.
3. **Does D improve directional prediction compared with A?** [M] No. D is significantly **worse** than A on direction AUC, AP and Brier at every horizon.
4. **Does directional discrimination remain close to chance?**
   - [M] Yes. AUC is 0.491–0.532 and conditional Brier is within ±0.0016 of the coin-flip 0.25.
   - A's direction AUC CIs exclude 0.5 narrowly, so there is detectable but tiny discrimination. D's h15 and h30 CIs include 0.5.
5. **Are the results consistent across trading days?**
   - [M] Movement: yes, AUC > 0.5 on 27 / 27 days for all six combinations.
   - Direction: no. AUC > 0.5 on only 10–24 days, and daily values range widely (D h30: 0.363–0.716).
6. **Are the results driven by a few days?** [M] No. Leave-one-day-out changes pooled AUCs by ≤ 0.008 (movement) and ≤ 0.014 (direction). The bootstrap had 1000 / 1000 valid replicates.
7. **Which horizon has the strongest movement signal?** [M] h30.
8. **Which horizon has the strongest directional signal?** [M] h5 for both models, though all directional values are near chance.
9. **Are A–D differences statistically distinguishable?** [M] Yes for 16 of the 18 paired comparisons in §7 (17 counting the marginal h30 Brier). The exception is h30 movement AUC.
   - [I] "Statistically distinguishable" here means a 95 % day-block bootstrap CI excluding 0, over 27 days with no multiplicity correction. It does not imply practical importance.

### 10.2 Statistical vs practical significance [I]

- With about 85–92k rows, many small differences become statistically detectable. Day blocking reduces, but does not remove, this effect.
- The direction-AUC differences that are statistically significant (A at 0.52–0.53) are practically negligible: conditional accuracy is 51.9–52.5 % against a 50 % base rate, before any costs.
- The movement signal (AUC ≈ 0.60–0.63) is the only consistent signal. Whether it has any practical value was **not** evaluated. No decision rule, strategy, cost model or profitability analysis was performed or authorised.

### 10.3 Limitations

1. **Validation reuse.** Validation was already used for PCA selection, D's early stopping, D3 and D5. These figures are optimistic and not an independent generalisation estimate.
2. **Uncalibrated probabilities.** The conditional P(UP | MOVE) is a renormalisation of three-class probabilities. Neither model was trained for this conditional task.
3. **Oracle conditioning in Stage 2.** It uses the *actual* movement indicator, which is unknown at prediction time. Stage 2 measures direction discrimination only, not a deployable two-step rule.
4. **27 days.** The day-block bootstrap has 27 clusters, about 17.3 distinct days per replicate. Percentile CIs from so few clusters may be somewhat narrow. [H]
5. **No multiplicity correction.** The report contains 36 single-model CIs, 18 paired differences, 6 gap CIs and the per-day statistics.
6. **Pooled AUC mixes sources of variation.** It includes cross-sectional (stock), time-of-day and between-day variation. Per-day AUCs remove only the between-day part.
7. **AP depends on prevalence** and uses sklearn's step-wise definition. Compare it against prevalence, not against 0.5.
8. **The fixed 0.5 threshold** is arbitrary for Stage 1. The threshold metrics there describe a rule that rarely predicts movement, not the best achievable trade-off. No threshold was tuned, as instructed.
9. **Brier-skill reference** uses the validation prevalence, an oracle constant. It is shown for scale only.
10. **Scope.** Only the stored D5 arrays were used. No feature-level, component-level (LightGBM vs logistic) or news-specific inference is possible from them.

---

## 11. Integrity and Git-status verification (after execution) [M]

| Check | Result |
|---|---|
| D5 outputs (`d5_outputs.sha256`: metrics, metadata, npz) | all OK — unchanged |
| D5 script / log / preflight / review / execution report / D5.1 report | `0cf317ef…` / `e28def8e…` / `e849e585…` / `cd7f3cf6…` / `f047f1a0…` / `6cf62b67…` — unchanged |
| D4 artifacts (`d4_artifacts.sha256`, CR stripped) | OK |
| D4 driver / metadata / inventory / report | `3d496a72…` / `d14eddf8…` / `2563cd14…` / `4ff8beea…` — unchanged |
| D3 report / json / log | `3fcc3fae…` / `f74ad79f…` / `6ded7eb0…` — unchanged |
| 40-file data checklist / 24-file D2 checklist / `models/` | 40 / 40, 24 / 24, 2 / 2 OK |
| News source parquet | `7213bcd4a42d8ea3ab95798461524bb0ca6479085ac1be2afe26e524fd42e65d` ✔ |
| Source / config / dependency changes | none (only tracked modification is the pre-existing `.gitignore` patch, +15) |
| `FLASH_TEST_UNLOCK` | unset; the test split remains locked and unexamined |

**Git:**

| Item | Value |
|---|---|
| Branch | `data-pipeline` |
| HEAD | `ffa1cda72861d22ac9f2ec71e029858e582cc9d3` (last commit 2026-11-05, i.e. no new commits) |
| `backup/pre-data-pipeline-20261109` | `ffa1cda…` (same) |
| Staged | 0 |
| Untracked (`-uall`) | 62 before this report; this report adds 1 |
| Pushes | none |

---

## 12. Stage boundary

E3 is complete. **None** of the following has started:
- E1 label sensitivity
- E2 decision rules
- E4 component investigation
- E5 calibration
- E6 retraining
- E7 news ablation
- E8 late-session analysis
- D6
- any model modification

Test data was not accessed and the test split remains locked. Waiting for explicit approval.
