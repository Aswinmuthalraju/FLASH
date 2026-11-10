# Stage D3: Validation Baseline Evaluation (real data, VAL split only)

- Date: 2026-11-09
- Repo: `C:\Users\prsan\Projects\FLASH`, branch `data-pipeline`, HEAD `ffa1cda72861d22ac9f2ec71e029858e582cc9d3` (uncommitted Session 2 patch applied)
- Scope: the three existing FLASH baselines (majority, class prior, momentum) at h = 5, 15 and 30 bars, scored on the **validation split only**.
- Not done: no predictive models were trained or fitted (no LightGBM, LR or FinBERT training). No test-split scoring. No code, config, test or dependency changes. No commits.

Unless marked *derived* or *interpretation*, every number below is copied from `validation_baselines_real.json`, which FLASH's own `evaluation/run_baselines.py` produced.

---

## 1. Preflight results (before the run)

| # | Check | Result |
|---|---|---|
| 1 | HEAD and backup branch at `ffa1cda` | PASS: both `ffa1cda72861…` |
| 2 | Current branch | PASS: `data-pipeline` |
| 3 | `FLASH_TEST_UNLOCK` | PASS: unset |
| 4 | Real market data and `configs/real.yaml` present | PASS |
| 5 | D2 outputs present; D2 JSON shows val = 27 days | PASS |
| 6 | 24 D1 data/report hashes unchanged (`d2_before.sha`) | PASS |
| 7 | `models/` hashes unchanged (`models_before.sha`) | PASS |
| 8 | Working tree: 44 status entries (untracked-all), none unexpected | PASS |
| 9 | Output directory did not exist (no overwrite risk); fresh 40-file baseline saved to `d3_before.sha` | PASS |

The interface was inspected before the run:
- `run_baselines.py` refuses `--split test` unless both `--unlock-test` and `FLASH_TEST_UNLOCK=I_HAVE_APPROVAL` are given. Neither was used.
- `--out` is required, so the default-path report `reports/baselines/real_val_baselines.json` (written by the `audit` stage of `run_pipeline.sh`) was not touched.
- The script writes only the `--out` file and stdout.

## 2. Exact command

```
cd C:\Users\prsan\Projects\FLASH
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe evaluation/run_baselines.py \
    --config configs/real.yaml --market data/raw/market --split val \
    --out reports/data_prep/stage_d3_validation_baselines/validation_baselines_real.json
```

- `--B` was left at its default of 1000 bootstrap replicates (seed 0, alpha 0.05, set inside `metrics_ext.block_bootstrap`).
- Started 2026-11-09T18:36:42+05:30 and finished 19:39:41+05:30 (elapsed 3,779 s, exit code 0). Most of the runtime is the day-block bootstrap: 9 model×horizon pairs × 1,000 replicates.
- Stdout went to `validation_baselines_real.log`, along with the command, cwd, unlock-variable state, timestamps and exit code.

## 3. Baseline methodology (as implemented; nothing changed)

Source files: `evaluation/baselines.py`, `evaluation/metrics_ext.py`, `src/evaluation/metrics.py`.

**Inputs**
- Market bars only, read from `data/raw/market`. **News and FinBERT are not used by any baseline.**
- Labels are the triple-barrier labels from `configs/real.yaml`. Classes are `[DOWN, NEUTRAL, UP]`.

**Row selection**
- `rows()` keeps rows where `split == s`, `valid_x` is true and the label is ≥ 0.
- Every baseline parameter is estimated from **train rows only**. Val rows are only scored.

**The three baselines**
- **majority:** a one-hot prediction on the most frequent train class.
- **prior:** constant train class frequencies.
- **momentum:**
  - z = (log close[t] − log close[t−h], same session) / (vol30 · √h).
  - Rows are bucketed into train terciles of z. The prediction is P(class | tercile), counted on train with Laplace smoothing a = 1.
  - If no same-session past return exists, it falls back to the prior.

**Metrics** (all implemented by FLASH):
- n, accuracy, macro-F1, AUC (one-vs-rest, macro), Brier (summed over classes, averaged over rows), ECE (10 bins), log loss.
- Per-class precision, recall, F1 and support; the confusion matrix [true × pred]; per-class reliability curves (10 bins, bins with ≥ 20 rows).
- Day-block bootstrap 95% CIs over the 27 val days.
- The non-prior models also get `diff_vs_ref_ci`, a bootstrap CI of (model − prior) for each metric. For Brier and ECE, negative means better than prior.

**Balanced accuracy is not implemented in FLASH.** The single figure given in §5 is *derived* (mean of the reported per-class recalls) and is not a FLASH output.

### Fitted-on-train parameters (`fit_on_train`)

| h | n_train | train prior [D, N, U] | majority | z tercile edges | P(class \| tercile) low / mid / high [D, N, U] |
|---|---|---|---|---|---|
| 5 | 278,800 | 0.1881, 0.6248, 0.1871 | NEUTRAL | −0.3643, 0.3530 | [.1892,.6130,.1978] / [.1741,.6564,.1696] / [.2010,.6051,.1939] |
| 15 | 270,600 | 0.2140, 0.5732, 0.2128 | NEUTRAL | −0.3794, 0.3662 | [.2222,.5641,.2136] / [.2076,.5927,.1997] / [.2121,.5628,.2251] |
| 30 | 258,300 | 0.2292, 0.5420, 0.2289 | NEUTRAL | −0.3970, 0.3868 | [.2453,.5314,.2233] / [.2195,.5604,.2201] / [.2227,.5341,.2433] |

NEUTRAL has the highest probability in every momentum tercile, so the argmax of every baseline is always NEUTRAL (§5).

## 4. Validation metrics

Val split: 27 trading days (2024-10-15 .. 2024-11-25). Bootstrap uses B = 1000 replicates over 27 day blocks. 95% CIs are in brackets.

### h = 5 (n = 91,800; support D 17,581 / N 56,517 / U 17,702)

| Model | Accuracy | Macro-F1 | AUC (OvR) | Brier | ECE | Log loss |
|---|---|---|---|---|---|---|
| majority | 0.6157 [0.6112, 0.6205] | 0.2540 [0.2529, 0.2553] | 0.5000 [0.5000, 0.5000] | 0.7687 [0.7590, 0.7776] | 0.3843 [0.3795, 0.3888] | 10.6199 |
| prior | 0.6157 [0.6112, 0.6205] | 0.2540 [0.2529, 0.2553] | 0.5000 [0.5000, 0.5000] | 0.5472 [0.5430, 0.5511] | 0.0092 [0.0043, 0.0136] | 0.9327 |
| momentum | 0.6157 [0.6112, 0.6205] | 0.2540 [0.2529, 0.2553] | **0.5187** [0.5141, 0.5233] | **0.5466** [0.5425, 0.5505] | **0.0087** [0.0039, 0.0130] | **0.9319** |

### h = 15 (n = 89,100; support D 19,697 / N 49,554 / U 19,849)

| Model | Accuracy | Macro-F1 | AUC (OvR) | Brier | ECE | Log loss |
|---|---|---|---|---|---|---|
| majority | 0.5562 [0.5468, 0.5654] | 0.2383 [0.2357, 0.2408] | 0.5000 [0.5000, 0.5000] | 0.8877 [0.8692, 0.9064] | 0.4438 [0.4346, 0.4532] | 12.2637 |
| prior | 0.5562 [0.5468, 0.5654] | 0.2383 [0.2357, 0.2408] | 0.5000 [0.5000, 0.5000] | 0.5926 [0.5860, 0.5994] | 0.0171 [0.0078, 0.0264] | 0.9951 |
| momentum | 0.5562 [0.5468, 0.5654] | 0.2383 [0.2357, 0.2408] | **0.5079** [0.5025, 0.5131] | **0.5925** [0.5859, 0.5994] | **0.0168** [0.0078, 0.0260] | **0.9949** |

### h = 30 (n = 85,050; support D 20,006 / N 45,035 / U 20,009)

| Model | Accuracy | Macro-F1 | AUC (OvR) | Brier | ECE | Log loss |
|---|---|---|---|---|---|---|
| majority | 0.5295 [0.5164, 0.5432] | 0.2308 [0.2270, 0.2347] | 0.5000 [0.5000, 0.5000] | 0.9410 [0.9135, 0.9672] | 0.4705 [0.4568, 0.4836] | 13.0001 |
| prior | 0.5295 [0.5164, 0.5432] | 0.2308 [0.2270, 0.2347] | 0.5000 [0.5000, 0.5000] | 0.6092 [0.6006, 0.6174] | 0.0125 [0.0013, 0.0256] | 1.0178 |
| momentum | 0.5295 [0.5164, 0.5432] | 0.2308 [0.2270, 0.2347] | **0.5118** [0.5036, 0.5189] | **0.6089** [0.6006, 0.6168] | **0.0122** [0.0013, 0.0250] | **1.0176** |

Log loss has no bootstrap CI because FLASH does not compute one. Majority's log loss is large because it assigns probability 0 to the true class on every non-NEUTRAL row; the metric clips that zero.

### Per-class metrics and confusion matrices

At every horizon, all three baselines produce the **same** argmax prediction (NEUTRAL on every row). As a result, the per-class metrics and confusion matrices are identical across the three models:

| h | DOWN P / R / F1 | NEUTRAL P / R / F1 | UP P / R / F1 | Confusion [true × pred], rows D, N, U |
|---|---|---|---|---|
| 5 | 0 / 0 / 0 | 0.6157 / 1.0 / 0.7621 | 0 / 0 / 0 | [[0,17581,0],[0,56517,0],[0,17702,0]] |
| 15 | 0 / 0 / 0 | 0.5562 / 1.0 / 0.7148 | 0 / 0 / 0 | [[0,19697,0],[0,49554,0],[0,19849,0]] |
| 30 | 0 / 0 / 0 | 0.5295 / 1.0 / 0.6924 | 0 / 0 / 0 | [[0,20006,0],[0,45035,0],[0,20009,0]] |

Per-class reliability curves (DOWN/NEUTRAL/UP) are stored in the JSON under `reliability` for each model and horizon. They are not reproduced here.

### Bootstrap CI of the difference vs prior (`diff_vs_ref_ci`, model − prior)

| h | Model | ΔAUC | ΔBrier | ΔECE | ΔMacro-F1 / ΔAcc |
|---|---|---|---|---|---|
| 5 | momentum | [+0.0141, +0.0233] | [−0.00091, −0.00031] | [−0.00068, −0.00023] | [0, 0] |
| 5 | majority | [0, 0] | [+0.2160, +0.2264] | [+0.3752, +0.3752] | [0, 0] |
| 15 | momentum | [+0.0025, +0.0131] | [−0.00042, +0.00009] | [−0.00052, −0.00007] | [0, 0] |
| 15 | majority | [0, 0] | [+0.2832, +0.3070] | [+0.4268, +0.4268] | [0, 0] |
| 30 | momentum | [+0.0036, +0.0189] | [−0.00068, +0.00023] | [−0.00058, +0.00004] | [0, 0] |
| 30 | majority | [0, 0] | [+0.3129, +0.3498] | [+0.4555, +0.4580] | [0, 0] |

## 5. Comparison across horizons

| Metric | h = 5 | h = 15 | h = 30 |
|---|---|---|---|
| n (val rows) | 91,800 | 89,100 | 85,050 |
| Val NEUTRAL share (= accuracy of all three) | 0.6157 | 0.5562 | 0.5295 |
| Macro-F1 (all three) | 0.2540 | 0.2383 | 0.2308 |
| Prior Brier | 0.5472 | 0.5926 | 0.6092 |
| Momentum Brier | 0.5466 | 0.5925 | 0.6089 |
| Momentum AUC | 0.5187 | 0.5079 | 0.5118 |
| Momentum ΔAUC vs prior, CI excludes 0? | yes | yes | yes |
| Momentum ΔBrier vs prior, CI excludes 0? | yes (better) | no | no |
| Prior ECE | 0.0092 | 0.0171 | 0.0125 |
| Majority Brier | 0.7687 | 0.8877 | 0.9410 |
| *Derived* balanced accuracy (mean per-class recall) | 0.3333 | 0.3333 | 0.3333 |

**Interpretation** (an assessment of the numbers above, not new measurements):
- **Val looks like train.** The train prior is well calibrated on val (ECE 0.009–0.017), so the val class mix is close to train's. As h grows, the label distribution shifts away from NEUTRAL (val NEUTRAL share falls from 0.616 to 0.530), so constant-prediction Brier and log loss get worse with horizon.
- **Momentum's edge is small.** It is the only baseline with ranking signal above chance, and that signal is weak: AUC 0.508–0.519, with every ΔAUC CI excluding zero. Its Brier gain over prior is about 0.0003–0.0009 and is statistically separable only at h = 5.
- **Accuracy and macro-F1 cannot tell the baselines apart** at any horizon, because all three predict NEUTRAL everywhere. Future models should be compared on Brier, log loss, AUC and ECE against **prior** (the strongest calibrated constant) and **momentum**, not on accuracy.
- **Bars to beat on val:** Brier below 0.5466 / 0.5925 / 0.6089 and AUC above 0.519 / 0.508 / 0.512 at h = 5 / 15 / 30.

## 6. Limitations

1. **These numbers use the val split.** Downstream stages that tune on val will make val estimates optimistic. Per D2 (L8), val is also used for PCA component selection. The test split remains untouched and locked.
2. **The bootstrap rests on only 27 day-blocks.** CIs capture day-to-day variation within one 6-week window (2024-10-15 .. 11-25), not regime change.
3. **Majority and prior are constants.** Their AUC is exactly 0.5 by construction, and their CIs on AUC, accuracy and macro-F1 reflect only label-mix variation across days.
4. **Argmax metrics are degenerate.** Every baseline predicts NEUTRAL on every row, so per-class P/R/F1 for DOWN and UP are 0.
5. **Balanced accuracy is not a FLASH metric.** The 0.3333 above is derived by hand from the reported recalls (1/3 by construction when only one class is ever predicted).
6. **No log-loss CI.** FLASH does not bootstrap log loss.
7. **Known data caveats from D1/D2 still apply** and are not re-assessed here:
   - 278 returns beyond 10σ
   - 3 OHLC violations
   - the Muhurat session dropped on 2024-11-01
   - a possibly pre-filled 375-bar grid
   - the FinBERT revision is not pinned in code (irrelevant to these market-only baselines)
8. **Momentum feature details.** It uses the same-session past return and vol30 only. Rows without a same-session h-bar history fall back to the prior; the share of fallback rows is not reported by FLASH.

## 7. Post-run integrity checks

| # | Check | Result |
|---|---|---|
| 1 | All 40 pre-D3 files (real data, npz caches, all prior reports, smoke dirs, `reports/baselines/*`, `models/*`) unchanged (`sha256sum -c d3_before.sha`) | PASS (40/40) |
| 2 | 24 D1/D2 files unchanged (`d2_before.sha`) | PASS (24/24) |
| 3 | `models/bundle.joblib`, `models/event_memory.npz` unchanged | PASS |
| 4 | News source sha256 = `7213bcd4a42d8ea3ab95798461524bb0ca6479085ac1be2afe26e524fd42e65d` | PASS |
| 5 | Session 2 patch still exactly applied (reverse `git apply --check`, excluding the by-design `MANIFEST.json`) | PASS |
| 6 | `git diff --name-only HEAD` shows only `.gitignore` (the patch's tracked change) | PASS |
| 7 | `FLASH_TEST_UNLOCK` unset; `--unlock-test` not used | PASS |
| 8 | `models_real/`, `reports/real/` and `reports/baselines/real_val_baselines.json` absent (no training, no default-path report) | PASS |
| 9 | `reports/baselines/` contains only the pre-existing `SYNTHETIC_val_baselines.json` (hash unchanged, covered by #1) | PASS |
| 10 | HEAD and `backup/pre-data-pipeline-20261109` both `ffa1cda…`; branch `data-pipeline` | PASS |
| 11 | Nothing staged (`git diff --cached` empty); no commits | PASS |
| 12 | Output JSON has `split = "val"` and horizons 5/15/30 only; no test scoring occurred | PASS |

## 8. New files (all in `reports/data_prep/stage_d3_validation_baselines/`)

| File | Size (bytes) | sha256 |
|---|---|---|
| `validation_baselines_real.json` | 29,502 | `f74ad79fe949f6b99d13685e1c0ba2a942c135f81b394de3f050406c50eb12e5` |
| `validation_baselines_real.log` | 1,245 | `6ded7eb027680bd6f765694580fcd1abba949b46a57e4e51097f7144ad50f427` |
| `STAGE_D3_VALIDATION_BASELINES_REPORT.md` | (this file) | — |

No other files were created, modified, moved or deleted.

## 9. Git status

- Branch `data-pipeline`, HEAD `ffa1cda72861d22ac9f2ec71e029858e582cc9d3`.
- No commits and nothing staged.
- Tracked modification: `.gitignore` only (from the Session 2 patch).
- Untracked entries (all files listed): 46 = 44 before D3 + the JSON and log above. This report adds one more, giving 47.

---

**Stage D3 is complete. Stopping here.** Model training, calibration, test evaluation and later stages have not started and are waiting for review and approval.
