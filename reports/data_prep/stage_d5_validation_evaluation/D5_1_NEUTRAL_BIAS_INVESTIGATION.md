# Stage D5.1 — NEUTRAL prediction bias investigation (diagnostic only)

**Scope:** read-only diagnosis of why Models A and D predict NEUTRAL for 98.4–99.99 % of validation rows.

**What was done:**
- No model was trained, retrained, calibrated or re-thresholded.
- No source, config, dependency or D1–D5 artifact was modified.
- No test labels or test predictions were accessed. The test split remains locked.

**Evidence tags:** every finding is tagged **[M]** (measured on validation data or read from existing artifacts) or **[H]** (hypothesis, not verified here).

**Data used:**
- `d5_val_probabilities.npz`, `d5_validation_metrics.json`, `d5_run_metadata.json`.
- Source code: `src/features/market_features.py`, `src/models/fusion_lgbm.py`, `src/models/baseline_logistic.py`, `src/inference/predictor.py`, `configs/real.yaml`.
- Existing LightGBM model text from `models_real/stage_d4/`, read only, using `feature_importance`.
- Raw 1-minute bars from `data/raw/market/*.parquet`, **filtered to the validation dates 2024-10-15..2024-11-25 immediately after reading**. Features and labels are rebuilt per symbol-day, so this reproduces validation exactly without touching any later rows.
- **Reconstruction check [M]:** the validation-only rebuild reproduced the D5 labels exactly for all 266,000 rows across the three horizons:

| h | Rows matched | Labels equal | `valid_x` true for all |
|---|---|---|---|
| 5 | 91,800 / 91,800 | ✔ | ✔ |
| 15 | 89,100 / 89,100 | ✔ | ✔ |
| 30 | 85,050 / 85,050 | ✔ | ✔ |

All computation ran as inline read-only Python. No script, cache or intermediate file was written. This report is the only new file.

---

## 1. Executive summary

1. **The NEUTRAL dominance is mostly rational given the information the models have. It is not a pipeline bug.** [M]
   - NEUTRAL is the plurality label at every horizon: 61.6 % / 55.6 % / 53.0 % (h5 / h15 / h30).
   - The models learned real signal about **whether** price will move: AUC for move-vs-neutral is 0.59–0.63.
   - They learned almost nothing about **which direction**: AUC for UP-vs-DOWN among moving rows is 0.49–0.53.
   - When the model lowers P(NEUTRAL), the freed probability is split roughly evenly between UP and DOWN. Neither directional class alone overtakes NEUTRAL, so argmax stays NEUTRAL.
2. **The label design makes NEUTRAL the plurality class by construction.** [M]
   - The symmetric barrier ±1·σ₃₀·√h would label 46–60 % of rows NEUTRAL even for a pure random walk (simulation, §2.4).
   - Real data adds more, because trailing σ₃₀ overstates forward volatility: the median ratio of forward RMS return to trailing σ is 0.87–0.97.
3. **The models' predictive features are volatility and time of day, not direction.** [M]
   - `vol30` and `tod` carry 35–48 % of LightGBM gain.
   - Every one of the 14 market features has univariate UP-vs-DOWN AUC within 0.47–0.53.
4. **D's 1,084 DOWN calls at h30 are mostly late-session move detections, not direction detections.** [M]
   - 69 % fall in session minutes 300–345.
   - Their realised labels are DOWN 38.8 %, NEUTRAL 16.6 %, **UP 44.6 %**.
   - D's UP-vs-DOWN AUC at h30 is **0.492**, below chance.
5. **The models are not sharp enough in the tails.** [M]
   - In the lowest-P(NEUTRAL) decile at h30, observed NEUTRAL is 0.29–0.32, yet the models still assign P(NEUTRAL) ≈ 0.39–0.42.
   - Mean P(NEUTRAL) also exceeds the validation NEUTRAL share by 1.5–2.9 pp. Training priors were more NEUTRAL than validation.
   - This is a secondary contributor and would not, on its own, produce useful directional calls. [M/H]
6. **Not measured here:** the blend's own contribution (0.75 LightGBM / 0.25 logistic) cannot be measured from the stored artifacts, because component probabilities were not saved (§6.4).

---

## 2. How labels are generated

### 2.1 Code [M]

`src/features/market_features.py::triple_barrier` and `batch_labels`. Settings from `configs/real.yaml`: `k: 1.0`, `vol_window: 30`, `scale_sqrt_h: true`, horizons 5 / 15 / 30.

For each symbol-day and each bar *i* with a full horizon left in the session:

- σᵢ = `vol30`, the standard deviation of the last 30 one-minute log returns up to and including bar *i*. It resets at each session open, so a row becomes valid from minute 30.
- The barrier is **thr = k · σᵢ · √h**, with k = 1. It is symmetric.
- The path is log(close_{i+j}) − log(close_i) for j = 1..h. **Only closes are used, not highs or lows.**
- **UP** if +thr is reached first, **DOWN** if −thr is reached first. **NEUTRAL** if neither is reached, or if both are first reached at the same bar, which is only possible when thr = 0.
- −1 (excluded) if the horizon runs past the session end or σ is not finite.
- Encoding: DOWN = 0, NEUTRAL = 1, UP = 2.

### 2.2 Thresholds [M]

| | h5 | h15 | h30 |
|---|---|---|---|
| Directional barrier, median | 11.5 bps | 19.8 bps | 27.8 bps |
| Directional barrier, 5th–95th percentile | 6.6–21.7 bps | 11.4–37.4 bps | 16.0–52.8 bps |
| Rows with σ = 0 (tie case) | 0 | 0 | 0 |

- There is **no separate "neutral threshold"**. NEUTRAL means neither barrier was hit within *h* bars.
- The barrier varies with **horizon** (through √h) and **trailing volatility** (through σ₃₀). Through σ₃₀ it also varies implicitly with **stock** and **time of day**.
- The barrier is the same for UP and DOWN, and k is constant.

### 2.3 Can the logic systematically favour NEUTRAL?

- **Bugs:** none found. [M]
  - The implementation matches its docstring.
  - The D2 independent naive re-implementation (`preprocessing/leakage_audit.py`, L3) agreed.
  - Tie handling cannot fire, since there are 0 rows with σ = 0.
- **Design properties that favour NEUTRAL:**
  1. **Barrier width.** A ±1·σ·√h barrier is wide relative to the typical *h*-bar path. The median of max|excursion| / barrier is 0.83 / 0.92 / 0.96, so the typical path stays inside the barrier. [M]
  2. **Volatility mean reversion.** The trailing σ₃₀ overstates forward volatility: median forward RMS / σ₃₀ = 0.870 / 0.946 / 0.966. [M] NEUTRAL share is almost entirely set by this *future* ratio: from 0.99 in the lowest ratio quintile to 0.15 in the highest at h5, and from 0.87 to 0.14 at h30. [M]
  3. **Session open.** The first 30 minutes of each session feed σ₃₀ for early rows. Rows at minutes 30–60 are the most NEUTRAL: 0.67–0.68 at h5, 0.60–0.66 at h30. [M] Opening volatility inflating σ is the likely cause. [H]
  4. **Close-only paths.** Barrier touches inside a bar (via high or low) are ignored. This can only increase NEUTRAL. [H — size not measured]
  5. **Session-end truncation.** Removes rows near the close. This changes which rows exist, not their label balance. [M]

### 2.4 Random-walk reference [M — simulation, no data]

Simulation: 200,000 synthetic paths per horizon under the same barrier rule (k = 1, √h scaling, σ estimated from 30 trailing returns where noted). Values are the NEUTRAL share.

| h | Gaussian, σ known | Gaussian, σ estimated from 30 bars | Student-t(3), σ estimated | **Observed on validation** |
|---|---|---|---|---|
| 5 | 0.561 | 0.538 | 0.601 | **0.616** |
| 15 | 0.493 | 0.465 | 0.497 | **0.556** |
| 30 | 0.461 | 0.435 | 0.446 | **0.530** |

NEUTRAL is the plurality class by design even with no market structure. Real data adds about 2–9 pp on top.

---

## 3. Class distributions (validation only) [M]

### 3.1 By horizon (DOWN / NEUTRAL / UP)

| h | Validation | Train (from D3 baseline fit) |
|---|---|---|
| 5 | 0.1915 / 0.6157 / 0.1928 | 0.1881 / 0.6248 / 0.1871 |
| 15 | 0.2211 / 0.5562 / 0.2228 | 0.2140 / 0.5732 / 0.2128 |
| 30 | 0.2352 / 0.5295 / 0.2353 | 0.2292 / 0.5420 / 0.2289 |

- Labels are symmetric between UP and DOWN.
- Train was more NEUTRAL than validation by 0.9 / 1.7 / 1.3 pp.

### 3.2 By stock: NEUTRAL share range across the 10 stocks

| h | Lowest | Highest |
|---|---|---|
| 5 | 0.601 (SBIN) | 0.642 (HINDUNILVR) |
| 15 | 0.526 (RELIANCE) | 0.598 (KOTAKBANK) |
| 30 | 0.491 (RELIANCE) | 0.597 (KOTAKBANK) |

- KOTAKBANK (median price ≈ ₹350, 23 % zero one-minute returns) and HINDUNILVR are consistently the most NEUTRAL.
- Directional imbalance per stock is small, at most ±6 pp at h30: HDFCBANK leans DOWN (0.275 vs 0.218 UP), SBIN and TCS lean UP.

### 3.3 By trading day (27 days)

| h | NEUTRAL share range (sd) | UP − DOWN range |
|---|---|---|
| 5 | 0.595–0.649 (0.013) | −0.06 to +0.11 |
| 15 | 0.505–0.605 (0.026) | −0.14 to +0.21 |
| 30 | 0.468–0.592 (0.037) | −0.19 to +0.30 |

- The most DOWN days were 10-22 and 11-12; the most UP days were 10-29 and 11-22.
- NEUTRAL is the plurality class on **every** day at every horizon.

### 3.4 By minute of session, h30 (DOWN / NEUTRAL / UP)

| Minutes | DOWN / NEUTRAL / UP |
|---|---|
| 30–45 | 0.169 / 0.662 / 0.168 |
| 120–180 | 0.196 / 0.561 / 0.242 |
| 300+ | 0.332 / 0.401 / 0.267 |

NEUTRAL falls steadily through the session, and the late session leans DOWN. h5 and h15 follow the same pattern with smaller amplitude.

### 3.5 By trailing-σ quintile, h30 NEUTRAL share

| σ₃₀ quintile | Lowest | 2 | 3 | 4 | Highest |
|---|---|---|---|---|---|
| NEUTRAL share | 0.365 | 0.487 | 0.523 | 0.581 | 0.691 |

High trailing volatility predicts **more** NEUTRAL, because the barrier widens while forward volatility mean-reverts. This is the main exploitable "move" signal.

---

## 4. Probability distributions [M]

Values per horizon are listed as h5 / h15 / h30.

### 4.1 Model A

| Quantity | h5 / h15 / h30 |
|---|---|
| Mean P(DOWN / NEUTRAL / UP) at h5 | 0.185 / 0.631 / 0.184 |
| Mean P(NEUTRAL) at h15, h30 | 0.585, 0.558 |
| Median P(NEUTRAL) | 0.627 / 0.575 / 0.545 |
| 1st–99th percentile of P(NEUTRAL) | 0.46–0.83 / 0.42–0.82 / 0.39–0.83 |
| Max directional probability: median | 0.198 / 0.222 / 0.238 |
| Max directional probability: 99th percentile | 0.288 / 0.306 / 0.328 |
| Max directional probability: maximum | 0.536 / 0.512 / 0.568 |
| P(N) − max directional: median | 0.428 / 0.353 / 0.306 |
| P(N) − max directional: 1st percentile | 0.169 / 0.118 / 0.061 |
| Rows where a directional class is top | 0.12 % / 0.08 % / 0.18 % |
| Rows where P(N) − max directional < 0.10 | 0.35 % / 0.65 % / 2.81 % |
| Rows with max probability < 0.5 (the live 0.5 gate would show NO SIGNAL) | 3.3 % / 14.1 % / 29.4 % |

### 4.2 Model D

| Quantity | h5 / h15 / h30 |
|---|---|
| Mean P(DOWN / NEUTRAL / UP) at h5 | 0.184 / 0.631 / 0.185 |
| Mean P(NEUTRAL) at h15, h30 | 0.584, 0.557 |
| Median P(NEUTRAL) | 0.636 / 0.584 / 0.557 |
| 1st–99th percentile of P(NEUTRAL) | 0.43–0.79 / 0.39–0.76 / 0.34–0.77 |
| Max directional probability: median | 0.192 / 0.218 / 0.232 |
| Max directional probability: 99th percentile | 0.308 / 0.332 / 0.362 |
| Max directional probability: maximum | 0.412 / 0.425 / 0.447 |
| P(N) − max directional: median | 0.444 / 0.366 / 0.325 |
| P(N) − max directional: 1st percentile | 0.127 / 0.064 / **−0.021** |
| Rows where a directional class is top | 0.01 % / 0.13 % / 1.60 % |
| Rows where P(N) − max directional < 0.10 | 0.55 % / 2.19 % / 6.56 % |
| Rows with max probability < 0.5 (live gate NO SIGNAL) | 6.0 % / 15.3 % / 28.0 % |

D's probability range is narrower than A's at both ends: P(NEUTRAL) spans 0.25–0.85 for D vs. 0.07–0.99 for A.

### 4.3 Is there directional information despite NEUTRAL winning argmax?

**Discrimination (AUC):**

| AUC | A h5 | A h15 | A h30 | D h5 | D h15 | D h30 |
|---|---|---|---|---|---|---|
| Move vs. neutral (score 1 − P(N)) | 0.594 | 0.606 | 0.629 | 0.605 | 0.616 | 0.633 |
| UP one-vs-rest | 0.562 | 0.564 | 0.578 | 0.565 | 0.563 | 0.569 |
| DOWN one-vs-rest | 0.584 | 0.590 | 0.600 | 0.595 | 0.596 | 0.603 |
| **UP vs. DOWN among moving rows (P(U) − P(D))** | **0.532** | **0.529** | **0.525** | **0.522** | **0.506** | **0.492** |

The UP and DOWN one-vs-rest AUCs are well above 0.5. However, they are inflated by the move-vs-neutral component: high P(UP) mostly means "high P(move)".

**Highest-ranked rows [M]:** realised label shares among the rows with the highest P(UP) or P(DOWN). The class the model favoured is in bold.

| Model / h | Bucket | Realised DOWN / NEUTRAL / UP |
|---|---|---|
| D h30 | top 1 % P(UP) | 0.445 / 0.231 / **0.325** — the opposite direction is more frequent |
| D h30 | top 1 % P(DOWN) | **0.360** / 0.165 / 0.475 — the opposite direction is more frequent |
| A h30 | top 1 % P(UP) | 0.393 / 0.236 / **0.371** |
| A h5 | top 1 % P(UP) | 0.268 / 0.451 / **0.281** — NEUTRAL still dominates |

**Lowest P(NEUTRAL) decile [M]:** realised label shares among the 10 % of rows with the lowest P(NEUTRAL).

| Model / h | Mean P(N) | Realised DOWN / NEUTRAL / UP |
|---|---|---|
| A h30 | 0.420 | **0.347** / 0.323 / 0.330 |
| D h30 | 0.389 | **0.392** / 0.293 / 0.315 |
| A h5 | 0.503 | 0.277 / 0.476 / 0.247 |
| D h5 | 0.483 | 0.294 / 0.440 / 0.266 |

At h30, NEUTRAL is no longer the most frequent realised class in this decile; DOWN is, narrowly. The models still give P(NEUTRAL) ≈ 0.39–0.42 there. At h5, NEUTRAL remains the realised majority even in the lowest decile.

**Interpretation:**
- **Directional information exists only as "movement" information.** The models can rank rows by how likely a barrier hit is. They cannot reliably say which barrier.
- Argmax NEUTRAL is therefore close to the Bayes decision almost everywhere.
- The exception is a small tail at h30, where both models stay under-confident (too close to the prior).

---

## 5. Stock-wise and horizon-wise findings [M]

### 5.1 Directional argmax calls by stock

Each cell is DOWN calls / UP calls.

| Stock | A h5 | D h5 | A h15 | D h15 | A h30 | D h30 |
|---|---|---|---|---|---|---|
| BHARTIARTL | 1 / 7 | 0 / 1 | 3 / 0 | 3 / 0 | 6 / 1 | 28 / 14 |
| HDFCBANK | 1 / 7 | 0 / 1 | 4 / 5 | 5 / 1 | 27 / 2 | **218 / 43** |
| HINDUNILVR | 4 / 9 | 0 / 3 | 7 / 2 | 12 / 9 | 9 / 1 | 96 / 25 |
| ICICIBANK | 4 / 11 | 0 / 0 | 3 / 5 | 6 / 0 | 17 / 2 | 100 / 31 |
| INFY | 0 / 5 | 0 / 3 | 3 / 1 | 1 / 13 | 6 / 1 | 106 / 41 |
| ITC | 6 / 9 | 0 / 0 | 7 / 3 | 26 / 15 | 20 / 2 | **198 / 47** |
| KOTAKBANK | 1 / 9 | 0 / 1 | 2 / 2 | 3 / 0 | 5 / 2 | 32 / 1 |
| RELIANCE | 4 / 9 | 0 / 2 | 4 / 5 | 0 / 10 | 11 / 3 | 66 / 29 |
| SBIN | 12 / 10 | 0 / 0 | 9 / 6 | 0 / 0 | 23 / 4 | 88 / 17 |
| TCS | 0 / 1 | 0 / 1 | 1 / 1 | 6 / 10 | 11 / 0 | 152 / 30 |

### 5.2 Per-stock AUCs, ranges across the 10 stocks

| | A h5 | A h15 | A h30 | D h5 | D h15 | D h30 |
|---|---|---|---|---|---|---|
| Move vs. neutral | 0.585–0.610 | 0.587–0.645 | 0.606–0.661 | 0.600–0.622 | 0.595–0.658 | 0.598–0.673 |
| UP vs. DOWN | 0.511–0.559 | 0.514–0.545 | 0.508–0.543 | 0.501–0.552 | 0.492–0.518 | **0.443–0.545** |

### 5.3 Observations

- **Move signal:** strongest for BHARTIARTL, ICICIBANK and RELIANCE at h30 (AUC 0.65–0.67), weakest for SBIN (≈ 0.60).
- **Direction signal:**
  - A: the best is KOTAKBANK at h5 (0.559). Every value is ≤ 0.56.
  - D h30: below 0.5 for 8 of 10 stocks (KOTAKBANK 0.443, TCS 0.468, HINDUNILVR 0.480). It is above 0.5 only for ICICIBANK (0.545) and RELIANCE (0.502).
- **Horizon pattern:**
  - The move signal improves with horizon.
  - The direction signal does not improve. It decays slightly for A and falls below chance for D at h30.
  - Lower NEUTRAL base rates at longer horizons are why the few directional calls appear mostly at h30.
- **Concentration of D's h30 DOWN calls:** HDFCBANK and ITC together account for 39 % of them. These are the stocks with the highest D mean P(DOWN) tail (95th-percentile max directional probability 0.33–0.34). HDFCBANK also has the most DOWN-leaning labels (0.275).

---

## 6. Model A versus Model D

### 6.1 Agreement [M]

| | h5 | h15 | h30 |
|---|---|---|---|
| Argmax agreement | 99.87 % | 99.79 % | 98.35 % |
| Correlation of P(N) between A and D | 0.82 | 0.84 | 0.85 |
| Correlation of P(U) − P(D) between A and D | 0.77 | 0.52 | 0.47 |

The two models agree closely on *whether* a move is likely, and much less on *which way*.

### 6.2 Why D makes 1,084 DOWN calls at h30 [M, interpretation H]

**Where the calls fall:**

| Minutes | 30–60 | 60–120 | 120–180 | 180–240 | 240–300 | 300–345 |
|---|---|---|---|---|---|---|
| D h30 DOWN calls | 0 | 1 | 40 | 54 | 245 | **744** |
| D mean P(DOWN) | 0.163 | 0.195 | 0.213 | 0.228 | 0.241 | 0.274 |
| Realised DOWN share | 0.179 | 0.222 | 0.196 | 0.250 | 0.231 | 0.332 |

- The calls cover 25 of 27 days and are concentrated at the end of the session.
- **What actually happened after a DOWN call:** DOWN 38.8 %, NEUTRAL 16.6 %, **UP 44.6 %**.
  - The call correctly identifies that a move happens: realised NEUTRAL is 16.6 % vs. a 53 % base rate.
  - It points the wrong way slightly more often than the right way.
- **Day-to-day instability:** precision on 11-08 was 0.61 and on 11-12 was 0.63, but on 11-07 it was 0.19 and on 11-18 it was 0.11.
- **Interpretation [H]:** D learned that late-session rows are high-move, and that the late session leaned DOWN in training (consistent with validation, §3.4). The late-session move signal pushes P(N) down. The learned DOWN tilt then makes DOWN the top class for some rows. The direction part does not generalise day to day.
- D's 278 UP calls at h30 behave the same way: precision 0.27, with DOWN 0.49 realised.

### 6.3 Why D makes almost no directional calls at h5 and h15 [M]

- At h5, NEUTRAL has its highest base rate (0.616). D's maximum P(DOWN) is 0.365, while its minimum P(N) is 0.321. The 1st percentile of P(N) − max directional is +0.127, so directional argmax is nearly impossible.
- At h15, D's 120 directional calls all fall in minutes 326–345. They come from the same late-session effect, with weaker magnitude. Precision was 0.34 for DOWN and 0.19 for UP.
- D's probability range is compressed compared with A's (§4.2), so D rarely reaches the region where a directional class wins.

### 6.4 Do the LightGBM and logistic components disagree? Does the 0.75 / 0.25 blend add to NEUTRAL dominance?

**Not measurable from the stored artifacts.** D5 saved only the blended P_D. Recovering the components would require rebuilding the 42-feature matrix with `prepare()` and running prediction with the existing D4 models on validation rows. That is a non-training computation, but it re-runs the full-dataset `prepare()` that processes test-period rows in memory, as D5 did. I did not run it without your approval; see E4.

What can be stated:

- [M] A convex blend P = 0.75·L + 0.25·S cannot produce a directional argmax unless at least one component has that directional class as its own argmax. It also cannot make any class probability more extreme than the more extreme component.
- [M] D's probabilities are compressed compared with A's: max P(N) 0.85 vs. 0.99, min P(N) 0.25 vs. 0.07.
- [H] Compression could come from the heavily regularised LightGBM (see below) or from averaging with the C = 0.05 logistic. The cause cannot be attributed without the components.

LightGBM regularisation settings: 107–174 trees at learning rate 0.03, 7 leaves, min_child_samples 300, reg_lambda 10, colsample 0.5.

### 6.5 Feature-group evidence

**Existing non-invasive method:** `Booster.feature_importance("gain")` on the D4 model text. These are training-time importances pooled over the three class trees. They are not validation evidence and not per-class.

| Group gain share | h5 | h15 | h30 |
|---|---|---|---|
| market (14) | 0.689 | 0.732 | 0.724 |
| derived (4) | 0.215 | 0.100 | 0.042 |
| derived_x (6) | 0.030 | 0.061 | 0.088 |
| regime (4) | 0.030 | 0.043 | 0.066 |
| news (11) | 0.024 | 0.039 | 0.040 |
| history (3) | 0.013 | 0.025 | 0.040 |

- **Top features:**
  - h5: `vol30` 0.24, `tod` 0.11, `ret15_vol` 0.10, `ret5_vol` 0.09, `vol_z` 0.09.
  - h30: `vol30` 0.32, `tod` 0.17, `var20` 0.09, `mkt_vol20` 0.05.
- **Features never split on:** `n_ev_M&A`, `n_ev_Regulation`, `n_ev_Legal/Fraud`, `n_ev_Management` at all horizons, plus some return features at h15 and h30.
- These importances describe what LightGBM used. **They do not show that news has, or lacks, value** (restrictions 5–7 of the D5 report).

**Univariate separation on validation, 14 market features [M]:** computed from the validation-only rebuild.

| AUC on validation | Range across the 14 market features |
|---|---|
| UP vs. DOWN | **0.471–0.529** at every horizon |
| Move vs. neutral | `vol30` 0.42 / 0.40 / 0.37 (inverted: high σ means more NEUTRAL); `var20` 0.43–0.39; `tod` 0.53–0.56; `vol_z` 0.54–0.55 |

- The best UP-vs-DOWN features are `close_pos` (0.471–0.484, a weak mean-reversion sign) and `cov20` / `mkt_vol20` at h30 (0.525–0.529).
- The 28 non-market features (regime, news, history, derived) were **not** measured univariately, because that also requires `prepare()`.

---

## 7. Ranked potential root causes

| # | Explanation | Supporting evidence | Contradicting evidence | Confidence | Verifying experiment |
|---|---|---|---|---|---|
| 1 | **Weak directional signal; argmax NEUTRAL is near the Bayes decision** | UP-vs-DOWN AUC 0.49–0.53 for both models [M]; market-feature UP-vs-DOWN AUC ≤ 0.53 [M]; top-1 % P(UP) or P(DOWN) rows split almost evenly between directions [M]; D h30 DOWN calls realise more UP than DOWN [M] | Lowest-P(N) decile at h30 has realised DOWN > NEUTRAL, so there is a small region where NEUTRAL is not optimal [M]; non-market features were not measured univariately | **High** | E3 two-stage decomposition with day-block CIs on direction AUC; E4 component and feature-group measurement |
| 2 | **Label design makes NEUTRAL the plurality class** (k = 1, σ₃₀·√h, close-only, trailing σ) | Random-walk simulation gives 44–60 % NEUTRAL [M]; observed 53–62 % [M]; forward/trailing σ ratio < 1 [M]; early-session rows most NEUTRAL [M] | Design is symmetric and bug-free; it sets the base rate, not the model's inability to pick a direction | **High** (as cause of the imbalance) | E1 label-design sensitivity (k, high/low touch, σ estimator) — descriptive, no training needed for class shares |
| 3 | **Learned signal is "volatility / time of day", which lowers P(N) but splits the freed mass between UP and DOWN** | Move-vs-neutral AUC 0.59–0.63 vs. UP-vs-DOWN ≈ 0.5 [M]; `vol30` + `tod` = 35–48 % of gain [M]; directional calls cluster late in the session [M] | — | **High** | E3 |
| 4 | **Unweighted multinomial loss on imbalanced classes** (no class weights or cost-sensitive decision rule) | Training NEUTRAL share 54–62 % [M]; log-loss-optimal models reproduce priors where signal is weak (theory) | With signal this weak, re-weighting would raise directional recall mainly through false positives [H] | **Medium** (proximate mechanism, not a defect) | E6 class-weighted retraining vs. current on a fresh holdout (needs authorisation) |
| 5 | **Insufficient sharpness / calibration, plus a train→validation prior shift** | Lowest-decile P(N) 0.39–0.50 vs. realised 0.29–0.48 [M]; mean P(N) above validation share by 1.5–2.9 pp [M]; train NEUTRAL share 0.9–1.7 pp above validation [M]; ECE worse than prior/momentum (D5) [M] | Even perfectly calibrated, only a small h30 tail would change argmax, and direction there is barely predictable (#1) | **Medium** | E5 calibration on an independent slice with a fresh evaluation window |
| 6 | **Model capacity / early stopping** (107–174 trees, heavy regularisation, early stopping on log loss) | D's probability range is compressed compared with A's [M] | D's AUCs match or beat A's [M], so extra capacity has not revealed hidden direction signal; log-loss early stopping is appropriate for probabilities | **Low–medium** [H] | E6 / E4 variants: capacity sweep evaluated by UP-vs-DOWN AUC on a fresh holdout |
| 7 | **0.75 / 0.25 blend dilutes or compresses directional probabilities** | D is compressed compared with A [M]; a convex blend cannot exceed its sharpest component [M, math] | Not measured: components unavailable | **Unknown / low** | **E4** (component probabilities on validation, prediction only) |
| 8 | **Confidence gate** | — | D5 metrics used plain argmax. The 0.5 gate (`inference.gate_threshold`) only applies to live Predictor labels (NO SIGNAL for 3–29 % of rows) | **Not a cause** of the D5 figures | — |

---

## 8. Recommended experiments (none started; each needs explicit authorisation)

**General caveat:** validation has already been used for PCA selection, early stopping, D3 and D5. Any experiment that *chooses* something (k, thresholds, weights, calibration) must be chosen on a walk-forward inside the training period, or on a newly defined held-out window. Choosing on validation would contaminate it further. The test split stays locked throughout.

| ID | Hypothesis | Method (summary) | Evaluation criteria | Needs |
|---|---|---|---|---|
| **E1** | NEUTRAL share is set mainly by barrier width and the σ estimator, not by data structure | Recompute class shares on train+val for k ∈ {0.5, 0.75, 1.0}, high/low-touch barriers, and σ estimators that exclude the opening period (e.g. EWMA or prior-day σ). Descriptive only; no model training | Class shares per setting vs. the random-walk reference; stability across days and stocks; forward/trailing σ ratio closer to 1 | Read-only approval; label code not changed (analysis copy in memory) |
| **E2** | A decision rule other than argmax (e.g. act when P(dir) − P(N) > τ, or expected utility with costs) yields directional calls with precision above the direction base rate | Choose τ by walk-forward inside train; evaluate once on a fresh window | Directional precision vs. base rate with day-block CIs; coverage; cost-adjusted P&L via existing backtest logic | Threshold-selection authorisation |
| **E3** | Direction is nearly unpredictable with current features; movement is predictable | Score the existing P_A / P_D as two stages: P(move) = 1 − P(N), and P(UP \| move). Day-block bootstrap CIs on both AUCs (can use the existing npz, validation only); optionally fit separate stage models later | Direction AUC CI excludes 0.5? Move AUC stability across days and stocks | Bootstrap on the npz: read-only approval. Separate models: training authorisation |
| **E4** | The blend or the logistic leg compresses D's probabilities; specific feature groups carry move vs. direction information | Rebuild features with `prepare()` and predict the LGBM and logistic legs separately on validation with the existing D4 models (no fitting). Measure component argmax, disagreement, Brier, and univariate AUC of all 42 features | Component argmax shares; share of rows where the components disagree on argmax; whether the blend changes argmax; per-group UP-vs-DOWN AUC | Approval to re-run `prepare()` (processes test-period rows in memory, as in D5; validation rows only scored) |
| **E5** | Calibrating on independent data sharpens the h30 tail enough to change argmax in the low-P(N) region | Temperature or vector scaling fitted on a calibration slice disjoint from evaluation; evaluate on a fresh window | ECE and Brier change; change in directional-argmax share; precision of new directional calls | Calibration authorisation and a new split protocol |
| **E6** | Class weighting or capacity changes raise directional recall without destroying proper scores | Retrain with balanced or cost weights, or a capacity sweep (trees, leaves), selected on walk-forward within train | Per-class recall / precision / F1, macro-F1, Brier, log loss, UP-vs-DOWN AUC vs. current; day-block CIs | Retraining authorisation |
| **E7** | News features add (or do not add) value to D | D vs. D-without-news ablation (already flagged in D5) | Paired Brier / log loss / AUC differences with CIs on a fresh window | Ablation authorisation |
| **E8** | Late-session DOWN calls reflect a training-period intraday seasonality that does not generalise | Use the existing npz and train labels: compare late-session DOWN skew per day in train vs. validation; precision of D h30 calls per day | Day-level variance of late-session DOWN share; whether call precision tracks it | Read-only approval |

---

## 9. Limitations and unresolved questions

1. **Component probabilities are unavailable** (LGBM leg vs. logistic leg of D). The blend's effect (#7) and component disagreement are unmeasured.
2. **Only the 14 market features were measured univariately on validation.** Regime, news, history and derived features were not, because that requires `prepare()`.
3. **Gain importances are training-time measures**, pooled over the three class trees. They show what the model used, not what generalises, and are not per-class.
4. **Univariate AUCs ignore interactions.** Small multivariate directional effects could exist. The models' UP-vs-DOWN AUC of 0.49–0.53 is the multivariate evidence.
5. **27 validation days.** Day-level effects (e.g. the variation in D h30 call precision) rest on few clusters. No bootstrap CIs were computed for the new diagnostic statistics.
6. **Validation reuse:** validation was already used for PCA, early stopping, D3 and D5. These diagnostics are descriptive, not independent evidence.
7. **The random-walk reference** is an idealised simulation (Gaussian or t(3), no microstructure, no intraday seasonality).
8. **Close-only barriers:** the effect of ignoring intrabar high/low touches on NEUTRAL share was not quantified (E1).
9. **Anomaly observed:**
   - `lgb.Booster(model_file=...)` **fails** on `models_real/stage_d4/lgbm_D_h*.txt`. The files have CRLF line endings, written by `write_text` on Windows, and LightGBM 4.7's file parser rejects them.
   - Loading the same text via `model_str` (Python universal newlines) works. D5 used model strings from the bundle, and D5 G3 confirmed string equality after newline handling.
   - The files were not modified. Any future stage that loads them via `model_file` will fail.
10. **Unresolved:**
    - Whether the late-session DOWN tilt is a stable intraday pattern or a training-period artifact (E8).
    - Whether any feature group, including news, carries direction information (E4 / E7).
    - Whether a cost-aware decision rule can extract value from the move signal alone (E2).

---

## 10. Integrity and Git-status verification (after all diagnostics) [M]

| Check | Result |
|---|---|
| 40-file D3-era checklist (includes all 10 `data/raw/market/*.parquet` files read in this stage) | 40 / 40 OK |
| 24-file D2 checklist | 24 / 24 OK |
| `models/` (2 files) | OK |
| D4 artifacts (`d4_artifacts.sha256`, CR stripped) | all OK |
| D4 driver / log / metadata / inventory / report | `3d496a72…` / `28247485…` / `d14eddf8…` / `2563cd14…` / `4ff8beea…` — unchanged |
| D3 report / json / log | `3fcc3fae…` / `f74ad79f…` / `6ded7eb0…` — unchanged |
| D5 outputs (`d5_outputs.sha256`: metrics, metadata, npz) | all OK |
| D5 script / preflight / review / execution report | `0cf317ef…` / `e849e585…` / `cd7f3cf6…` / `f047f1a0…` — unchanged |
| News source parquet | `7213bcd4a42d8ea3ab95798461524bb0ca6479085ac1be2afe26e524fd42e65d` ✔ |
| FinBERT cache | `43447937…` — unchanged |
| `FLASH_TEST_UNLOCK` | unset |

**Git:**

| Item | Value |
|---|---|
| Branch | `data-pipeline` |
| HEAD | `ffa1cda72861d22ac9f2ec71e029858e582cc9d3` |
| `backup/pre-data-pipeline-20261109` | `ffa1cda…` (same) |
| Tracked modifications | `.gitignore` only (+15, the pre-existing patch) |
| Staged | 0 |
| Untracked (`-uall`) | 61 before this report; this report adds 1 |

Nothing was committed, staged or pushed.

---

## Stage boundary

D5.1 is diagnostic only and is complete. **None** of the following has started:
- the proposed experiments E1–E8
- model modification or retraining
- calibration
- threshold or gate tuning
- test evaluation or test unlock
- D6

Waiting for review and explicit approval.
