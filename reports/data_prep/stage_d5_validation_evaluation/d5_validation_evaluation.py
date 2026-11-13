"""FLASH Stage D5 -- VALIDATION-ONLY evaluation of the six D4 models vs the three D3 baselines.

Reuses the UNMODIFIED FLASH components:
    src.data.build.get_data, src.evaluation.experiments.prepare / variant_cols   (features; cache-only FinBERT)
    src.models.fusion_lgbm.load_booster                                         (D4 LightGBM text models)
    evaluation.baselines.Baselines                                               (D3 baselines, train-only fit)
    evaluation.metrics_ext.full_report / block_bootstrap                         (D3 metric code)
    src.inference.predictor.Predictor / replay                                   (one-day streaming cross-check)

Data scope (accepted in D4): prepare() processes the FULL chronological dataset in memory, including
test-period rows. This driver selects ONLY validation rows for prediction and scoring (train rows only inside
Baselines.fit, exactly as D3). Any request for a non-train/val split raises. No calibration is fitted,
no thresholds or gates are tuned; probabilities are UNCALIBRATED (D4 identity calibrator).

Validation was already used for PCA n_components selection, LightGBM early stopping (D) and D3 -> results
are NOT an independent generalisation estimate.

Gates (each aborts BEFORE any metric is computed, writing d5_gate_failure.json):
  G1 env/no-overwrite   G2 D4 artifact hashes   G3 bundle structure/params vs D4 metadata
  G4 PCA/HMM/col_names reproducibility   G5 validation row counts + class supports
  G6 feature finiteness report + probability validity   G7 D3 baseline point-metric reproduction (1e-12)
  G8 one-day streaming Predictor vs batch consistency

Run (from the repo root; only after approval):
    .venv/Scripts/python.exe reports/data_prep/stage_d5_validation_evaluation/d5_validation_evaluation.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import lightgbm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

import evaluation.baselines as BL  # noqa: E402
from evaluation.metrics_ext import block_bootstrap, full_report  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.build import get_data  # noqa: E402
from src.evaluation import experiments as X  # noqa: E402
from src.evaluation.metrics import CLASS_NAMES, brier_multiclass  # noqa: E402
from src.features.market_features import market_return  # noqa: E402
from src.features.news_features import raw_matrix  # noqa: E402
from src.inference.predictor import Predictor, replay  # noqa: E402
from src.models.fusion_lgbm import load_booster  # noqa: E402

# ------------------------------------------------------------------------------------------- constants
CONFIG = ROOT / "configs" / "real.yaml"
MARKET, NEWS = "data/raw/market", "data/raw/news.csv"
FINBERT_CACHE = ROOT / "data" / "processed_real" / "finbert_cache.npz"
# Optional run tag (same as d4_train_driver.py): FLASH_D4_RUN_TAG=stage_d4_mac evaluates models_real/<tag>/ and writes to
# reports/data_prep/<tag>_d5_validation/, leaving the original D4/D5 record untouched.
_TAG = os.environ.get("FLASH_D4_RUN_TAG", "").strip()
D4_DIR = ROOT / "models_real" / (_TAG or "stage_d4")
D4_REPORTS = ROOT / "reports" / "data_prep" / (_TAG or "stage_d4_model_training")
D3_JSON = ROOT / "reports" / "data_prep" / "stage_d3_validation_baselines" / "validation_baselines_real.json"
OUT = ROOT / "reports" / "data_prep" / (f"{_TAG}_d5_validation" if _TAG else "stage_d5_validation_evaluation")
OUT.mkdir(parents=True, exist_ok=True)
OUT_METRICS = OUT / "d5_validation_metrics.json"
OUT_META = OUT / "d5_run_metadata.json"
OUT_PROBS = OUT / "d5_val_probabilities.npz"
OUT_SHA = OUT / "d5_outputs.sha256"
OUT_FAIL = OUT / "d5_gate_failure.json"

B, SEED, ALPHA = 1000, 0, 0.05
TOL_REGIME = dict(rtol=1e-9, atol=1e-12)
TOL_D3 = 1e-12
IDENTITY_CAL = {"method": "temperature", "maps": [1.0]}
EXPECTED_N = {5: 91_800, 15: 89_100, 30: 85_050}
EXPECTED_SUPPORT = {5: (17_581, 56_517, 17_702), 15: (19_697, 49_554, 19_849), 30: (20_006, 45_035, 20_009)}
# streaming-vs-batch thresholds copied from tests/test_predictor.py
STREAM_D = dict(median_max=1e-9, frac_gt_1e6_max=0.002, coverage_min=0.9)
STREAM_A = dict(max_abs=1e-8, coverage_min=0.9)
BASELINES = ("majority", "prior", "momentum")


class GateFailure(Exception):
    pass


# ---------------------------------------------------------------------------------------------- guards

class _NoTestDataset(X.Dataset):
    """Dataset whose rows() refuses anything except train/val."""

    def rows(self, split, h):
        if split not in ("train", "val"):
            raise RuntimeError(f"D5: access to split {split!r} is prohibited")
        return super().rows(split, h)


_orig_bl_rows = BL.rows


def _guarded_bl_rows(t, split, h):
    """Runtime wrapper for evaluation.baselines.rows (used by Baselines.fit); source file is untouched."""
    if split not in ("train", "val"):
        raise RuntimeError(f"D5: baselines access to split {split!r} is prohibited")
    return _orig_bl_rows(t, split, h)


BL.rows = _guarded_bl_rows


class _CacheOnlyEncoder:
    device = "not loaded (cache-only guard)"

    def encode(self, headlines):
        raise RuntimeError(f"D5: {len(headlines)} headline(s) missing from finbert_cache.npz; refusing to run FinBERT")


# --------------------------------------------------------------------------------------------- helpers

def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def softmax_rows(Z: np.ndarray) -> np.ndarray:
    """Same numerics as Predictor's A path: exp(z - max) / sum."""
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def prob_check(P: np.ndarray, n: int) -> dict:
    s = P.sum(axis=1)
    out = {"shape_ok": P.shape == (n, 3), "nan_rows": int(np.isnan(P).any(1).sum()),
           "inf_rows": int(np.isinf(P).any(1).sum()), "negative_rows": int((P < 0).any(1).sum()),
           "gt1_rows": int((P > 1 + 1e-12).any(1).sum()), "bad_sum_rows(|sum-1|>1e-6)": int((np.abs(s - 1) > 1e-6).sum()),
           "max_abs_sum_err": float(np.nanmax(np.abs(s - 1))) if n else 0.0}
    out["valid"] = out["shape_ok"] and all(out[k] == 0 for k in
                                           ("nan_rows", "inf_rows", "negative_rows", "gt1_rows", "bad_sum_rows(|sum-1|>1e-6)"))
    return out


def finiteness(Xs: np.ndarray, names: list[str]) -> dict:
    bad = ~np.isfinite(Xs)
    per = {names[j]: {"nan": int(np.isnan(Xs[:, j]).sum()), "inf": int(np.isinf(Xs[:, j]).sum())}
           for j in range(Xs.shape[1]) if bad[:, j].any()}
    return {"rows_with_nonfinite": int(bad.any(1).sum()), "features_with_nonfinite": per}


def pred_distribution(P: np.ndarray) -> dict:
    c = np.bincount(P.argmax(1), minlength=3)
    return {"argmax_counts": {n: int(c[i]) for i, n in enumerate(CLASS_NAMES)},
            "argmax_share": {n: round(float(c[i] / c.sum()), 6) for i, n in enumerate(CLASS_NAMES)},
            "mean_prob": {n: float(P[:, i].mean()) for i, n in enumerate(CLASS_NAMES)}}


def rowwise_losses(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row terms whose MEAN equals full_report's log_loss and brier_multiclass exactly."""
    nll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))
    br = ((P - np.eye(3)[y]) ** 2).sum(axis=1)
    return nll, br


def logloss_bootstrap(P, y, day, P_ref=None) -> dict:
    """Supplement to block_bootstrap (which has no log loss). Replicates its EXACT day sampling
    (default_rng(SEED), rng.choice(days, len(days), replace=True) once per replicate) so replicate i uses the
    same days as block_bootstrap's replicate i. Brier is recomputed here as an alignment check."""
    rng = np.random.default_rng(SEED)
    days = np.unique(day)
    by_day = {d: np.flatnonzero(day == d) for d in days}
    nll, br = rowwise_losses(P, y)
    nll_r = rowwise_losses(P_ref, y)[0] if P_ref is not None else None
    ll, bb, dd = [], [], []
    for _ in range(B):
        idx = np.concatenate([by_day[d] for d in rng.choice(days, len(days), replace=True)])
        v = float(nll[idx].mean())
        ll.append(v)
        bb.append(float(br[idx].mean()))
        if nll_r is not None:
            dd.append(v - float(nll_r[idx].mean()))
    q = [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]                     # nanpercentile: same as block_bootstrap
    out = {"log_loss_ci": [float(x) for x in np.nanpercentile(ll, q)],
           "_brier_ci_alignment_check": [float(x) for x in np.nanpercentile(bb, q)]}
    if nll_r is not None:
        out["log_loss_diff_vs_ref_ci"] = [float(x) for x in np.nanpercentile(dd, q)]
    return out


def close(a, b, tol) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=tol))


# ------------------------------------------------------------------------------------------------ gates

def gate_artifacts() -> dict:
    inv = json.loads((D4_REPORTS / "d4_artifact_inventory.json").read_text(encoding="utf-8"))
    res = {}
    for a in inv:
        got = sha256(ROOT / a["path"])
        res[a["path"]] = {"expected": a["sha256"], "actual": got, "ok": got == a["sha256"]}
    if not all(v["ok"] for v in res.values()):
        raise GateFailure(f"G2 D4 artifact hash mismatch: {res}")
    return res


def gate_bundle(bundle: dict, cfg: dict, d4meta: dict) -> dict:
    hs = cfg["labels"]["horizons"]
    chk = {"horizons": bundle["horizons"] == hs, "cfg_equal_real_yaml": bundle["cfg"] == cfg,
           "n_cols_42": len(bundle["col_names"]) == 42}
    for h in hs:
        d, a = bundle["D"][h], bundle["A"][h]
        md, ma = d4meta["models"]["D"][str(h)], d4meta["models"]["A"][str(h)]
        txt = (D4_DIR / f"lgbm_D_h{h}.txt").read_text(encoding="utf-8")
        bst = load_booster(d["model"])
        chk[f"h{h}"] = {
            "D_cols_full_range": d["cols"] == list(range(len(bundle["col_names"]))),
            "D_feature_names_match_d4": [bundle["col_names"][c] for c in d["cols"]] == md["features"],
            "D_model_equals_txt_file": d["model"] == txt,
            "D_booster_iterations": int(bst.current_iteration()), "D_best_iteration_d4": md["best_iteration"],
            "D_iterations_match": int(bst.current_iteration()) == md["best_iteration"],
            "D_booster_num_feature_42": int(bst.num_feature()) == 42,
            "D_logit_W_shape": list(d["logit"]["W"].shape), "D_logit_b_shape": list(np.shape(d["logit"]["b"])),
            "D_logit_w": float(d["logit"]["w"]), "D_logit_w_is_cfg": float(d["logit"]["w"]) == cfg["fusion"]["blend_logit_weight"],
            "A_W_shape": list(a["W"].shape), "A_b_shape": list(np.shape(a["b"])),
            "cal_identity": d["cal"] == IDENTITY_CAL and a["cal"] == IDENTITY_CAL,
            "A_n_features_d4": ma["n_features"],
        }
        c = chk[f"h{h}"]
        c["ok"] = (c["D_cols_full_range"] and c["D_feature_names_match_d4"] and c["D_model_equals_txt_file"]
                   and c["D_iterations_match"] and c["D_booster_num_feature_42"] and c["D_logit_W_shape"] == [3, 42]
                   and c["D_logit_b_shape"] == [3] and c["D_logit_w_is_cfg"] and c["A_W_shape"] == [3, ma["n_features"]]
                   and c["A_b_shape"] == [3] and c["cal_identity"])
    ok = chk["horizons"] and chk["cfg_equal_real_yaml"] and chk["n_cols_42"] and all(chk[f"h{h}"]["ok"] for h in hs)
    if not ok:
        raise GateFailure(f"G3 bundle structure/params mismatch: {chk}")
    return chk


def gate_regime(ds, bundle: dict) -> dict:
    pe, he = ds.regime.pca.export(), ds.regime.hmm.export()
    res = {"col_names_equal": ds.col_names == bundle["col_names"], "groups_equal": ds.groups == bundle["groups"],
           "hmm_names_equal": he["names"] == bundle["hmm"]["names"]}
    for k in ("mean", "scale", "P", "c"):
        a, b = np.asarray(pe[k]), np.asarray(bundle["pca"][k])
        res[f"pca_{k}"] = {"shape_equal": a.shape == b.shape,
                           "allclose": bool(a.shape == b.shape and np.allclose(a, b, **TOL_REGIME)),
                           "max_abs_diff": float(np.max(np.abs(a - b))) if a.shape == b.shape else None}
    res["pca_clip_equal"] = float(pe["clip"]) == float(bundle["pca"]["clip"])
    for k in ("startprob", "transmat", "means", "covars"):
        a, b = np.asarray(he[k]), np.asarray(bundle["hmm"][k])
        res[f"hmm_{k}"] = {"shape_equal": a.shape == b.shape,
                           "allclose": bool(a.shape == b.shape and np.allclose(a, b, **TOL_REGIME)),
                           "max_abs_diff": float(np.max(np.abs(a - b))) if a.shape == b.shape else None}
    ok = all(v["allclose"] if isinstance(v, dict) else v for v in res.values())
    res["tolerance"] = TOL_REGIME
    if not ok:
        raise GateFailure(f"G4 PCA/HMM/feature-layout reproducibility failed: {res}")
    return res


def gate_val_rows(ds, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    t = ds.table
    va = ds.rows("val", h)
    y = ds.y(va, h)
    sp = t["split"].to_numpy()[va]
    day = t["day_idx"].to_numpy()[va]
    val_days = np.unique(t["day_idx"].to_numpy()[t["split"].to_numpy() == "val"])
    sup = tuple(int(v) for v in np.bincount(y, minlength=3))
    res = {"n": int(len(va)), "expected_n": EXPECTED_N[h], "support": sup, "expected_support": EXPECTED_SUPPORT[h],
           "all_split_val": bool((sp == "val").all()), "all_valid_x": bool(t["valid_x"].to_numpy()[va].all()),
           "all_labels_ge0": bool((t[f"y{h}"].to_numpy()[va] >= 0).all()),
           "day_range": [int(day.min()), int(day.max())], "val_day_range": [int(val_days.min()), int(val_days.max())],
           "n_days": int(len(np.unique(day))), "sorted_unique_indices": bool(np.all(np.diff(va) > 0))}
    ok = (res["n"] == EXPECTED_N[h] and sup == EXPECTED_SUPPORT[h] and res["all_split_val"] and res["all_valid_x"]
          and res["all_labels_ge0"] and res["day_range"][0] >= res["val_day_range"][0]
          and res["day_range"][1] <= res["val_day_range"][1] and res["n_days"] == 27 and res["sorted_unique_indices"])
    if not ok:
        raise GateFailure(f"G5 validation rows/supports mismatch at h={h}: {res}")
    return va, y, day, res


def compare_d3(name: str, r: dict, d3: dict, h: int) -> dict:
    """Point metrics of a recomputed baseline vs the saved D3 JSON (bootstrap not recomputed)."""
    diffs, ok = {}, True
    for k in ("n", "macro_f1", "auc_ovr", "brier", "ece", "accuracy", "log_loss"):
        d = abs(float(r[k]) - float(d3[k]))
        diffs[k] = d
        ok &= d <= TOL_D3
    pc_ok = all(close([r["per_class"][c][m] for m in ("precision", "recall", "f1", "support")],
                      [d3["per_class"][c][m] for m in ("precision", "recall", "f1", "support")], TOL_D3)
                for c in CLASS_NAMES)
    cm_ok = r["confusion_matrix[true x pred]"] == d3["confusion_matrix[true x pred]"]
    rel_ok = json.loads(json.dumps(r["reliability"])) == d3["reliability"]
    ok = bool(ok and pc_ok and cm_ok and rel_ok)
    return {"baseline": name, "h": h, "abs_diffs": diffs, "per_class_ok": pc_ok, "confusion_ok": cm_ok,
            "reliability_ok": rel_ok, "ok": ok}


def stream_check(ds, bars, bundle, batch: dict) -> dict:
    """One complete VALIDATION day replayed through the existing streaming Predictor (modes A and D), compared
    with the batch reconstruction. Mirrors tests/test_predictor.py. Event raw vectors come from the batch event
    table (as in the test); event_memory.npz is not needed because history features are already in the events."""
    t = ds.table
    bars_sorted = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    if not ((bars_sorted["symbol"].to_numpy() == t["symbol"].to_numpy()).all()
            and (bars_sorted["timestamp"].to_numpy() == t["timestamp"].to_numpy()).all()):
        raise GateFailure("G8 bars_sorted is not row-aligned with ds.table")
    sp, dix = t["split"].to_numpy(), t["day_idx"].to_numpy()
    d0 = int(np.unique(dix[sp == "val"]).min())                      # first validation day (fixed rule)
    day_rows = np.flatnonzero((dix == d0) & (sp == "val"))
    if not (sp[day_rows] == "val").all():
        raise GateFailure("G8 replay rows not all validation")
    day_end = t["timestamp"].to_numpy()[day_rows].max()
    mret = market_return(bars_sorted)
    ev = ds.base.ev
    order = np.argsort(ev["timestamp"].to_numpy(), kind="stable")
    e = ev.iloc[order]
    keep = (e["timestamp"].to_numpy() <= day_end)                    # nothing after the replayed day is passed
    R = raw_matrix(ev)[order][keep]
    e = e[keep]
    events = (e["timestamp"].tolist(), e["symbol"].tolist(), e["tmin"].to_numpy(), R)
    tmin = t["tmin"].to_numpy()
    res = {"day_idx": d0, "date": str(t["timestamp"].iloc[day_rows[0]].date()), "bars_replayed": int(len(day_rows)),
           "events_available": int(len(e)), "thresholds": {"D": STREAM_D, "A": STREAM_A}}
    ok_all = True
    for mode in ("A", "D"):
        got = dict(replay(Predictor(bundle, mode), bars_sorted, mret, tmin, day_rows, events if mode == "D" else None))
        res[mode] = {"predictions_emitted": len(got)}
        for j, h in enumerate(bundle["horizons"]):
            va = batch[h]["va"]
            in_day = va[(dix[va] == d0)]
            pos = {int(r): i for i, r in enumerate(va)}
            common = [r for r in in_day if int(r) in got]
            cov = len(common) / max(len(in_day), 1)
            live = np.array([got[int(r)].probs[j] for r in common]) if common else np.zeros((0, 3))
            ref = batch[h]["P_" + mode][[pos[int(r)] for r in common]] if common else np.zeros((0, 3))
            err = np.abs(live - ref).max(axis=1) if common else np.array([np.inf])
            m = {"batch_rows_that_day": int(len(in_day)), "compared": len(common), "coverage": round(cov, 4),
                 "max_err": float(err.max()), "median_err": float(np.median(err)),
                 "frac_err_gt_1e-6": float((err > 1e-6).mean()), "live_nonfinite_rows": int((~np.isfinite(live)).any(1).sum())}
            if mode == "D":
                m["ok"] = bool(cov >= STREAM_D["coverage_min"] and m["median_err"] < STREAM_D["median_max"]
                               and m["frac_err_gt_1e-6"] < STREAM_D["frac_gt_1e6_max"])
            else:
                m["ok"] = bool(cov >= STREAM_A["coverage_min"] and m["max_err"] < STREAM_A["max_abs"])
            ok_all = ok_all and m["ok"]
            res[mode][f"h{h}"] = m
    res["ok"] = bool(ok_all)
    if not ok_all:
        raise GateFailure(f"G8 streaming Predictor vs batch mismatch: {res}")
    return res


# ------------------------------------------------------------------------------------------------- main

def main() -> int:
    t_all = time.time()
    meta = {"stage": "D5 validation-only evaluation", "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "driver": "reports/data_prep/stage_d5_validation_evaluation/d5_validation_evaluation.py",
            "versions": {"python": platform.python_version(), "lightgbm": lightgbm.__version__,
                         "sklearn": sklearn.__version__, "numpy": np.__version__},
            "bootstrap": {"B": B, "seed": SEED, "alpha": ALPHA, "block": "trading day"},
            "data_scope": ("prepare() processes the FULL chronological dataset in memory incl. test-period rows; ONLY "
                           "validation rows are predicted/scored; train rows only inside Baselines.fit (D3 procedure); "
                           "no test rows predicted, calibrated or scored"),
            "probabilities": "UNCALIBRATED (D4 identity calibrator); no calibration, threshold or gate tuning",
            "validation_bias": ("validation already used for PCA n_components, LightGBM early stopping (D) and D3; "
                                "results are NOT an independent generalisation estimate"),
            "gates": {}}
    caught: list[str] = []
    warnings.simplefilter("default")

    def _record(message, category, filename, lineno, file=None, line=None):
        s = f"{category.__name__}: {message} ({Path(filename).name}:{lineno})"
        caught.append(s)
        log("WARNING " + s)
    warnings.showwarning = _record

    # G1 environment / no overwrite
    if os.environ.get("FLASH_TEST_UNLOCK"):
        raise SystemExit("D5: FLASH_TEST_UNLOCK is set; refusing to run")
    for p in (OUT_METRICS, OUT_META, OUT_PROBS, OUT_SHA, OUT_FAIL):
        if p.exists():
            raise SystemExit(f"D5: {p} already exists; refusing to overwrite")

    try:
        meta["gates"]["G2_d4_artifacts"] = gate_artifacts()
        cfg = load_config(CONFIG)
        d4meta = json.loads((D4_REPORTS / "d4_training_metadata.json").read_text(encoding="utf-8"))
        d3 = json.loads(D3_JSON.read_text(encoding="utf-8"))
        meta["inputs"] = {"d3_json_sha256": sha256(D3_JSON),
                          "d4_metadata_sha256": sha256(D4_REPORTS / "d4_training_metadata.json")}
        bundle = joblib.load(D4_DIR / "bundle.joblib")              # trusted local D4 artifact (sha checked in G2)
        meta["gates"]["G3_bundle"] = gate_bundle(bundle, cfg, d4meta)
        log("G2/G3 passed: D4 artifacts intact, bundle structure matches D4 metadata")

        bars, news = get_data(cfg, MARKET, NEWS)
        fb_before = sha256(FINBERT_CACHE)
        ds = X.prepare(cfg, bars, news, _CacheOnlyEncoder(), FINBERT_CACHE)
        ds.__class__ = _NoTestDataset
        if sha256(FINBERT_CACHE) != fb_before:
            raise GateFailure("FinBERT cache changed during prepare()")
        meta["finbert_cache_sha256"] = fb_before
        t = ds.table
        meta["split_days"] = {s: int(len(np.unique(t["day_idx"].to_numpy()[t["split"].to_numpy() == s])))
                              for s in ("train", "val")}                    # train/val day counts only
        log(f"prepare(): XF {ds.XF.shape}; split day counts {meta['split_days']}")
        meta["gates"]["G4_regime_reproducibility"] = gate_regime(ds, bundle)
        log("G4 passed: refitted PCA/HMM equal the D4 bundle")

        cols_A = X.variant_cols(ds, "A")
        if [ds.col_names[c] for c in cols_A] != d4meta["models"]["A"]["5"]["features"]:
            raise GateFailure("G3b model-A feature list differs from D4 metadata")

        # predictions (validation rows only)
        batch, G5, G6, G7 = {}, {}, {}, {}
        for h in cfg["labels"]["horizons"]:
            va, y, day, G5[h] = gate_val_rows(ds, h)
            Xv = ds.XF[va]
            fin = {"A_features": finiteness(Xv[:, cols_A], [ds.col_names[c] for c in cols_A]),
                   "D_features": finiteness(Xv, ds.col_names)}
            a, d = bundle["A"][h], bundle["D"][h]
            P_A = softmax_rows(Xv[:, cols_A] @ a["W"].T + a["b"])
            lg = d["logit"]
            P_lgbm = load_booster(d["model"]).predict(Xv)
            P_logit = softmax_rows(np.nan_to_num(Xv) @ lg["W"].T + lg["b"])   # as FusionBlend.predict_proba
            P_D = (1.0 - lg["w"]) * P_lgbm + lg["w"] * P_logit
            bl = BL.Baselines().fit(t, h)                                     # train rows only (D3 procedure)
            if not np.array_equal(_orig_bl_rows(t, "val", h), va):
                raise GateFailure(f"G5b baseline val rows differ from Dataset val rows at h={h}")
            P_base = bl.predict(t, va)
            G6[h] = {"finiteness": fin, "P_A": prob_check(P_A, len(va)), "P_lgbm": prob_check(P_lgbm, len(va)),
                     "P_logit": prob_check(P_logit, len(va)), "P_D": prob_check(P_D, len(va)),
                     **{f"P_{n}": prob_check(P_base[n], len(va)) for n in BASELINES}}
            if not all(v["valid"] for k, v in G6[h].items() if k.startswith("P_")):
                raise GateFailure(f"G6 invalid probabilities at h={h}: {G6[h]}")
            batch[h] = {"va": va, "y": y, "day": day, "P_A": P_A, "P_D": P_D, **{f"P_{n}": P_base[n] for n in BASELINES},
                        "fit_on_train": bl.describe()}
            # G7: recomputed baselines must reproduce D3 point metrics
            reps = {n: full_report(P_base[n], y) for n in BASELINES}
            G7[h] = {n: compare_d3(n, reps[n], d3["horizons"][str(h)]["models"][n], h) for n in BASELINES}
            G7[h]["fit_on_train_equal"] = json.loads(json.dumps(bl.describe())) == d3["horizons"][str(h)]["fit_on_train"]
            if not (all(G7[h][n]["ok"] for n in BASELINES) and G7[h]["fit_on_train_equal"]):
                raise GateFailure(f"G7 D3 baseline reproduction failed at h={h}: {G7[h]}")
            batch[h]["base_reports"] = reps
            log(f"h={h}: G5 rows OK (n={len(va):,}), G6 probabilities valid, G7 D3 baselines reproduced")
        meta["gates"].update({"G5_val_rows": G5, "G6_finiteness_and_probabilities": G6, "G7_d3_reproduction": G7})

        meta["gates"]["G8_streaming_crosscheck"] = stream_check(ds, bars, bundle, batch)
        log("G8 passed: one-day streaming Predictor matches batch reconstruction")

        # save probabilities before the long bootstrap (validation rows only)
        npz = {}
        ts_ns = pd.DatetimeIndex(t["timestamp"]).asi8                     # UTC epoch ns (tz-aware safe)
        for h, b in batch.items():
            npz.update({f"h{h}_row_index": b["va"], f"h{h}_y": b["y"], f"h{h}_day_idx": b["day"],
                        f"h{h}_symbol": t["symbol"].to_numpy()[b["va"]].astype(str),
                        f"h{h}_timestamp_ns": ts_ns[b["va"]],
                        f"h{h}_P_A": b["P_A"], f"h{h}_P_D": b["P_D"], **{f"h{h}_P_{n}": b[f"P_{n}"] for n in BASELINES}})
        np.savez_compressed(OUT_PROBS, **npz)
        log(f"wrote {OUT_PROBS.relative_to(ROOT)}")
    except Exception as g:                                     # gate failure OR any other pre-metric error
        return _fail(meta, caught, g, "before metrics (no metrics computed)", 2 if isinstance(g, GateFailure) else 3)

    try:
        return _metrics_phase(meta, caught, batch, t_all)
    except Exception as g:                                     # incl. alignment checks and final G2 re-check
        return _fail(meta, caught, g, "during metrics/bootstrap (d5_val_probabilities.npz is a PARTIAL-RUN output)", 4)


def _fail(meta: dict, caught: list, err: Exception, phase: str, code: int) -> int:
    """Record the failure; never deletes anything. Completed runs are identified ONLY by
    d5_outputs.sha256 + status == 'complete' in d5_run_metadata.json / d5_validation_metrics.json."""
    import traceback
    meta["status"] = f"FAILED {phase}"
    meta["failure_type"] = "gate_failure" if isinstance(err, GateFailure) else type(err).__name__
    meta["failure"] = str(err)
    meta["traceback"] = traceback.format_exc()
    meta["warnings"] = caught
    OUT_FAIL.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8", newline="\n")
    log(f"FAILED {phase} -> {OUT_FAIL.relative_to(ROOT).as_posix()}: {str(err)[:2000]}")
    return code


def _metrics_phase(meta: dict, caught: list, batch: dict, t_all: float) -> int:
    results = {"status": "incomplete", "split": "val", "probabilities": "uncalibrated", "horizons": {}}
    n_cmp = 0
    for h, b in batch.items():
        y, day = b["y"], b["day"]
        if not (len(b["P_A"]) == len(b["P_D"]) == len(y) == len(day) == len(b["va"])
                and all(len(b[f"P_{n}"]) == len(y) for n in BASELINES)):
            raise RuntimeError(f"D5: array alignment failure at h={h}")
        hr = {"n": int(len(y)), "n_days": int(len(np.unique(day))), "baselines_fit_on_train": b["fit_on_train"],
              "baselines": {}, "models": {}}
        for n in BASELINES:
            hr["baselines"][n] = {**{k: v for k, v in b["base_reports"][n].items()},
                                  "prediction_distribution": pred_distribution(b[f"P_{n}"]),
                                  "bootstrap_ci_from_d3": "see D3 JSON (not recomputed)"}
        for m in ("A", "D"):
            P = b[f"P_{m}"]
            t0 = time.time()
            r = full_report(P, y)
            r["prediction_distribution"] = pred_distribution(P)
            refs = {n: b[f"P_{n}"] for n in BASELINES}
            if m == "D":
                refs["model_A"] = b["P_A"]
            r["paired_vs"] = {}
            for rn, Pref in refs.items():
                bb = block_bootstrap(P, y, day, B=B, seed=SEED, P_ref=Pref, alpha=ALPHA)
                ll = logloss_bootstrap(P, y, day, P_ref=Pref)
                if not close(ll["_brier_ci_alignment_check"], bb["ci"]["brier"], 1e-12):
                    raise RuntimeError(f"D5: bootstrap alignment check failed (h={h}, {m} vs {rn})")
                r.setdefault("bootstrap_ci", {**bb["ci"], "log_loss": ll["log_loss_ci"], "n_days": bb["n_days"], "B": B})
                r["paired_vs"][rn] = {"diff_ci(model-ref)": {**bb["diff_vs_ref_ci"], "log_loss": ll["log_loss_diff_vs_ref_ci"]},
                                      "point_diff(model-ref)": {
                                          "brier": r["brier"] - brier_multiclass(Pref, y),
                                          "log_loss": r["log_loss"] - float(rowwise_losses(Pref, y)[0].mean()),
                                          "auc_ovr": r["auc_ovr"] - full_report(Pref, y)["auc_ovr"]}}
                n_cmp += 1
            hr["models"][m] = r
            log(f"h={h} {m}: brier {r['brier']:.4f} logloss {r['log_loss']:.4f} auc {r['auc_ovr']:.4f} "
                f"[{time.time() - t0:.0f}s incl. bootstrap]")
        results["horizons"][h] = hr
    results["multiple_comparisons"] = {"paired_comparisons": n_cmp, "metrics_per_comparison": 6,
                                       "note": "exploratory; no multiplicity correction applied"}

    meta["warnings"] = caught
    meta["gates"]["G2_d4_artifacts_after"] = gate_artifacts()
    meta["finbert_cache_unchanged_end"] = sha256(FINBERT_CACHE) == meta["finbert_cache_sha256"]
    if not meta["finbert_cache_unchanged_end"]:
        raise RuntimeError("D5: FinBERT cache changed during the run")
    meta["total_seconds"] = round(time.time() - t_all, 1)
    results["status"] = meta["status"] = "complete"                # only reached when every step succeeded
    OUT_METRICS.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8", newline="\n")
    OUT_META.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8", newline="\n")
    files = [OUT_METRICS, OUT_META, OUT_PROBS]
    OUT_SHA.write_text("".join(f"{sha256(p)} *{p.relative_to(ROOT).as_posix()}\n" for p in files),
                       encoding="utf-8", newline="\n")
    for p in files:
        log(f"wrote {p.relative_to(ROOT).as_posix()}  {p.stat().st_size:,} B")
    log(f"done in {meta['total_seconds']:.0f}s  warnings={len(caught)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
