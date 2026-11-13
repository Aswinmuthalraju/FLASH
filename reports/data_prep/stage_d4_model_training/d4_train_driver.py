"""FLASH Stage D4 -- isolated, test-free training driver (real data).

Reuses the UNMODIFIED FLASH components:
    src.evaluation.experiments.prepare / variant_cols   (features, FinBERT cache, PCA -> HMM regime)
    src.models.baseline_logistic.BaselineLogistic       (model A: standalone L2 logistic, price features)
    src.models.fusion_lgbm.FusionBlend                  (model D: LightGBM + configured L2-logit blend)
    src.inference.predictor.make_bundle                 (existing inference bundle format)
    src.features.news_features.memory_from_events       (event memory for the slow path)

Data scope (read this first):
  * prepare() processes the FULL chronological dataset IN MEMORY -- features, labels, HMM forward
    filtering and event forward returns are computed for every row, INCLUDING test-period rows.
    This is inherent to the existing pipeline (the train/val/test day split is derived from all days).
  * Predictive model fitting (A, and D's LightGBM + logistic leg) uses TRAIN rows only.
  * VAL rows are used only for (a) LightGBM early stopping and (b) the existing RegimePCA
    n_components selection inside prepare().
  * TEST rows are never used for model fitting, prediction, calibration or scoring.

Differences from scripts/train.py (deliberate, D4 scope):
  * NEVER selects, slices, predicts or scores TEST rows (fit_variant() / score() are not used).
  * NO calibration is fitted. The bundle carries an explicit IDENTITY calibrator
    (temperature, T = 1.0) only so the existing Predictor can load it.
  * Validation rows are used ONLY as LightGBM's early-stopping eval_set (inside FusionBlend.fit).
  * Event memory keeps only events whose outcome was known by the end of the VAL period.
  * Artifacts go to models_real/stage_d4/ (refuses to run if it already exists); never touches models/.
  * FinBERT is never loaded: a guard encoder aborts if any headline is missing from the existing cache,
    so finbert_cache.npz is read-only for this run.

Run (from the repo root):
    .venv/Scripts/python.exe reports/data_prep/stage_d4_model_training/d4_train_driver.py
    FLASH_D4_RUN_TAG=stage_d4_mac .venv/bin/python reports/data_prep/stage_d4_model_training/d4_train_driver.py
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
import sklearn  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.build import get_data  # noqa: E402
from src.evaluation import experiments as X  # noqa: E402
from src.features.market_features import CLASS_NAMES  # noqa: E402
from src.features.news_features import memory_from_events  # noqa: E402
from src.inference.predictor import make_bundle  # noqa: E402
from src.models.baseline_logistic import BaselineLogistic  # noqa: E402
from src.models.fusion_lgbm import FusionBlend  # noqa: E402

CONFIG = ROOT / "configs" / "real.yaml"
MARKET = "data/raw/market"
NEWS = "data/raw/news.csv"
FINBERT_CACHE = ROOT / "data" / "processed_real" / "finbert_cache.npz"
# Optional run tag: FLASH_D4_RUN_TAG=stage_d4_mac writes to models_real/<tag>/ and reports/data_prep/<tag>/ instead of
# the original D4 locations, so a re-run on another machine never overwrites (or conflicts with) the D4 record.
_TAG = os.environ.get("FLASH_D4_RUN_TAG", "").strip()
MODEL_DIR = ROOT / "models_real" / (_TAG or "stage_d4")
REPORT_DIR = ROOT / "reports" / "data_prep" / (_TAG or "stage_d4_model_training")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
META_JSON = REPORT_DIR / "d4_training_metadata.json"
INVENTORY_JSON = REPORT_DIR / "d4_artifact_inventory.json"
SHA_FILE = REPORT_DIR / "d4_artifacts.sha256"
IDENTITY_CAL = {"method": "temperature", "maps": [1.0]}   # p**(1/1) renormalised == p  (NOT fitted)


# ------------------------------------------------------------------------------------------- guards

class _NoTestDataset(X.Dataset):
    """Same dataclass, but asking for TEST rows is a hard error (defence in depth; the driver never does)."""

    def rows(self, split, h):
        if split not in ("train", "val"):
            raise RuntimeError(f"D4 driver: access to split {split!r} is prohibited")
        return super().rows(split, h)


class _CacheOnlyEncoder:
    """Stands in for FinBERT. encode_cached() only calls encode() on a cache miss -- we refuse, so the
    cache file is never rewritten and FinBERT is never loaded (frozen + unused)."""
    device = "not loaded (cache-only guard)"

    def encode(self, headlines):
        raise RuntimeError(f"D4 driver: {len(headlines)} headline(s) missing from finbert_cache.npz; "
                           "refusing to run FinBERT or rewrite the cache")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def class_dist(y: np.ndarray) -> dict:
    c = np.bincount(y, minlength=3)
    return {n: {"count": int(c[i]), "share": round(float(c[i] / c.sum()), 4)} for i, n in enumerate(CLASS_NAMES)}


def log(msg: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------- main

def main() -> None:
    if os.environ.get("FLASH_TEST_UNLOCK"):
        raise SystemExit("D4 driver: FLASH_TEST_UNLOCK is set; refusing to run")
    for p in (MODEL_DIR, META_JSON, INVENTORY_JSON, SHA_FILE):
        if p.exists():
            raise SystemExit(f"D4 driver: {p} already exists; refusing to overwrite")
    if not FINBERT_CACHE.exists():
        raise SystemExit("D4 driver: FinBERT cache missing")

    t_all = time.time()
    caught: list[str] = []
    warnings.simplefilter("default")       # each distinct warning once per location (scripts/_common.py would hide all)
    _orig_show = warnings.showwarning

    def _record(message, category, filename, lineno, file=None, line=None):
        s = f"{category.__name__}: {message} ({Path(filename).name}:{lineno})"
        caught.append(s)
        log("WARNING " + s)
    warnings.showwarning = _record

    cfg = load_config(CONFIG)
    seed = cfg["seed"]
    log(f"config {CONFIG.relative_to(ROOT)}  seed={seed}  horizons={cfg['labels']['horizons']}")

    # 1) data + features (existing FLASH pipeline, unmodified)
    t0 = time.time()
    bars, news = get_data(cfg, MARKET, NEWS)
    log(f"data: {len(bars):,} bars, {bars.symbol.nunique()} symbols, {len(news):,} headlines")
    cache_sha_before = sha256(FINBERT_CACHE)
    ds = X.prepare(cfg, bars, news, _CacheOnlyEncoder(), FINBERT_CACHE)
    ds.__class__ = _NoTestDataset
    if sha256(FINBERT_CACHE) != cache_sha_before:
        raise SystemExit("D4 driver: FinBERT cache changed during prepare(); aborting before training")
    t_prep = time.time() - t0
    r = ds.regime
    log(f"prepare(): {t_prep:.0f}s  XF {ds.XF.shape}  PCA {r.pca.n_components_} comps "
        f"(val explained var {r.pca.val_explained_:.3f})  HMM states {r.names}")

    meta = {
        "stage": "D4", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "driver": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "config": "configs/real.yaml", "market": MARKET, "news": NEWS,
        "seed": seed, "horizons": cfg["labels"]["horizons"],
        "versions": {"python": platform.python_version(), "lightgbm": lightgbm.__version__,
                     "sklearn": sklearn.__version__, "numpy": np.__version__},
        "data": {"bars": int(len(bars)), "symbols": int(bars.symbol.nunique()), "headlines": int(len(news)),
                 "finbert_cache_sha256": cache_sha_before, "finbert_loaded": False},
        "regime": {"pca_components": int(r.pca.n_components_), "pca_val_explained_var": float(r.pca.val_explained_),
                   "pca_note": "existing RegimePCA selects n_components using the VAL split (known D2 caveat L8)",
                   "hmm_states": list(r.names), "hmm_fit": "train rows only (existing fit_regime)"},
        "feature_count_total": int(ds.XF.shape[1]),
        "test_split": "never selected, predicted or scored by this driver",
        "data_scope": ("prepare() processes the FULL chronological dataset in memory (features, labels, HMM "
                       "forward filtering, event forward returns), INCLUDING test-period rows; predictive model "
                       "fitting uses TRAIN rows only; VAL rows are used only for LightGBM early stopping and the "
                       "existing RegimePCA n_components selection; NO test rows are used for model fitting, "
                       "prediction, calibration or scoring"),
        "calibration": "NOT fitted in D4; bundle carries identity calibrator (temperature T=1.0)",
        "prepare_seconds": round(t_prep, 1),
        "models": {"A": {}, "D": {}},
    }

    # 2) fit per horizon -- mirrors fit_variant() minus every test/calibration line
    finals = {"A": {}, "D": {}}
    for h in cfg["labels"]["horizons"]:
        tr, va = ds.rows("train", h), ds.rows("val", h)
        ytr, yva = ds.y(tr, h), ds.y(va, h)

        # --- A: standalone L2 logistic regression on price features (train only; val unused)
        cols = X.variant_cols(ds, "A")
        t0 = time.time()
        n_warn = len(caught)
        mA = BaselineLogistic().fit(ds.XF[tr][:, cols], ytr)          # same defaults as fit_variant
        dA = time.time() - t0
        finals["A"][h] = X.Fitted("A", h, cols, "logit", mA, None, None, None, None, None, None)
        meta["models"]["A"][h] = {
            "kind": "L2 multinomial logistic regression (BaselineLogistic, StandardScaler fit on train)",
            "n_train": int(len(tr)), "n_val_used": 0, "n_features": len(cols),
            "features": [ds.col_names[c] for c in cols], "train_class_dist": class_dist(ytr),
            "params": {k: v for k, v in mA.clf.get_params().items()}, "seed": "n/a (lbfgs is deterministic)",
            "n_iter": [int(i) for i in np.atleast_1d(mA.clf.n_iter_)],
            "converged": bool(np.all(np.atleast_1d(mA.clf.n_iter_) < mA.clf.max_iter)),
            "train_seconds": round(dA, 2), "warnings": caught[n_warn:],
        }
        log(f"h={h:>2} A: n_train={len(tr):,} feats={len(cols)} n_iter={mA.clf.n_iter_} {dA:.1f}s")

        # --- D: LightGBM fusion + configured logistic blend; val = early-stopping eval_set ONLY
        cols = X.variant_cols(ds, "D")
        t0 = time.time()
        n_warn = len(caught)
        mD = FusionBlend(cfg["fusion"], seed).fit(ds.XF[tr][:, cols], ytr, ds.XF[va][:, cols], yva)
        dD = time.time() - t0
        finals["D"][h] = X.Fitted("D", h, cols, "lgbm", mD, None, None, None, None, None, None)
        gbm = mD.gbm.model
        best_score = gbm.best_score_.get("valid_0", {}).get("multi_logloss")
        meta["models"]["D"][h] = {
            "kind": "FusionBlend = (1-w)*LightGBM + w*L2-logistic (logit leg fit on train only)",
            "n_train": int(len(tr)), "n_val_used": int(len(va)),
            "val_usage": "LightGBM eval_set for early stopping (multi_logloss, patience "
                         f"{mD.gbm.early}) -- nothing else", "n_features": len(cols),
            "features": [ds.col_names[c] for c in cols],
            "train_class_dist": class_dist(ytr), "val_class_dist": class_dist(yva),
            "lgbm_params": {k: v for k, v in mD.gbm.params.items()}, "early_stopping_rounds": mD.gbm.early,
            "seed": seed, "best_iteration": int(mD.best_iteration),
            "stopped_early": bool(gbm.best_iteration_ and gbm.best_iteration_ < mD.gbm.params["n_estimators"]),
            "early_stopping_val_multi_logloss_at_best_iter": None if best_score is None else float(best_score),
            "early_stopping_note": "stopping criterion only; NOT a performance estimate",
            "blend_logit_weight": mD.w,
            "logit_leg": None if mD.logit is None else {
                "C": mD.C, "max_iter": mD.logit.clf.max_iter,
                "n_iter": [int(i) for i in np.atleast_1d(mD.logit.clf.n_iter_)],
                "converged": bool(np.all(np.atleast_1d(mD.logit.clf.n_iter_) < mD.logit.clf.max_iter))},
            "train_seconds": round(dD, 2), "warnings": caught[n_warn:],
        }
        log(f"h={h:>2} D: n_train={len(tr):,} n_val(ES)={len(va):,} feats={len(cols)} "
            f"best_iter={mD.best_iteration} {dD:.1f}s")

    # 3) bundle in the existing inference format; calibrator slots = explicit identity (not fitted)
    class _Identity:
        @staticmethod
        def export():
            return dict(IDENTITY_CAL)
    for name in ("A", "D"):
        for h in finals[name]:
            finals[name][h].calibrator = _Identity()
    bundle = make_bundle(ds, finals)
    bundle["stage_d4"] = {"calibration": "identity (not fitted)", "test_split": "untouched",
                          "event_memory": "events known by end of VAL only"}

    # 4) event memory restricted to outcomes known by the end of the validation period
    val_end = float(ds.table.loc[ds.table["split"].to_numpy() == "val", "tmin"].max())
    ka = ds.base.ev["known_at"].to_numpy()
    keep = np.isfinite(ka) & (ka <= val_end)
    mem = memory_from_events(ds.base.ev[keep].reset_index(drop=True), ds.base.E[keep], cfg["news"]["memory_k"])
    meta["event_memory"] = {"events_total": int(len(ka)), "events_kept": int(keep.sum()),
                            "cutoff_tmin": val_end, "rule": "known_at <= last VAL bar tmin"}

    # 5) write artifacts (new directory only)
    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    out = [MODEL_DIR / "bundle.joblib", MODEL_DIR / "event_memory.npz"]
    joblib.dump(bundle, out[0])
    np.savez(out[1], emb=mem.emb, fwd=mem.fwd, known_at=mem.known_at)
    for h in finals["D"]:
        p = MODEL_DIR / f"lgbm_D_h{h}.txt"
        p.write_text(finals["D"][h].model.model_string(), encoding="utf-8")
        out.append(p)
        meta["models"]["D"][h]["artifact"] = str(p.relative_to(ROOT)).replace("\\", "/")
    for name in ("A", "D"):
        for h in finals[name]:
            meta["models"][name][h]["bundle_key"] = f"bundle.joblib -> ['{name}'][{h}]"

    inv = [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha256(p)}
           for p in out]
    meta["artifacts"] = inv
    meta["warnings_all"] = caught
    meta["total_seconds"] = round(time.time() - t_all, 1)
    meta["finbert_cache_unchanged"] = sha256(FINBERT_CACHE) == cache_sha_before

    INVENTORY_JSON.write_text(json.dumps(inv, indent=2), encoding="utf-8")
    SHA_FILE.write_text("".join(f"{a['sha256']} *{a['path']}\n" for a in inv), encoding="utf-8")
    META_JSON.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    warnings.showwarning = _orig_show
    for a in inv:
        log(f"wrote {a['path']}  {a['bytes']:,} B  {a['sha256']}")
    log(f"done in {meta['total_seconds']:.0f}s  warnings={len(caught)}")


if __name__ == "__main__":
    main()
