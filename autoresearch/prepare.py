"""FROZEN evaluation harness for the FLASH autoresearch loop (do not edit in the loop).

Adapted from karpathy/autoresearch: one fixed harness + one editable file (train_model.py).

Leakage guard: the loop NEVER sees the test split. Chronological layout of trading days:
    train | val_fit (first half of val days) | val_dev (second half of val days) | test (untouched)
train_model.py may fit on `train`, early-stop / calibrate on `val_fit`, and is scored on `val_dev`.
The score is the mean calibrated multiclass log-loss over all horizons (LOWER IS BETTER).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from src.config import load_config, resolve  # noqa: E402
from src.data.build import get_data  # noqa: E402
from src.evaluation.experiments import Dataset, make_encoder, prepare  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402

TIME_BUDGET_S = 240          # hard wall-clock budget for one full run (all horizons)


@dataclass
class Split:
    h: int
    Xtr: np.ndarray
    ytr: np.ndarray
    Xfit: np.ndarray         # early stopping / calibration
    yfit: np.ndarray
    Xdev: np.ndarray         # scored
    ydev: np.ndarray
    shock_dev: np.ndarray
    meta_tr: dict
    meta_fit: dict
    meta_dev: dict


def load_dataset() -> Dataset:
    import joblib
    cp = ROOT / "autoresearch" / ".cache_ds.joblib"
    if cp.exists():
        return joblib.load(cp)
    ds = _build_dataset()
    joblib.dump(ds, cp)
    return ds


def _build_dataset() -> Dataset:
    cfg = load_config()
    bars, news = get_data(cfg)
    return prepare(cfg, bars, news, make_encoder(cfg, finbert=True), resolve(cfg, "processed") / "finbert_cache.npz")


def _meta(ds: Dataset, rows: np.ndarray) -> dict:
    t = ds.table
    return {"symbol": t["symbol"].to_numpy()[rows], "day_idx": t["day_idx"].to_numpy()[rows], "rows": rows}


def make_split(ds: Dataset, h: int) -> Split:
    t = ds.table
    tr, va = ds.rows("train", h), ds.rows("val", h)
    days = np.unique(t["day_idx"].to_numpy()[va])
    cut = days[len(days) // 2]
    d = t["day_idx"].to_numpy()[va]
    fit, dev = va[d < cut], va[d >= cut]
    X = ds.XF
    return Split(h, X[tr], ds.y(tr, h), X[fit], ds.y(fit, h), X[dev], ds.y(dev, h), ds.shock(dev),
                 _meta(ds, tr), _meta(ds, fit), _meta(ds, dev))


def run(fn, horizons=None) -> dict:
    """fn(split, ds) -> calibrated probabilities (n_dev, 3). Prints the summary in autoresearch style."""
    t0 = time.time()
    ds = load_dataset()
    t_prep = time.time() - t0
    horizons = horizons or ds.base.cfg["labels"]["horizons"]
    per = {}
    t1 = time.time()
    for h in horizons:
        sp = make_split(ds, h)
        P = np.clip(np.asarray(fn(sp, ds), dtype=float), 1e-7, 1)
        P = P / P.sum(1, keepdims=True)
        assert P.shape == (len(sp.ydev), 3), f"bad shape {P.shape}"
        m = evaluate(P, sp.ydev)
        m["logloss"] = float(log_loss(sp.ydev, P, labels=[0, 1, 2]))
        sh = sp.shock_dev
        m["auc_shock"] = evaluate(P[sh], sp.ydev[sh])["auc_ovr"]
        per[h] = m
    train_s = time.time() - t1
    val = float(np.mean([per[h]["logloss"] for h in horizons]))
    for h in horizons:
        m = per[h]
        print(f"h={h:>2} logloss {m['logloss']:.5f} auc {m['auc_ovr']:.4f} f1 {m['macro_f1']:.4f} "
              f"brier {m['brier']:.4f} ece {m['ece']:.4f} auc_shock {m['auc_shock']:.4f} n={m['n']}")
    print("---")
    print(f"val_logloss:      {val:.6f}")
    print(f"mean_auc:         {np.mean([per[h]['auc_ovr'] for h in horizons]):.6f}")
    print(f"mean_auc_shock:   {np.nanmean([per[h]['auc_shock'] for h in horizons]):.6f}")
    print(f"train_seconds:    {train_s:.1f}")
    print(f"prep_seconds:     {t_prep:.1f}")
    if train_s > TIME_BUDGET_S:
        print(f"OVER TIME BUDGET ({TIME_BUDGET_S}s)")
    return {"val_logloss": val, "per": per}
