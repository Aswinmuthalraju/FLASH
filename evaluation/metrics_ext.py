"""Extended metrics on top of src/evaluation/metrics.py (whose functions are reused unchanged)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from src.evaluation.metrics import CLASS_NAMES, brier_multiclass, conf_matrix, ece, evaluate, reliability_curve


def full_report(P: np.ndarray, y: np.ndarray) -> dict:
    out = evaluate(P, y)                                    # n, macro_f1, auc_ovr, brier, ece, accuracy
    p, r, f, s = precision_recall_fscore_support(y, P.argmax(1), labels=[0, 1, 2], zero_division=0)
    out["per_class"] = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                        for i, c in enumerate(CLASS_NAMES)}
    out["confusion_matrix[true x pred]"] = conf_matrix(P, y).tolist()
    out["log_loss"] = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))
    out["reliability"] = {}
    for i, c in enumerate(CLASS_NAMES):                      # one-vs-rest reliability, bins with >= 20 rows
        xs, ys, ns = reliability_curve(P[:, i], (y == i).astype(float))
        out["reliability"][c] = {"mean_pred": xs.round(4).tolist(), "observed": ys.round(4).tolist(), "n": ns.tolist()}
    return out


_POINT = {"macro_f1": lambda P, y: evaluate(P, y)["macro_f1"], "auc_ovr": lambda P, y: evaluate(P, y)["auc_ovr"],
          "brier": brier_multiclass, "ece": ece, "accuracy": lambda P, y: float((P.argmax(1) == y).mean())}


def block_bootstrap(P: np.ndarray, y: np.ndarray, day: np.ndarray, B: int = 1000, seed: int = 0,
                    P_ref: np.ndarray | None = None, alpha: float = 0.05) -> dict:
    """Day-level block bootstrap: resample whole trading days (all symbols/bars of a day together), which keeps
    intraday autocorrelation and cross-stock correlation inside each block. With P_ref, also CI of (P - P_ref)."""
    rng = np.random.default_rng(seed)
    days = np.unique(day)
    by_day = {d: np.flatnonzero(day == d) for d in days}
    stats = {k: [] for k in _POINT}
    diffs = {k: [] for k in _POINT} if P_ref is not None else None
    for _ in range(B):
        idx = np.concatenate([by_day[d] for d in rng.choice(days, len(days), replace=True)])
        for k, f in _POINT.items():
            v = f(P[idx], y[idx])
            stats[k].append(v)
            if diffs is not None:
                diffs[k].append(v - f(P_ref[idx], y[idx]))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    out = {"n_days": int(len(days)), "B": B,
           "ci": {k: [float(x) for x in np.nanpercentile(v, q)] for k, v in stats.items()}}
    if diffs is not None:
        out["diff_vs_ref_ci"] = {k: [float(x) for x in np.nanpercentile(v, q)] for k, v in diffs.items()}
    return out
