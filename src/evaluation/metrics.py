"""Classification metrics + plots. Classes: 0=DOWN, 1=NEUTRAL, 2=UP."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]


def brier_multiclass(P: np.ndarray, y: np.ndarray) -> float:
    Y = np.eye(P.shape[1])[y]
    return float(((P - Y) ** 2).sum(axis=1).mean())


def ece(P: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the top-class confidence."""
    conf, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(err)


def evaluate(P: np.ndarray, y: np.ndarray) -> dict:
    if len(y) == 0:
        return {k: float("nan") for k in ("n", "macro_f1", "auc_ovr", "brier", "ece", "accuracy")}
    pred = P.argmax(1)
    try:
        auc = float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=[0, 1, 2]))
    except ValueError:          # a class missing from this subset
        auc = float("nan")
    return {"n": int(len(y)), "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2])),
            "auc_ovr": auc, "brier": brier_multiclass(P, y), "ece": ece(P, y), "accuracy": float((pred == y).mean())}


def conf_matrix(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    return confusion_matrix(y, P.argmax(1), labels=[0, 1, 2])


def reliability_curve(p: np.ndarray, t: np.ndarray, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            xs.append(p[m].mean()); ys.append(t[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def plot_calibration(curves: dict[str, tuple[np.ndarray, np.ndarray]], y: np.ndarray, path, title: str = ""):
    """curves: name -> (P_uncalibrated or P, P_calibrated) is not required; pass name -> P (n,3)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for c, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        for name, P in curves.items():
            xs, ys, _ = reliability_curve(P[:, c], (y == c).astype(float))
            ax.plot(xs, ys, marker="o", ms=4, label=name)
        ax.set_title(f"P({CLASS_NAMES[c]})"); ax.set_xlabel("predicted"); ax.grid(alpha=.3)
    axes[0].set_ylabel("observed frequency"); axes[0].legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_confusion(cm: np.ndarray, path, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES); ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
