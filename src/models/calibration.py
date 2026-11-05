"""Probability calibration (Unit 1: calibrated probabilities).

One-vs-rest isotonic regression (default) or Platt scaling per class, fitted on the VALIDATION set, followed
by renormalisation so the three probabilities sum to 1. `export()` gives plain arrays for the NumPy hot path
(isotonic == piecewise-linear np.interp on the fitted thresholds).
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method: str = "isotonic", n_classes: int = 3):
        assert method in ("isotonic", "platt", "temperature")
        self.method, self.C = method, n_classes
        self.maps: list = []

    def fit(self, P: np.ndarray, y: np.ndarray) -> "Calibrator":
        self.maps = []
        if self.method == "temperature":
            from scipy.optimize import minimize_scalar
            from scipy.special import softmax

            L = np.log(np.clip(P, 1e-9, 1))
            nll = lambda T: -np.log(np.clip(softmax(L / T, axis=1)[np.arange(len(y)), y], 1e-12, 1)).mean()  # noqa: E731
            self.maps = [float(minimize_scalar(nll, bounds=(0.5, 3.0), method="bounded").x)]
            return self
        for c in range(self.C):
            t = (y == c).astype(float)
            if self.method == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip", y_min=_EPS, y_max=1 - _EPS).fit(P[:, c], t)
                self.maps.append((iso.X_thresholds_.astype(float), iso.y_thresholds_.astype(float)))
            else:
                lr = LogisticRegression(C=1e6, max_iter=200).fit(_logit(P[:, c])[:, None], t)
                self.maps.append((float(lr.coef_[0, 0]), float(lr.intercept_[0])))
        return self

    def _apply_col(self, c: int, p: np.ndarray) -> np.ndarray:
        a, b = self.maps[c]
        if self.method == "isotonic":
            return np.interp(p, a, b)
        return 1.0 / (1.0 + np.exp(-(a * _logit(p) + b)))

    def transform(self, P: np.ndarray) -> np.ndarray:
        if self.method == "temperature":
            Q = np.clip(P, 1e-9, 1) ** (1.0 / self.maps[0])
            return Q / Q.sum(axis=1, keepdims=True)
        Q = np.column_stack([self._apply_col(c, P[:, c]) for c in range(self.C)])
        return Q / Q.sum(axis=1, keepdims=True)

    def export(self) -> dict:
        return {"method": self.method, "maps": self.maps}


class FastCalibrator:
    """Hot-path version built from Calibrator.export(); no sklearn, no allocation beyond tiny arrays."""

    def __init__(self, exported: dict):
        self.method = exported["method"]
        self.maps = exported["maps"]

    def __call__(self, p: np.ndarray, out: np.ndarray) -> np.ndarray:
        if self.method == "temperature":
            np.maximum(p, 1e-9, out=out)
            np.power(out, 1.0 / self.maps[0], out=out)
        elif self.method == "isotonic":
            for c, (xs, ys) in enumerate(self.maps):
                out[c] = np.interp(p[c], xs, ys)
        else:
            for c, (a, b) in enumerate(self.maps):
                q = min(max(p[c], _EPS), 1 - _EPS)
                out[c] = 1.0 / (1.0 + np.exp(-(a * np.log(q / (1 - q)) + b)))
        out /= out.sum()
        return out
