"""Unit 2: multinomial logistic regression (L2) -- the BASELINE only.

MLE view: lbfgs maximises the multinomial log-likelihood minus an L2 penalty (= MAP with a Gaussian prior).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class BaselineLogistic:
    def __init__(self, C: float = 1.0, max_iter: int = 500):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=max_iter)   # L2, multinomial (lbfgs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineLogistic":
        self.clf.fit(self.scaler.fit_transform(X), y)             # scaler fitted on train only
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(X))

    def export(self) -> dict:
        """Plain arrays for the NumPy hot path: p = softmax(((x - mean) / std) @ W.T + b)."""
        W = self.clf.coef_ / self.scaler.scale_[None, :]
        b = self.clf.intercept_ - W @ self.scaler.mean_
        return {"W": W.astype(float), "b": b.astype(float)}
