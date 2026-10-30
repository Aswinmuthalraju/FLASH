"""Unit 3 (unsupervised): PCA compresses market features before the HMM; a GMM fitted with EM
initialises the HMM state means / covariances.

Everything is fitted on the TRAIN split only. The number of PCA components is chosen on VALIDATION:
the smallest n whose reconstruction explains >= `target_var` of the validation variance.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

CLIP = 8.0   # standardised features are clipped to +-8 sigma (robustness to news-day spikes)


class RegimePCA:
    def __init__(self, target_var: float = 0.90, use_pca: bool = True):
        self.target_var = target_var
        self.use_pca = use_pca
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.n_components_: int | None = None
        self.val_explained_: float | None = None

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.scaler.transform(X), -CLIP, CLIP)

    def fit(self, X_train: np.ndarray, X_val: np.ndarray) -> "RegimePCA":
        self.scaler.fit(X_train)
        if not self.use_pca:                                    # ablation: HMM sees standardised features
            self.n_components_ = X_train.shape[1]
            return self
        Xt, Xv = self._scale(X_train), self._scale(X_val)
        self.pca = PCA(n_components=Xt.shape[1]).fit(Xt)
        mu, comp = self.pca.mean_, self.pca.components_
        Zv = (Xv - mu) @ comp.T
        total = ((Xv - mu) ** 2).sum()
        n = Xt.shape[1]
        for k in range(1, Xt.shape[1] + 1):
            resid = ((Xv - mu) - Zv[:, :k] @ comp[:k]) ** 2
            ev = 1.0 - resid.sum() / total
            if ev >= self.target_var:
                n, self.val_explained_ = k, float(ev)
                break
        else:
            self.val_explained_ = 1.0
        self.n_components_ = n
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scale(X)
        if not self.use_pca:
            return Xs
        return (Xs - self.pca.mean_) @ self.pca.components_[: self.n_components_].T

    def export(self) -> dict:
        """z = clip((x - mean) / std) @ P + c   (the hot path's PCA transform)."""
        d = len(self.scaler.mean_)
        if not self.use_pca:
            P, c = np.eye(d), np.zeros(d)
        else:
            P = self.pca.components_[: self.n_components_].T
            c = -self.pca.mean_ @ P
        return {"mean": self.scaler.mean_.copy(), "scale": self.scaler.scale_.copy(), "P": P.copy(), "c": c.copy(), "clip": CLIP}


def fit_gmm(Z: np.ndarray, n_states: int, seed: int = 7) -> GaussianMixture:
    """EM-fitted Gaussian mixture; its components seed the HMM emission distributions."""
    return GaussianMixture(n_components=n_states, covariance_type="full", n_init=3, reg_covar=1e-4,
                           random_state=seed).fit(Z)
