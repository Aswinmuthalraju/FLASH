"""Unit 4 (sequential): Gaussian HMM regimes.

LIVE: only the forward-FILTERED posterior P(state_t | z_1..t) is used, computed one bar at a time by
ForwardFilter.step() in plain NumPy with a scaling factor.

    Bayes rule at every bar:   posterior(j)  ∝  likelihood(z_t | j)  *  prior(j),
                               prior(j) = sum_i posterior_{t-1}(i) * A[i, j]      (predict step)
    The normaliser c_t = P(z_t | z_1..t-1) is the scaling factor (also gives the log-likelihood).

Viterbi (`viterbi`) uses the whole sequence -- future data -- so it is for OFFLINE plots and regime
analysis only. It is never a feature and is not imported anywhere in src/inference or src/features.
"""
from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM

from src.models.regime_pca_gmm import fit_gmm

_LOG2PI = np.log(2.0 * np.pi)


class ForwardFilter:
    """Incremental HMM forward algorithm with scaling. State: the filtered posterior `alpha`."""

    def __init__(self, startprob, transmat, means, covars):
        self.pi = np.asarray(startprob, float)
        self.A = np.asarray(transmat, float)
        self.mu = np.asarray(means, float)
        covars = np.asarray(covars, float)
        self.K, self.d = self.mu.shape
        self.prec = np.linalg.inv(covars)
        _, logdet = np.linalg.slogdet(covars)
        self.logc = -0.5 * (logdet + self.d * _LOG2PI)
        self.alpha = self.pi.copy()
        self.first = True
        self.loglik = 0.0
        self._diff = np.empty((self.K, self.d))

    def reset(self) -> None:
        self.first = True
        self.loglik = 0.0
        self.alpha = self.pi.copy()

    def step(self, z: np.ndarray) -> np.ndarray:
        np.subtract(z, self.mu, out=self._diff)                                  # (K, d)
        quad = np.einsum("kd,kd->k", self._diff, np.matmul(self.prec, self._diff[:, :, None])[:, :, 0])
        logb = self.logc - 0.5 * quad                                            # log N(z | mu_k, Sigma_k)
        mx = logb.max()
        b = np.exp(logb - mx)
        prior = self.pi if self.first else self.alpha @ self.A                   # predict
        a = prior * b                                                            # update (Bayes)
        c = a.sum()                                                              # scaling factor
        self.alpha = a / c
        self.first = False
        self.loglik += np.log(c) + mx
        return self.alpha


class HMMRegime:
    def __init__(self, n_states: int = 4, seed: int = 7, n_iter: int = 60):
        self.K = n_states
        self.seed = seed
        self.n_iter = n_iter
        self.model: GaussianHMM | None = None
        self.state_names: list[str] = [f"S{i}" for i in range(n_states)]

    def fit(self, Z: np.ndarray, lengths: list[int]) -> "HMMRegime":
        gmm = fit_gmm(Z, self.K, self.seed)                          # EM initialisation (Unit 3)
        m = GaussianHMM(n_components=self.K, covariance_type="full", init_params="", params="stmc",
                        n_iter=self.n_iter, tol=1e-3, min_covar=1e-4, random_state=self.seed)
        m.startprob_ = np.full(self.K, 1.0 / self.K)
        stay = 0.99                                                   # sticky prior; EM refines it
        m.transmat_ = np.full((self.K, self.K), (1 - stay) / (self.K - 1))
        np.fill_diagonal(m.transmat_, stay)
        m.means_ = gmm.means_.copy()
        m.covars_ = gmm.covariances_.copy()
        m.fit(Z, lengths)
        self.model = m
        return self

    # ---------------------------------------------------------------- live path pieces
    def new_filter(self) -> ForwardFilter:
        m = self.model
        return ForwardFilter(m.startprob_, m.transmat_, m.means_, m.covars_)

    def filter_posteriors(self, Z: np.ndarray, lengths: list[int]) -> np.ndarray:
        """Forward-filtered posteriors for stacked sequences (same code path as live)."""
        out = np.empty((len(Z), self.K))
        f = self.new_filter()
        r = 0
        for L in lengths:
            f.reset()
            for t in range(r, r + L):
                out[t] = f.step(Z[t])
            r += L
        return out

    # -------------------------------------------------------------------- offline only
    def viterbi(self, Z: np.ndarray, lengths: list[int]) -> np.ndarray:
        """OFFLINE ONLY (uses future observations): most likely state path for plots / regime analysis."""
        return self.model.decode(Z, lengths)[1]

    # ------------------------------------------------------------------------ naming
    def name_states(self, post: np.ndarray, vol: np.ndarray, drift: np.ndarray) -> list[str]:
        """Human labels from posterior-weighted mean volatility and drift on the training data."""
        w = post / post.sum(0, keepdims=True)
        v, d = (w * vol[:, None]).sum(0), (w * drift[:, None]).sum(0)
        order = np.argsort(v)
        names = [""] * self.K
        names[order[0]], names[order[-1]] = "calm", "volatile"
        mid = [i for i in order[1:-1]]
        mid.sort(key=lambda i: d[i])
        for rank, i in enumerate(mid):
            if len(mid) == 1:
                names[i] = "bull" if d[i] >= 0 else "bear"
            else:
                names[i] = "bear" if rank == 0 else "bull" if rank == len(mid) - 1 else "transition"
        self.state_names = names
        return names

    def export(self) -> dict:
        m = self.model
        return {"startprob": m.startprob_.copy(), "transmat": m.transmat_.copy(), "means": m.means_.copy(),
                "covars": m.covars_.copy(), "names": list(self.state_names)}


def group_lengths(keys: np.ndarray) -> list[int]:
    """Lengths of consecutive runs of equal keys (rows must be ordered so each sequence is contiguous)."""
    change = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    bounds = np.concatenate([[0], change, [len(keys)]])
    return np.diff(bounds).tolist()
