"""Derived fusion features: volatility-normalised returns and news x regime / volatility interactions.

Pure per-row functions of columns that already exist in the fusion vector, so they are available in the hot path
without any cross-sectional state. `derive_matrix` (offline, vectorised) and `derive_row` (hot path, scalar math,
zero allocation) must agree -- tests/test_derived.py.

Group "derived":   market-only        (ret*_vol, mkt_rel)
Group "derived_x": uses news/regime/history columns (dropped by the matching ablations)
"""
from __future__ import annotations

import numpy as np

MARKET_DERIVED = ["ret1_vol", "ret5_vol", "ret15_vol", "mkt_rel"]
CROSS_DERIVED = ["sent_x_bull", "sent_x_vol", "sent_x_volz", "sent_x_h", "sent_over_vol", "hmean_over_vol"]
DERIVED = MARKET_DERIVED + CROSS_DERIVED
ND = len(DERIVED)
_NEEDS = ["ret1", "ret5", "ret15", "vol30", "beta20", "mkt_ret1", "vol_z", "n_sent", "h_mean",
          "regime_bull", "regime_volatile"]
_EPS = 1e-9


def derived_index(col_names: list[str]) -> np.ndarray:
    """Positions of the source columns inside the base fusion vector (missing regime state -> -1)."""
    return np.array([col_names.index(n) if n in col_names else -1 for n in _NEEDS])


def derive_matrix(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    def g(k):
        return X[:, idx[k]] if idx[k] >= 0 else np.zeros(len(X))

    v = g(3) + _EPS
    out = np.empty((len(X), ND))
    out[:, 0], out[:, 1], out[:, 2] = g(0) / v, g(1) / v, g(2) / v
    out[:, 3] = g(1) - g(4) * g(5)
    s = g(7)
    out[:, 4], out[:, 5], out[:, 6] = s * g(9), s * g(10), s * g(6)
    out[:, 7], out[:, 8], out[:, 9] = s * g(8), s / v, g(8) / v
    return out


def derive_row(x: np.ndarray, idx: np.ndarray, out: np.ndarray) -> np.ndarray:
    i = idx
    v = x[i[3]] + _EPS
    out[0], out[1], out[2] = x[i[0]] / v, x[i[1]] / v, x[i[2]] / v
    out[3] = x[i[1]] - x[i[4]] * x[i[5]]
    s, hm = x[i[7]], x[i[8]]
    out[4] = s * x[i[9]] if i[9] >= 0 else 0.0
    out[5] = s * x[i[10]] if i[10] >= 0 else 0.0
    out[6], out[7], out[8], out[9] = s * x[i[6]], s * hm, s / v, hm / v
    return out
