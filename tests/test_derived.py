import numpy as np

from src.features.derived import DERIVED, ND, derive_matrix, derive_row, derived_index
from src.features.market_features import FEATURES
from src.features.news_features import NEWS_FEATURES

NAMES = FEATURES + ["regime_bear", "regime_volatile", "regime_calm", "regime_bull"] + NEWS_FEATURES


def test_row_matches_matrix():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, len(NAMES)))
    idx = derived_index(NAMES)
    M = derive_matrix(X, idx)
    out = np.empty(ND)
    for i in range(len(X)):
        assert np.allclose(derive_row(X[i], idx, out), M[i], rtol=1e-12, atol=1e-12)
    assert M.shape == (50, len(DERIVED)) and np.isfinite(M).all()


def test_missing_regime_state_is_zero():
    names = [n for n in NAMES if n != "regime_bull"]
    X = np.random.default_rng(1).normal(size=(5, len(names)))
    idx = derived_index(names)
    M = derive_matrix(X, idx)
    assert (M[:, DERIVED.index("sent_x_bull")] == 0).all()
    assert derive_row(X[0], idx, np.empty(ND))[DERIVED.index("sent_x_bull")] == 0
