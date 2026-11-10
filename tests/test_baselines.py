"""Reference baselines + extended metrics (evaluation/)."""
import numpy as np
import pytest

from evaluation.baselines import Baselines, past_return_z, rows
from evaluation.metrics_ext import block_bootstrap, full_report
from evaluation.run_baselines import check_split_allowed
from src.config import load_config
from src.data.build import build_synthetic
from src.features.market_features import build_table


@pytest.fixture(scope="module")
def table():
    cfg = load_config()
    cfg["synthetic"]["n_days"] = 10
    bars, _, _ = build_synthetic(cfg, seed=5, save=False)
    return cfg, build_table(bars, cfg)


def test_probabilities_valid_and_prior_matches_train(table):
    cfg, t = table
    bl = Baselines().fit(t, 15)
    idx = rows(t, "val", 15)
    for name, P in bl.predict(t, idx).items():
        assert P.shape == (len(idx), 3) and np.allclose(P.sum(1), 1) and np.isfinite(P).all(), name
    ytr = t["y15"].to_numpy()[rows(t, "train", 15)]
    assert np.allclose(bl.prior_, np.bincount(ytr, minlength=3) / len(ytr))


def test_fit_uses_train_only(table):
    cfg, t = table
    a = Baselines().fit(t, 15)
    bad = t.copy()
    m = bad.split != "train"
    bad.loc[m, "y15"] = 2                                      # corrupt every val/test label
    bad.loc[m, "close"] = bad.loc[m, "close"] * 1.5
    b = Baselines().fit(bad, 15)
    assert np.allclose(a.prior_, b.prior_) and np.allclose(a.table_, b.table_) and np.allclose(a.edges_, b.edges_)


def test_momentum_feature_is_causal_and_session_local(table):
    cfg, t = table
    z = past_return_z(t, 15)
    bad = t.copy()
    r = 5000                                                   # a mid-session row
    bad.loc[bad.index > r, "close"] *= 3.0                     # change only the FUTURE
    assert z[r] == past_return_z(bad, 15)[r]
    first_bars = t.groupby(["symbol", "day_idx"]).head(15).index
    assert np.isnan(z[first_bars]).all()                      # never reaches into the previous session


def test_full_report_and_bootstrap(table):
    cfg, t = table
    bl = Baselines().fit(t, 15)
    idx = rows(t, "val", 15)
    y = t["y15"].to_numpy()[idx].astype(int)
    P = bl.predict(t, idx)
    rep = full_report(P["momentum"], y)
    assert set(rep["per_class"]) == {"DOWN", "NEUTRAL", "UP"}
    assert np.array(rep["confusion_matrix[true x pred]"]).sum() == len(y)
    bs = block_bootstrap(P["momentum"], y, t["day_idx"].to_numpy()[idx], B=50, P_ref=P["prior"])
    lo, hi = bs["ci"]["brier"]
    assert lo <= rep["brier"] <= hi or np.isclose(lo, hi)
    assert bs["n_days"] == len(np.unique(t["day_idx"].to_numpy()[idx]))
    bs_self = block_bootstrap(P["prior"], y, t["day_idx"].to_numpy()[idx], B=20, P_ref=P["prior"])
    assert bs_self["diff_vs_ref_ci"]["brier"] == [0.0, 0.0]


def test_test_split_is_locked(monkeypatch):
    monkeypatch.delenv("FLASH_TEST_UNLOCK", raising=False)
    with pytest.raises(SystemExit):
        check_split_allowed("test", unlock=True)
    with pytest.raises(SystemExit):
        check_split_allowed("test", unlock=False)
    check_split_allowed("val", unlock=False)
    monkeypatch.setenv("FLASH_TEST_UNLOCK", "I_HAVE_APPROVAL")
    check_split_allowed("test", unlock=True)
