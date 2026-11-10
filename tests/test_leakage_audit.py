"""Leakage audit (preprocessing/leakage_audit.py): each check passes on clean data AND fails on a planted leak."""
import numpy as np
import pandas as pd
import pytest

from preprocessing import leakage_audit as LA
from preprocessing.prepare_market import drop_late_open_days
from src.config import load_config
from src.data.build import build_synthetic
from src.data.market_loader import regularize_bars, validate_bars
from src.features.market_features import WARMUP, build_table


@pytest.fixture(scope="module")
def data():
    cfg = load_config()
    cfg["synthetic"]["n_days"] = 8
    bars, news, _ = build_synthetic(cfg, seed=11, save=False)
    return cfg, bars, news, build_table(bars, cfg)


def _late_open(bars, first_minute=45):
    """Delete the first `first_minute` minutes of one symbol-day: first trade at 10:00."""
    day0 = bars.timestamp.dt.normalize().min()
    m = (bars.symbol == "TCS") & (bars.timestamp.dt.normalize() == day0)
    idx = bars[m].index[:first_minute]
    return bars.drop(idx), first_minute


def test_backfill_leak_exists_in_flash_loader_and_is_caught(data):
    cfg, bars, _, _ = data
    raw, f = _late_open(bars)
    reg = regularize_bars(validate_bars(raw), cfg)             # FLASH's own loader path
    t = build_table(reg, cfg)
    rep = LA.l5_backfill(reg, t, cfg)
    assert not rep["pass"] and rep["valid_rows_with_future_backfilled_price"] == f - WARMUP
    # the leak is real: the valid row at WARMUP already contains the close of minute f
    day = reg[(reg.symbol == "TCS")].sort_values("timestamp").iloc[: f + 1]
    assert day.close.iloc[WARMUP] == day.close.iloc[f]


def test_adapter_removes_backfill_leak(data):
    cfg, bars, _, _ = data
    raw, _ = _late_open(bars)
    kept, dropped = drop_late_open_days(raw, WARMUP, cfg["market"]["session_open"])
    assert len(dropped) == 1 and dropped[0].startswith("TCS")
    reg = regularize_bars(validate_bars(kept), cfg)
    assert LA.l5_backfill(reg, build_table(reg, cfg), cfg)["pass"]


def test_first_trade_exactly_at_warmup_is_not_a_leak(data):
    cfg, bars, _, _ = data
    raw, _ = _late_open(bars, first_minute=WARMUP)
    kept, dropped = drop_late_open_days(raw, WARMUP, cfg["market"]["session_open"])
    assert dropped == []
    reg = regularize_bars(validate_bars(kept), cfg)
    assert LA.l5_backfill(reg, build_table(reg, cfg), cfg)["pass"]


def test_clean_checks_pass(data):
    cfg, bars, _, t = data
    assert LA.l1_splits(t, cfg)["pass"]
    assert LA.l2_purge(t, cfg)["pass"]
    assert LA.l3_labels(t, cfg, n_sample=300)["pass"]
    assert LA.l4_causal_features(bars, cfg, n_sample=40)["pass"]


def test_purge_check_catches_window_crossing_session(data):
    cfg, _, _, t = data
    bad = t.copy()
    n = cfg["market"]["bars_per_day"]
    last = bad.index[(bad.tmin % n) == n - 2][:3]
    bad.loc[last, "y15"] = 1                                     # label whose window runs into the next day
    assert not LA.l2_purge(bad, cfg)["pass"]


def test_label_check_catches_corrupted_labels(data):
    cfg, _, _, t = data
    bad = t.copy()
    m = bad.y15 >= 0
    bad.loc[m, "y15"] = (bad.loc[m, "y15"] + 1) % 3
    assert LA.l3_labels(bad, cfg, n_sample=200)["mismatches"] > 0


def test_causality_check_catches_centered_window(data, monkeypatch):
    cfg, bars, _, _ = data
    real = LA.batch_day_features

    def leaky(o, h, l, c, v, mret, n):
        X = real(o, h, l, c, v, mret, n)
        X[:-5, 0] = np.log(c[5:] / c[:-5])                       # feature 0 now looks 5 bars ahead
        return X
    monkeypatch.setattr(LA, "batch_day_features", leaky)
    assert not LA.l4_causal_features(bars, cfg, n_sample=40)["pass"]


def test_split_check_catches_shuffled_split(data):
    cfg, _, _, t = data
    bad = t.copy()
    bad["split"] = np.where(bad.index % 2, "train", "test")
    bad.loc[bad.index[:10], "split"] = "val"
    assert not LA.l1_splits(bad, cfg)["pass"]
