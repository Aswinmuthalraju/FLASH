import pandas as pd
import pytest

from src.data.align import assign_split, chronological_split, split_days


def test_split_is_chronological_disjoint_and_complete(cfg, synth):
    bars, _, _ = synth
    parts = chronological_split(bars, cfg)
    tr, va, te = (parts[k]["timestamp"] for k in ("train", "val", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
    assert len(tr) + len(va) + len(te) == len(bars)
    for p in parts.values():                               # each part keeps time order, nothing shuffled
        assert p["timestamp"].is_monotonic_increasing or p.groupby("symbol")["timestamp"].apply(lambda s: s.is_monotonic_increasing).all()


def test_split_never_cuts_inside_a_day(cfg, synth):
    bars, _, _ = synth
    parts = chronological_split(bars, cfg)
    days = [set(p["timestamp"].dt.normalize()) for p in parts.values()]
    assert not (days[0] & days[1]) and not (days[1] & days[2]) and not (days[0] & days[2])


def test_split_fractions(cfg):
    days = pd.bdate_range("2024-01-01", periods=50, tz="Asia/Kolkata")
    sp = split_days(days, cfg)
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (30, 10, 10)


def test_too_few_days_rejected(cfg):
    with pytest.raises(ValueError):
        split_days(pd.bdate_range("2024-01-01", periods=2, tz="Asia/Kolkata"), cfg)


def test_timestamp_outside_split_rejected(cfg):
    days = pd.bdate_range("2024-01-01", periods=10, tz="Asia/Kolkata")
    sp = split_days(days, cfg)
    with pytest.raises(ValueError):
        assign_split(pd.Series([pd.Timestamp("2025-01-01", tz="Asia/Kolkata")]), sp)
