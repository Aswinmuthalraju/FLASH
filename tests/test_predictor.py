"""End-to-end: the hot-loop Predictor must reproduce the batch pipeline's calibrated probabilities."""
import copy

import numpy as np
import pytest

from src.features.market_features import market_return
from src.features.news_features import raw_matrix
from src.evaluation.experiments import fit_variant, make_encoder, prepare
from src.inference.predictor import NO_SIGNAL, Predictor, make_bundle, replay


@pytest.fixture(scope="module")
def world(cfg, synth):
    c = copy.deepcopy(cfg)
    c["labels"]["horizons"] = [15]
    c["fusion"]["lgbm"].update(n_estimators=60, min_child_samples=30)
    bars, news, _ = synth
    ds = prepare(c, bars, news, make_encoder(c, finbert=False))
    finals = {"D": {15: fit_variant(ds, "D", 15)}, "A": {15: fit_variant(ds, "A", 15)}}
    bundle = make_bundle(ds, finals)
    bars_sorted = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return c, ds, finals, bundle, bars_sorted


def _replay_all_test_days(ds, pred, bars_sorted, with_news):
    t = ds.table
    mret = market_return(bars_sorted)
    ev = ds.base.ev
    events = None
    if with_news:
        order = np.argsort(ev["timestamp"].to_numpy(), kind="stable")
        e = ev.iloc[order]
        events = (e["timestamp"].tolist(), e["symbol"].tolist(), e["tmin"].to_numpy(), raw_matrix(ev)[order])
    got = {}
    test_rows = np.flatnonzero((t["split"] == "test").to_numpy())
    for d in np.unique(t["day_idx"].to_numpy()[test_rows]):
        rows = test_rows[t["day_idx"].to_numpy()[test_rows] == d]
        for r, p in replay(pred, bars_sorted, mret, t["tmin"].to_numpy(), rows, events):
            got[r] = p
    return got


def test_hot_loop_matches_batch_for_full_model(world):
    c, ds, finals, bundle, bars_sorted = world
    f = finals["D"][15]
    pred = Predictor(bundle, "D")
    got = _replay_all_test_days(ds, pred, bars_sorted, with_news=True)
    pos = {int(r): i for i, r in enumerate(f.rows_test)}
    common = [r for r in got if r in pos]
    assert len(common) > 0.9 * len(f.rows_test)
    live = np.array([got[r].probs[0] for r in common])
    batch = f.P_test_cal[[pos[r] for r in common]]
    err = np.abs(live - batch).max(axis=1)
    assert np.median(err) < 1e-9
    assert (err > 1e-6).mean() < 0.002, f"{(err > 1e-6).mean():.4f} of bars differ"
    assert np.allclose(live.sum(1), 1.0)
    # regime posterior is a probability vector and regime name comes from the bundle
    p = got[common[0]]
    assert abs(p.regime_post.sum() - 1) < 1e-9 and p.regime in bundle["hmm"]["names"]
    assert p.latency_us > 0 and p.stage_ns.shape == (6,)


def test_hot_loop_matches_batch_for_baseline(world):
    c, ds, finals, bundle, bars_sorted = world
    f = finals["A"][15]
    got = _replay_all_test_days(ds, Predictor(bundle, "A"), bars_sorted, with_news=False)
    pos = {int(r): i for i, r in enumerate(f.rows_test)}
    common = [r for r in got if r in pos]
    live = np.array([got[r].probs[0] for r in common])
    assert np.abs(live - f.P_test_cal[[pos[r] for r in common]]).max() < 1e-8
    assert got[common[0]].regime is None


def test_news_changes_the_prediction_and_decays(world):
    c, ds, finals, bundle, bars_sorted = world
    pred = Predictor(bundle, "D")
    mret = market_return(bars_sorted)
    t = ds.table
    rows = np.flatnonzero(((t["symbol"] == c["symbols"][0]) & (t["day_idx"] == 3)).to_numpy())[:120]
    from src.features.news_features import raw_vector
    base, shocked = [], []
    for with_news in (False, True):
        pred = Predictor(bundle, "D")
        pred.new_day()
        outs = []
        for k, r in enumerate(rows):
            if with_news and k == 60:
                pred.cache.set(c["symbols"][0], raw_vector(0.95, "Earnings", 1.0, (1.5, 0.2, 0.99)), t["tmin"].iat[r] - 0.5)
            p = pred.on_bar(0, bars_sorted["high"].iat[r], bars_sorted["low"].iat[r], bars_sorted["close"].iat[r],
                            bars_sorted["volume"].iat[r], mret[r], t["tmin"].iat[r])
            if p is not None:
                outs.append(p.probs[0].copy())
        (shocked if with_news else base).extend(outs)
    base, shocked = np.array(base), np.array(shocked)
    n_before = 60 - 30
    assert np.allclose(base[:n_before], shocked[:n_before])          # identical before the news
    assert np.abs(base[n_before:] - shocked[n_before:]).max() > 0.01  # news moved the prediction


def test_confidence_gate_and_explanations(world):
    c, ds, finals, bundle, bars_sorted = world
    pred = Predictor(bundle, "D")
    got = _replay_all_test_days(ds, pred, bars_sorted, with_news=True)
    pred.gate = 0.99
    r0 = next(iter(got))
    t = ds.table
    mret = market_return(bars_sorted)
    rows = np.flatnonzero((t["day_idx"] == t["day_idx"].iat[r0]).to_numpy() & (t["symbol"] == t["symbol"].iat[r0]).to_numpy())
    pred.new_day()
    last = None
    for r in rows[:60]:
        last = pred.on_bar(pred.sym_idx[t["symbol"].iat[r]], bars_sorted["high"].iat[r], bars_sorted["low"].iat[r],
                           bars_sorted["close"].iat[r], bars_sorted["volume"].iat[r], mret[r], t["tmin"].iat[r]) or last
    assert last.labels == [NO_SIGNAL]
    pred.gate = 0.0
    assert last is not None
    ex = pred.explain(15, top=5)
    assert set(ex["group_contribution"]) == {"market", "regime", "news", "history", "derived", "derived_x"}
    assert len(ex["top_features"]) == 5
    # contributions + bias reproduce the raw LightGBM margin for that class
    raw = pred.boosters[15].predict(pred.X1, raw_score=True, num_threads=1)[0]
    cls = ["DOWN", "NEUTRAL", "UP"].index(ex["class"])
    total = sum(ex["group_contribution"].values()) + ex["bias"]
    assert abs(total - raw[cls]) < 1e-6
