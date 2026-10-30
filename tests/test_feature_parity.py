"""The incremental fast-path features must equal the batch features."""
import numpy as np

from src.features.market_features import FEATURES, build_table, batch_features, market_return, triple_barrier
from src.inference.feature_cache import FeatureState


def test_incremental_equals_batch(cfg, synth):
    bars, _, _ = synth
    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    X = batch_features(bars, cfg)
    mret = market_return(bars)
    day = bars["timestamp"].dt.normalize().to_numpy()
    o = {k: bars[k].to_numpy() for k in ("high", "low", "close", "volume")}
    sym = bars["symbol"].to_numpy()
    state = FeatureState(cfg["market"]["bars_per_day"])
    worst = 0.0
    n_valid = 0
    for r in range(len(bars)):
        if r == 0 or sym[r] != sym[r - 1] or day[r] != day[r - 1]:
            state.reset()
        ok = state.update(o["high"][r], o["low"][r], o["close"][r], o["volume"][r], mret[r])
        assert ok == (not np.isnan(X[r]).any()), f"validity mismatch at row {r}"
        if ok:
            n_valid += 1
            worst = max(worst, float(np.max(np.abs(state.out - X[r]) / (np.abs(X[r]) + 1e-9))))
    assert n_valid > 1000
    assert worst < 1e-8, f"max relative difference {worst}"


def test_no_nan_features_after_warmup_and_nan_before(cfg, synth):
    bars, _, _ = synth
    t = build_table(bars, cfg)
    first_day = t[(t.symbol == cfg["symbols"][0]) & (t.day_idx == 0)]
    assert first_day[FEATURES].iloc[:30].isna().all().all()
    assert not first_day[FEATURES].iloc[30:].isna().any().any()


def test_triple_barrier_hand_cases():
    sigma = np.full(10, 0.01)
    up = np.exp(np.cumsum([0, 0.004, 0.004, 0.004, 0, 0, 0, 0, 0, 0]))
    assert triple_barrier(up, sigma, 3, 1.0, False)[0] == 2            # +1.2% > 1% => UP
    dn = np.exp(np.cumsum([0, -0.02, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert triple_barrier(dn, sigma, 3, 1.0, False)[0] == 0
    flat = np.exp(np.cumsum([0, 0.001, -0.001, 0.001] + [0] * 6))
    assert triple_barrier(flat, sigma, 3, 1.0, False)[0] == 1
    both = np.exp(np.cumsum([0, 0.02, -0.04, 0] + [0] * 6))          # up hit first, then down
    assert triple_barrier(both, sigma, 3, 1.0, False)[0] == 2
    both2 = np.exp(np.cumsum([0, -0.02, 0.04, 0] + [0] * 6))
    assert triple_barrier(both2, sigma, 3, 1.0, False)[0] == 0
    y = triple_barrier(up, sigma, 3, 1.0, False)
    assert (y[-3:] == -1).all()                                         # horizon must fit inside the day


def test_labels_use_only_the_future_features_use_only_the_past(cfg, synth):
    """Features at row t must not change when future prices change."""
    bars, _, _ = synth
    b = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    X1 = batch_features(b, cfg)
    b2 = b.copy()
    cut = 200
    sel = (b2.symbol == b2.symbol.iat[0]) & (b2.timestamp.dt.normalize() == b2.timestamp.dt.normalize().iat[0])
    rows = np.flatnonzero(sel.to_numpy())
    b2.loc[rows[cut:], ["open", "high", "low", "close"]] *= 1.5
    b2.loc[rows[cut:], "volume"] *= 3
    X2 = batch_features(b2, cfg)
    assert np.array_equal(X1[rows[:cut]], X2[rows[:cut]], equal_nan=True)
