import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.evaluation.backtest import run_backtest
from src.evaluation.metrics import brier_multiclass, evaluate
from src.models.calibration import Calibrator, FastCalibrator


def _miscalibrated(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, n, p=[0.3, 0.4, 0.3])
    logits = np.eye(3)[y] * 1.0 + rng.standard_normal((n, 3))
    P = np.exp(3.0 * logits)                                  # over-confident on purpose
    return P / P.sum(1, keepdims=True), y


def test_calibration_improves_brier_and_sums_to_one():
    P, y = _miscalibrated()
    va, te = slice(0, 3000), slice(3000, None)
    for method in ("isotonic", "platt"):
        cal = Calibrator(method).fit(P[va], y[va])
        Q = cal.transform(P[te])
        assert np.allclose(Q.sum(1), 1.0)
        assert brier_multiclass(Q, y[te]) < brier_multiclass(P[te], y[te])


def test_fast_calibrator_equals_sklearn_isotonic_and_batch():
    P, y = _miscalibrated()
    cal = Calibrator("isotonic").fit(P[:3000], y[:3000])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(P[:3000, 0], (y[:3000] == 0).astype(float))
    assert np.allclose(np.interp(P[3000:3100, 0], *cal.maps[0]), iso.predict(P[3000:3100, 0]))
    for method in ("isotonic", "platt"):
        cal = Calibrator(method).fit(P[:3000], y[:3000])
        fast, out = FastCalibrator(cal.export()), np.zeros(3)
        Q = cal.transform(P[3000:3050])
        for i in range(50):
            assert np.allclose(fast(P[3000 + i], out), Q[i])


def test_metrics_perfect_and_uniform():
    y = np.array([0, 1, 2, 1, 0, 2])
    perfect = np.eye(3)[y]
    m = evaluate(perfect, y)
    assert m["macro_f1"] == 1.0 and m["auc_ovr"] == 1.0 and m["brier"] == 0.0
    assert abs(brier_multiclass(np.full((6, 3), 1 / 3), y) - 2 / 3) < 1e-12


def _toy_table(n_days=2, n=50, syms=("A", "B")):
    rows = []
    for s in syms:
        for d in range(n_days):
            c = 100 * np.exp(np.cumsum(np.full(n, 0.001)))      # steadily rising
            rows += [dict(symbol=s, day_idx=d, close=x) for x in c]
    return pd.DataFrame(rows)


def test_backtest_costs_non_overlap_and_session_boundary():
    t = _toy_table()
    rows = np.arange(len(t))
    P = np.zeros((len(t), 3)); P[:, 2] = 0.9; P[:, 1] = 0.1          # always confident UP
    h, cost = 5, 3.0
    res, trades = run_backtest(t, rows, P, h, 0.55, cost, entry_delay=1)
    assert (trades["dir"] == 1).all()
    assert np.allclose(trades["gross"], 0.001 * h)                    # rising series, exact
    assert np.allclose(trades["net"], 0.001 * h - 2 * cost / 1e4)
    # one position at a time per symbol: consecutive trades on a symbol-day are >= entry_delay + h + 1 rows apart
    for _, g in trades.groupby(["symbol", "day"]):
        assert (np.diff(g["row"].to_numpy()) >= h + 1).all()
    # exits never cross the day boundary
    assert (t["day_idx"].to_numpy()[trades["row"] + 1 + h] == trades["day"].to_numpy()).all()
    assert res["hit_rate"] == 1.0
    # threshold above the confidence => no trades
    assert run_backtest(t, rows, P, h, 0.95, cost)[0]["trades"] == 0
    # neutral top class => no trades
    Pn = np.zeros((len(t), 3)); Pn[:, 1] = 0.9; Pn[:, 2] = 0.1
    assert run_backtest(t, rows, Pn, h, 0.55, cost)[0]["trades"] == 0
