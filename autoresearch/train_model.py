"""THE EDITABLE FILE of the FLASH autoresearch loop.

Current best recipe (exp 5): engineered features -> small-tree LightGBM (75%) + L2 logistic (25%) -> temperature scaling.
Contract: predict(split, ds) -> calibrated probabilities for split.Xdev, shape (n, 3).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare import run

_FE = {}


def engineer(ds):
    """Extra columns for every table row (causal: only same-timestamp cross-section and current-row values)."""
    if id(ds) in _FE:
        return _FE[id(ds)]
    t, X = ds.table, pd.DataFrame(ds.XF, columns=ds.col_names)
    g, ex = t.groupby("timestamp"), {}
    for c in ["ret1", "ret5", "ret15", "vol_z", "range"]:
        ex["xs_" + c] = (t[c] - g[c].transform("mean")).to_numpy()
        ex["rk_" + c] = g[c].rank(pct=True).to_numpy()
    ex["sym"] = t["symbol"].astype("category").cat.codes.to_numpy()
    for c in ["ret1", "ret5", "ret15"]:
        ex[c + "_vol"] = (t[c] / (t["vol30"] + 1e-9)).to_numpy()
    ex["mkt_rel"] = (t["ret5"] - t["beta20"] * t["mkt_ret1"]).to_numpy()
    ex["sent_x_bull"] = X["n_sent"] * X["regime_bull"]
    ex["sent_x_vol"] = X["n_sent"] * X["regime_volatile"]
    ex["sent_x_volz"] = X["n_sent"] * X["vol_z"]
    ex["sent_x_h"] = X["n_sent"] * X["h_mean"]
    ex["sent_over_vol"] = X["n_sent"] / (X["vol30"] + 1e-9)
    ex["hmean_over_vol"] = X["h_mean"] / (X["vol30"] + 1e-9)
    _FE[id(ds)] = np.hstack([ds.XF, pd.DataFrame(ex).to_numpy()])
    return _FE[id(ds)]


def temperature(Pf, yf, Pd):
    L = np.log(np.clip(Pf, 1e-9, 1))
    T = minimize_scalar(lambda T: log_loss(yf, softmax(L / T, 1), labels=[0, 1, 2]), bounds=(0.5, 3), method="bounded").x
    return softmax(np.log(np.clip(Pd, 1e-9, 1)) / T, 1)


def fit_predict(Xtr, ytr, Xf, yf, Xd, w_lr=0.25, seed=7):
    """Returns calibrated probabilities for Xd. Shared by the loop and by final test scoring."""
    m = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=3000, learning_rate=0.03, num_leaves=7,
                           min_child_samples=300, reg_lambda=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                           random_state=seed, verbose=-1, n_jobs=8)
    m.fit(Xtr, ytr, eval_set=[(Xf, yf)], callbacks=[lgb.early_stopping(50, verbose=False)])
    Pg_f, Pg_d = m.predict_proba(Xf), m.predict_proba(Xd)
    a, b, c = (np.nan_to_num(z) for z in (Xtr, Xf, Xd))
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=300)).fit(a, ytr)
    Pl_f, Pl_d = lr.predict_proba(b), lr.predict_proba(c)
    return temperature((1 - w_lr) * Pg_f + w_lr * Pl_f, yf, (1 - w_lr) * Pg_d + w_lr * Pl_d)


def predict(sp, ds):
    XA = engineer(ds)
    return fit_predict(XA[sp.meta_tr["rows"]], sp.ytr, XA[sp.meta_fit["rows"]], sp.yfit, XA[sp.meta_dev["rows"]])


if __name__ == "__main__":
    run(predict)
