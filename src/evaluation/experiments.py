"""Dataset assembly + experiments A-D (+ shock subset). Ablations live in ablation.py.

Column layout of the fusion matrix XF:  [ market (14) | regime posteriors (K) | news (11) | history (3) ]
Experiments: A logistic/price only, B LGBM/market, C LGBM/market+HMM, D LGBM/everything.
Every model is trained on TRAIN, early-stopped and calibrated on VAL, scored on TEST.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import resolve
from src.data.align import Calendar
from src.evaluation.metrics import brier_multiclass, conf_matrix, evaluate
from src.features.market_features import FEATURES, IDX, PRICE_IDX, REGIME_FEATURES, build_table
from src.features.news_features import (NEWS_DIM, NEWS_FEATURES, build_events, memory_from_events, news_matrix)
from src.models.baseline_logistic import BaselineLogistic
from src.models.calibration import Calibrator
from src.models.fusion_lgbm import FusionLGBM
from src.models.hmm_regime import HMMRegime, group_lengths
from src.models.regime_pca_gmm import RegimePCA
from src.news.sentiment import FinBertEncoder, HashEncoder

N_MKT = len(FEATURES)
SHOCK_MINUTES = 30


class LazyFinBert:
    """Loads FinBERT on first use, so fully-cached runs never pay the model load."""

    def __init__(self, cfg):
        self.cfg, self._m = cfg, None

    def encode(self, headlines):
        if self._m is None:
            self._m = FinBertEncoder(self.cfg["news"]["finbert"], self.cfg["news"]["batch_size"])
        return self._m.encode(headlines)

    @property
    def device(self):
        return self._m.device if self._m else "not loaded (cache hit)"


def make_encoder(cfg: dict, finbert: bool = True):
    """finbert=False returns the hash stub -- for tests only."""
    return LazyFinBert(cfg) if finbert else HashEncoder()


@dataclass
class Base:
    cfg: dict
    table: pd.DataFrame
    cal: Calendar
    ev: pd.DataFrame
    E: np.ndarray
    news: np.ndarray            # (n, 14) decayed news vectors
    age: np.ndarray             # minutes since the latest usable news (nan = none)


@dataclass
class Regime:
    pca: RegimePCA
    hmm: HMMRegime
    post: np.ndarray            # (n, K), NaN on invalid rows
    names: list[str]


@dataclass
class Dataset:
    base: Base
    regime: Regime
    XF: np.ndarray
    col_names: list[str]
    groups: dict[str, list[int]] = field(default_factory=dict)

    @property
    def table(self):
        return self.base.table

    def rows(self, split: str, h: int) -> np.ndarray:
        t = self.table
        m = (t["split"].to_numpy() == split) & t["valid_x"].to_numpy() & (t[f"y{h}"].to_numpy() >= 0)
        return np.flatnonzero(m)

    def y(self, rows, h):
        return self.table[f"y{h}"].to_numpy()[rows].astype(int)

    def shock(self, rows) -> np.ndarray:
        a = self.base.age[rows]
        return np.isfinite(a) & (a <= SHOCK_MINUTES)


def prepare_base(cfg: dict, bars: pd.DataFrame, news: pd.DataFrame, encoder, cache_path=None) -> Base:
    table = build_table(bars, cfg)
    cal = Calendar.from_bars(bars, cfg)
    ev, E = build_events(news, table, cal, cfg, encoder, cache_path)
    N, age = news_matrix(table, ev, cal, cfg)
    return Base(cfg, table, cal, ev, E, N, age)


def fit_regime(base: Base, use_pca: bool = True) -> Regime:
    cfg, t = base.cfg, base.table
    rc = cfg["regime"]
    v = t["valid_x"].to_numpy()
    X = t.loc[v, REGIME_FEATURES].to_numpy()
    sp = t.loc[v, "split"].to_numpy()
    keys = (t.loc[v, "symbol"].astype(str) + "_" + t.loc[v, "day_idx"].astype(str)).to_numpy()
    tr = sp == "train"
    pca = RegimePCA(rc["pca_target_var"], use_pca).fit(X[tr], X[sp == "val"])
    Z = pca.transform(X)
    hmm = HMMRegime(rc["n_states"], rc["gmm_seed"]).fit(Z[tr], group_lengths(keys[tr]))
    post_v = hmm.filter_posteriors(Z, group_lengths(keys))
    names = hmm.name_states(post_v[tr], X[tr, REGIME_FEATURES.index("vol30")], X[tr, REGIME_FEATURES.index("mean20")])
    post = np.full((len(t), rc["n_states"]), np.nan)
    post[v] = post_v
    return Regime(pca, hmm, post, names)


def assemble(base: Base, regime: Regime) -> Dataset:
    K = regime.post.shape[1]
    XF = np.hstack([base.table[FEATURES].to_numpy(), regime.post, base.news])
    names = FEATURES + [f"regime_{n}" for n in regime.names] + NEWS_FEATURES
    g = {"market": list(range(N_MKT)),
         "regime": list(range(N_MKT, N_MKT + K)),
         "news": list(range(N_MKT + K, N_MKT + K + NEWS_DIM - 3)),
         "history": list(range(N_MKT + K + NEWS_DIM - 3, N_MKT + K + NEWS_DIM))}
    return Dataset(base, regime, XF, names, g)


def prepare(cfg, bars, news, encoder, cache_path=None, use_pca=True) -> Dataset:
    base = prepare_base(cfg, bars, news, encoder, cache_path)
    return assemble(base, fit_regime(base, use_pca))


# --------------------------------------------------------------------------------- variants

def variant_cols(ds: Dataset, name: str) -> list[int]:
    g = ds.groups
    return {"A": [PRICE_IDX[i] for i in range(len(PRICE_IDX))],
            "B": g["market"],
            "C": g["market"] + g["regime"],
            "D": g["market"] + g["regime"] + g["news"] + g["history"]}[name]


@dataclass
class Fitted:
    name: str
    h: int
    cols: list[int]
    kind: str                   # "logit" | "lgbm"
    model: object
    calibrator: Calibrator
    rows_test: np.ndarray
    P_test_raw: np.ndarray
    P_test_cal: np.ndarray
    P_val_raw: np.ndarray
    P_val_cal: np.ndarray


def fit_variant(ds: Dataset, name: str, h: int, cols: list[int] | None = None, kind: str | None = None) -> Fitted:
    cfg = ds.base.cfg
    cols = variant_cols(ds, name) if cols is None else cols
    kind = kind or ("logit" if name == "A" else "lgbm")
    tr, va, te = ds.rows("train", h), ds.rows("val", h), ds.rows("test", h)
    Xtr, Xva, Xte = ds.XF[tr][:, cols], ds.XF[va][:, cols], ds.XF[te][:, cols]
    ytr, yva = ds.y(tr, h), ds.y(va, h)
    if kind == "logit":
        m = BaselineLogistic().fit(Xtr, ytr)
        pva, pte = m.predict_proba(Xva), m.predict_proba(Xte)
    else:
        m = FusionLGBM(cfg["fusion"]["lgbm"], cfg["seed"]).fit(Xtr, ytr, Xva, yva)
        pva, pte = m.predict_proba(Xva), m.predict_proba(Xte)
    cal = Calibrator(cfg["fusion"]["calibration"]).fit(pva, yva)
    return Fitted(name, h, cols, kind, m, cal, te, pte, cal.transform(pte), pva, cal.transform(pva))


def score(ds: Dataset, f: Fitted) -> dict:
    y = ds.y(f.rows_test, f.h)
    sh = ds.shock(f.rows_test)
    out = {"all": evaluate(f.P_test_cal, y), "shock": evaluate(f.P_test_cal[sh], y[sh]),
           "calm": evaluate(f.P_test_cal[~sh], y[~sh])}
    out["all"]["brier_uncalibrated"] = brier_multiclass(f.P_test_raw, y)
    out["cm"] = conf_matrix(f.P_test_cal, y).tolist()
    return out


def run_experiments(ds: Dataset, horizons=None, names=("A", "B", "C", "D")):
    horizons = horizons or ds.base.cfg["labels"]["horizons"]
    fitted: dict[tuple[str, int], Fitted] = {}
    results: dict[int, dict[str, dict]] = {}
    for h in horizons:
        results[h] = {}
        for n in names:
            f = fit_variant(ds, n, h)
            fitted[(n, h)] = f
            results[h][n] = score(ds, f)
    return fitted, results


def results_table(results: dict, split: str = "all") -> pd.DataFrame:
    rows = []
    for h, per in results.items():
        for n, r in per.items():
            rows.append({"horizon": h, "model": n, **{k: v for k, v in r[split].items()}})
    return pd.DataFrame(rows)
