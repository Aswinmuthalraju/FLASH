"""Fast path (hot loop): CPU only, NumPy only, no pandas / DB / network / model loading.

    new bar -> incremental features (ring buffers) -> PCA transform -> HMM forward step
            -> cached news vector -> LightGBM single-row predict -> calibration -> confidence gate

Mode "D" is the full model; mode "A" is the logistic baseline (price features only) used for the latency
comparison. Everything expensive (bundle loading, FinBERT, event memory queries) happens before the loop or
in the slow path (features.news_features.NewsProcessor).
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns as _pc

import joblib
import numpy as np

from src.config import resolve
from src.features.market_features import CLASS_NAMES, FEATURES, PRICE_IDX, REGIME_IDX
from src.features.news_features import NEWS_DIM
from src.inference.feature_cache import FeatureState, NewsCache
from src.models.calibration import FastCalibrator
from src.models.fusion_lgbm import load_booster
from src.models.hmm_regime import ForwardFilter

STAGES = ["features", "pca", "hmm", "news", "model", "calib+gate"]
NO_SIGNAL = "NO STRONG SIGNAL"
N_MKT = len(FEATURES)


class Prediction:
    __slots__ = ("symbol", "horizons", "probs", "labels", "regime", "regime_post", "latency_us", "stage_ns")

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "regime": self.regime,
                "regime_post": None if self.regime_post is None else self.regime_post.tolist(),
                "horizons": {h: {"P_UP": float(p[2]), "P_NEUTRAL": float(p[1]), "P_DOWN": float(p[0]),
                                 "signal": lab} for h, p, lab in zip(self.horizons, self.probs, self.labels)},
                "latency_us": self.latency_us}


class Predictor:
    def __init__(self, bundle: dict, mode: str = "D", horizons: list[int] | None = None):
        assert mode in ("A", "D")
        cfg = bundle["cfg"]
        self.cfg, self.mode = cfg, mode
        self.symbols: list[str] = cfg["symbols"]
        self.horizons = horizons or bundle["horizons"]
        self.gate = cfg["inference"]["gate_threshold"]
        self.states = [FeatureState(cfg["market"]["bars_per_day"]) for _ in self.symbols]
        self.sym_idx = {s: i for i, s in enumerate(self.symbols)}
        self.probs = np.zeros((len(self.horizons), 3))
        self._raw = np.zeros((len(self.horizons), 3))
        self.stage_ns = np.zeros(len(STAGES), dtype=np.int64)
        self.col_names = bundle["col_names"]
        self.groups = bundle["groups"]
        if mode == "D":
            pc = bundle["pca"]
            self.mean, self.scale = pc["mean"].copy(), pc["scale"].copy()
            self.P, self.c = np.ascontiguousarray(pc["P"]), pc["c"].copy()
            self.clip = float(pc["clip"])
            h = bundle["hmm"]
            self.regime_names = h["names"]
            self.K = len(self.regime_names)
            self.filters = [ForwardFilter(h["startprob"], h["transmat"], h["means"], h["covars"]) for _ in self.symbols]
            self.cache = NewsCache(self.symbols, cfg["news"]["decay_tau_min"], cfg["news"]["max_age_min"])
            self.ncols = N_MKT + self.K + NEWS_DIM
            self.X1 = np.zeros((1, self.ncols))
            self.xf = self.X1[0]
            self.news_view = self.xf[N_MKT + self.K:]
            self.ridx = np.array(REGIME_IDX)
            self._tmp = np.empty(len(REGIME_IDX))
            self._z = np.empty(self.P.shape[1])
            self.boosters = {}
            for hh in self.horizons:
                assert bundle["D"][hh]["cols"] == list(range(self.ncols)), "D model must use the full column layout"
                self.boosters[hh] = load_booster(bundle["D"][hh]["model"])
            self.cals = {hh: FastCalibrator(bundle["D"][hh]["cal"]) for hh in self.horizons}
        else:
            self.regime_names = []
            self.W = {hh: np.ascontiguousarray(bundle["A"][hh]["W"]) for hh in self.horizons}
            self.b = {hh: bundle["A"][hh]["b"] for hh in self.horizons}
            self.cals = {hh: FastCalibrator(bundle["A"][hh]["cal"]) for hh in self.horizons}
            self._xp = np.empty(len(PRICE_IDX))
            self._pidx = np.array(PRICE_IDX)

    # ------------------------------------------------------------------ session handling
    def new_day(self) -> None:
        for s in self.states:
            s.reset()
        if self.mode == "D":
            for f in self.filters:
                f.reset()

    # --------------------------------------------------------------------------- hot loop
    def on_bar(self, si: int, high: float, low: float, close: float, volume: float, mret: float,
               tmin: float) -> Prediction | None:
        """One new 1-minute bar of symbol index `si`. Returns None during the daily warm-up."""
        t0 = _pc()
        st = self.states[si]
        if not st.update(high, low, close, volume, mret):
            return None
        t1 = _pc()
        probs, raw, hs = self.probs, self._raw, self.horizons
        if self.mode == "D":
            tmp = self._tmp
            np.take(st.out, self.ridx, out=tmp)
            np.subtract(tmp, self.mean, out=tmp)
            np.divide(tmp, self.scale, out=tmp)
            np.clip(tmp, -self.clip, self.clip, out=tmp)
            np.dot(tmp, self.P, out=self._z)
            self._z += self.c
            t2 = _pc()
            alpha = self.filters[si].step(self._z)                           # forward-filtered posterior
            t3 = _pc()
            xf = self.xf
            xf[:N_MKT] = st.out
            xf[N_MKT:N_MKT + self.K] = alpha
            self.cache.read(self.symbols[si], tmin, self.news_view)
            t4 = _pc()
            for j, hh in enumerate(hs):
                raw[j] = self.boosters[hh].predict(self.X1, num_threads=1)[0]
            t5 = _pc()
            regime_post = alpha
        else:
            t2 = t3 = t4 = _pc()
            x = st.out[self._pidx]
            for j, hh in enumerate(hs):
                z = self.W[hh] @ x + self.b[hh]
                z = np.exp(z - z.max())
                raw[j] = z / z.sum()
            t5 = _pc()
            regime_post = None
        labels = []
        for j, hh in enumerate(hs):
            p = self.cals[hh](raw[j], probs[j])
            k = int(p.argmax())
            labels.append(NO_SIGNAL if p[k] < self.gate else CLASS_NAMES[k])
        t6 = _pc()
        ns = self.stage_ns
        ns[0], ns[1], ns[2], ns[3], ns[4], ns[5] = t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5
        out = Prediction()
        out.symbol, out.horizons, out.probs, out.labels = self.symbols[si], hs, probs.copy(), labels
        out.regime_post = None if regime_post is None else regime_post.copy()
        out.regime = None if regime_post is None else self.regime_names[int(regime_post.argmax())]
        out.stage_ns = ns.copy()
        out.latency_us = (t6 - t0) / 1000.0
        return out

    # ----------------------------------------------------------------------- explainability
    def explain(self, horizon: int, cls: int | None = None, top: int = 8, x: np.ndarray | None = None) -> dict:
        """pred_contrib for the LAST predicted row (or a stored fusion vector `x`), grouped into
        market / regime / news / history. Not part of the timed hot loop."""
        row = self.X1 if x is None else np.ascontiguousarray(x, dtype=float).reshape(1, -1)
        booster = self.boosters[horizon]
        c = booster.predict(row, pred_contrib=True, num_threads=1).reshape(3, -1)
        if cls is None:
            cls = int(booster.predict(row, num_threads=1)[0].argmax())
        contrib, bias = c[cls, :-1], float(c[cls, -1])
        by_group = {g: float(contrib[idx].sum()) for g, idx in self.groups.items()}
        group_abs = {g: float(np.abs(contrib[idx]).sum()) for g, idx in self.groups.items()}
        order = np.argsort(-np.abs(contrib))[:top]
        gname = {i: g for g, idx in self.groups.items() for i in idx}
        feats = [{"feature": self.col_names[i], "group": gname[i], "value": float(row[0, i]),
                  "contribution": float(contrib[i])} for i in order]
        return {"class": CLASS_NAMES[cls], "bias": bias, "group_contribution": by_group, "group_abs": group_abs,
                "top_features": feats}


# ---------------------------------------------------------------------------- bundle I/O (offline)

def make_bundle(ds, finals: dict) -> dict:
    """finals: {"D": {h: Fitted}, "A": {h: Fitted}} from experiments.fit_variant."""
    cfg = ds.base.cfg
    hs = sorted(finals["D"])
    b = {"cfg": cfg, "horizons": hs, "col_names": ds.col_names, "groups": ds.groups,
         "pca": ds.regime.pca.export(), "hmm": ds.regime.hmm.export(), "D": {}, "A": {}}
    for h in hs:
        fd, fa = finals["D"][h], finals["A"][h]
        b["D"][h] = {"model": fd.model.model_string(), "cal": fd.calibrator.export(), "cols": fd.cols}
        b["A"][h] = {**fa.model.export(), "cal": fa.calibrator.export()}
    return b


def save_bundle(bundle: dict, memory, cfg: dict) -> Path:
    d = resolve(cfg, "models")
    joblib.dump(bundle, d / "bundle.joblib")
    np.savez(d / "event_memory.npz", emb=memory.emb, fwd=memory.fwd, known_at=memory.known_at)
    return d / "bundle.joblib"


def load_bundle(cfg: dict) -> dict:
    # SECURITY NOTE: joblib/pickle executes code on load. bundle.joblib is an artifact written by scripts/train.py
    # on this machine; never load a bundle obtained from an untrusted source.
    return joblib.load(resolve(cfg, "models") / "bundle.joblib")


def load_memory(cfg: dict):
    from src.news.event_memory import EventMemory

    z = np.load(resolve(cfg, "models") / "event_memory.npz")
    m = EventMemory(cfg["news"]["memory_k"], z["emb"].shape[1])
    m.emb, m.fwd, m.known_at = z["emb"], z["fwd"], z["known_at"]
    return m


# --------------------------------------------------------------------------------- replay driver

def replay(pred: Predictor, bars, mret: np.ndarray, tmin: np.ndarray, rows: np.ndarray, events=None):
    """Feed the rows (one session, sorted by timestamp) through the hot loop.

    bars: DataFrame with symbol, timestamp, high, low, close, volume (indexed positionally by `rows`).
    events: optional (timestamps list, symbols list, tmins, raw (n, 13)) pushed into the news cache STRICTLY before
    each bar -- exactly what the slow path would have produced by then.
    Returns [(row, Prediction)].
    """
    sym = bars["symbol"].to_numpy()
    ts = bars["timestamp"].tolist()
    hi, lo, cl, vo = (bars[k].to_numpy() for k in ("high", "low", "close", "volume"))
    order = rows[np.lexsort((np.array([pred.sym_idx[s] for s in sym[rows]]), np.array([ts[r].value for r in rows])))]
    pred.new_day()
    out = []
    if events is not None:
        e_ts, e_sym, e_tmin, e_raw = events
        j = 0
    for r in order:
        if events is not None:
            while j < len(e_ts) and e_ts[j] < ts[r]:
                pred.cache.set(e_sym[j], e_raw[j], e_tmin[j])
                j += 1
        p = pred.on_bar(pred.sym_idx[sym[r]], hi[r], lo[r], cl[r], vo[r], mret[r], tmin[r])
        if p is not None:
            out.append((int(r), p))
    return out
