"""Reference baselines (NOT FLASH models; nothing here is trainable in the course's sense).

All three are estimated from TRAIN rows only and use the same row definition as FLASH's experiments
(split == s, valid_x, label >= 0), so their numbers are directly comparable with experiments A-D.

  majority      one-hot on the most frequent train class          (hard reference; terrible Brier by design)
  prior         constant train class frequencies                  (the Brier/log-loss floor to beat)
  momentum      P(class | tercile of z) from train counts, z = past h-bar log return / (vol30 * sqrt(h))
                -> uses only close[t], close[t-h] (same session) and vol30[t]; learns momentum OR reversal
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K = 3


def rows(t: pd.DataFrame, split: str, h: int) -> np.ndarray:
    m = (t["split"].to_numpy() == split) & t["valid_x"].to_numpy() & (t[f"y{h}"].to_numpy() >= 0)
    return np.flatnonzero(m)


def past_return_z(t: pd.DataFrame, h: int) -> np.ndarray:
    """(log close[t] - log close[t-h]) / (vol30[t] * sqrt(h)), within the same symbol-session; NaN otherwise."""
    lc = np.log(t["close"].to_numpy())
    key = t["symbol"].astype(str) + "_" + t["day_idx"].astype(str)
    prev = pd.Series(lc).groupby(key.to_numpy()).shift(h).to_numpy()     # t-h of the SAME session only
    return (lc - prev) / (t["vol30"].to_numpy() * np.sqrt(h))


class Baselines:
    def __init__(self, smoothing: float = 1.0):
        self.a = smoothing

    def fit(self, t: pd.DataFrame, h: int) -> "Baselines":
        tr = rows(t, "train", h)
        y = t[f"y{h}"].to_numpy()[tr].astype(int)
        self.h = h
        self.prior_ = np.bincount(y, minlength=K) / len(y)
        self.major_ = int(np.argmax(self.prior_))
        z = past_return_z(t, h)[tr]
        ok = np.isfinite(z)
        self.edges_ = np.quantile(z[ok], [1 / 3, 2 / 3])
        b = np.digitize(z[ok], self.edges_)
        cnt = np.zeros((3, K))
        np.add.at(cnt, (b, y[ok]), 1)
        self.table_ = (cnt + self.a) / (cnt + self.a).sum(1, keepdims=True)
        self.n_train_ = int(len(y))
        return self

    def predict(self, t: pd.DataFrame, idx: np.ndarray) -> dict[str, np.ndarray]:
        n = len(idx)
        maj = np.zeros((n, K))
        maj[:, self.major_] = 1.0
        pri = np.tile(self.prior_, (n, 1))
        z = past_return_z(t, self.h)[idx]
        mom = pri.copy()                                                  # no past return -> prior
        ok = np.isfinite(z)
        mom[ok] = self.table_[np.digitize(z[ok], self.edges_)]
        return {"majority": maj, "prior": pri, "momentum": mom}

    def describe(self) -> dict:
        return {"horizon": self.h, "n_train": self.n_train_, "train_prior[DOWN,NEUTRAL,UP]": self.prior_.round(4).tolist(),
                "majority_class": ["DOWN", "NEUTRAL", "UP"][self.major_], "momentum_z_tercile_edges": self.edges_.round(4).tolist(),
                "momentum_P(class|tercile)": self.table_.round(4).tolist()}
