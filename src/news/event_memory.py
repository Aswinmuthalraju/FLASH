"""Historical event impact: cosine nearest neighbours among PAST events only.

Each memory entry = (FinBERT [CLS] embedding, normalised forward return that followed the event,
`known_at` = trading-minute at which that forward return became observable). A query at time `now`
may only see entries with known_at < now, i.e. whose outcome had already happened. That is what keeps the
feature free of look-ahead (tests/test_event_memory.py).
"""
from __future__ import annotations

import numpy as np


def _unit(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float32)
    return E / (np.linalg.norm(E, axis=-1, keepdims=True) + 1e-9)


def _summarise(sims: np.ndarray, fwd: np.ndarray, k: int) -> tuple[float, float, float]:
    if len(sims) == 0:
        return 0.0, 0.0, 0.0
    top = np.argsort(-sims)[:k]
    r = fwd[top]
    return float(r.mean()), float(r.std()), float(sims[top].mean())


class EventMemory:
    def __init__(self, k: int = 5, dim: int = 768):
        self.k = k
        self.emb = np.zeros((0, dim), np.float32)
        self.fwd = np.zeros(0)
        self.known_at = np.zeros(0)

    def __len__(self):
        return len(self.fwd)

    def add(self, emb: np.ndarray, fwd_ret: float, known_at: float) -> None:
        self.emb = np.vstack([self.emb, _unit(emb)[None]])
        self.fwd = np.append(self.fwd, fwd_ret)
        self.known_at = np.append(self.known_at, known_at)

    def add_many(self, emb: np.ndarray, fwd: np.ndarray, known_at: np.ndarray) -> None:
        self.emb = np.vstack([self.emb, _unit(emb)])
        self.fwd = np.concatenate([self.fwd, fwd])
        self.known_at = np.concatenate([self.known_at, known_at])

    def query(self, emb: np.ndarray, now: float) -> tuple[float, float, float]:
        """(mean fwd return, std fwd return, mean cosine similarity) over the top-k PAST neighbours."""
        ok = self.known_at < now
        if not ok.any():
            return 0.0, 0.0, 0.0
        sims = self.emb[ok] @ _unit(emb)
        return _summarise(sims, self.fwd[ok], self.k)


def history_features(emb: np.ndarray, event_tmin: np.ndarray, fwd: np.ndarray, known_at: np.ndarray, k: int) -> np.ndarray:
    """Vectorised batch version for training: (n, 3) = (mean, std, sim) per event. Entries with
    NaN fwd/known_at (outcome undefined) are never used as neighbours but are still queried."""
    n = len(emb)
    U = _unit(emb)
    S = U @ U.T                                                    # (n, n) cosine
    usable = np.isfinite(known_at)[None, :] & (known_at[None, :] < event_tmin[:, None])
    out = np.zeros((n, 3))
    for i in range(n):
        m = usable[i]
        if m.any():
            out[i] = _summarise(S[i, m], fwd[m], k)
    return out
