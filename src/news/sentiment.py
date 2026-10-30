"""Slow path: FinBERT sentiment in [-1, 1] and the [CLS] embedding (one forward pass gives both).

FinBertEncoder is the real thing (GPU if available). HashEncoder is a tiny deterministic stand-in used ONLY
by unit tests so they do not need a 440 MB download; it is never used for training, evaluation or the demo.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

EMB_DIM = 768


class FinBertEncoder:
    def __init__(self, name: str = "ProsusAI/finbert", batch_size: int = 32, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device).eval()
        self.batch_size = batch_size
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.i_pos, self.i_neg = labels["positive"], labels["negative"]

    def encode(self, headlines: list[str]) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        sent, emb = [], []
        with torch.no_grad():
            for i in range(0, len(headlines), self.batch_size):
                batch = list(headlines[i:i + self.batch_size])
                t = self.tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
                out = self.model(**t, output_hidden_states=True)
                p = torch.softmax(out.logits, dim=-1)
                sent.append((p[:, self.i_pos] - p[:, self.i_neg]).cpu().numpy())
                emb.append(out.hidden_states[-1][:, 0].cpu().numpy())      # [CLS]
        return np.concatenate(sent).astype(np.float32), np.concatenate(emb).astype(np.float32)


_POS = {"jumps", "beats", "record", "up", "approves", "eases", "rally", "rallies", "cheer", "surges", "clears", "cuts", "upside"}
_NEG = {"falls", "misses", "weak", "declines", "disappointing", "plunge", "slide", "tumbles", "hikes", "resigns",
        "fraud", "selloff", "probe", "concerns", "abandons"}


class HashEncoder:
    """TEST STUB. Bag-of-words hashed into EMB_DIM dims + a crude lexicon sentiment. Deterministic."""

    def __init__(self, *_, **__):
        self.device = "stub"

    def encode(self, headlines: list[str]) -> tuple[np.ndarray, np.ndarray]:
        sent = np.zeros(len(headlines), np.float32)
        emb = np.zeros((len(headlines), EMB_DIM), np.float32)
        for i, h in enumerate(headlines):
            toks = [t.strip(",.;$%").lower() for t in h.split()]
            for t in toks:
                emb[i, int(hashlib.md5(t.encode()).hexdigest(), 16) % EMB_DIM] += 1.0
            sent[i] = np.clip((sum(t in _POS for t in toks) - sum(t in _NEG for t in toks)) / 2.0, -1, 1)
        return sent, emb


def encode_cached(encoder, headlines: list[str], cache: Path | None):
    """Encode unique headlines once; persist to an .npz keyed by sha1 so reruns are instant."""
    keys = [hashlib.sha1(h.encode()).hexdigest() for h in headlines]
    store: dict[str, tuple[float, np.ndarray]] = {}
    if cache is not None and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        store = {k: (s, e) for k, s, e in zip(z["keys"], z["sent"], z["emb"])}
    todo = sorted({(k, h) for k, h in zip(keys, headlines) if k not in store})
    if todo:
        s, e = encoder.encode([h for _, h in todo])
        for (k, _), si, ei in zip(todo, s, e):
            store[k] = (float(si), ei)
        if cache is not None:
            ks = list(store)
            np.savez(cache, keys=np.array(ks), sent=np.array([store[k][0] for k in ks], np.float32),
                     emb=np.stack([store[k][1] for k in ks]))
    sent = np.array([store[k][0] for k in keys], np.float32)
    emb = np.stack([store[k][1] for k in keys]) if keys else np.zeros((0, EMB_DIM), np.float32)
    return sent, emb


class CachedEncoder:
    """Serve headlines from the on-disk encode cache (written by training) and fall back to the real encoder
    only for unseen text, so the demo does not need to load FinBERT for replayed history."""

    def __init__(self, encoder, cache: Path | None):
        self.enc = encoder
        self.store: dict[str, tuple[float, np.ndarray]] = {}
        if cache is not None and Path(cache).exists():
            z = np.load(cache, allow_pickle=False)
            self.store = {k: (float(s), e) for k, s, e in zip(z["keys"], z["sent"], z["emb"])}

    @property
    def device(self):
        return getattr(self.enc, "device", "cache")

    def encode(self, headlines: list[str]):
        keys = [hashlib.sha1(h.encode()).hexdigest() for h in headlines]
        miss = [(k, h) for k, h in zip(keys, headlines) if k not in self.store]
        if miss:
            s, e = self.enc.encode([h for _, h in miss])
            for (k, _), si, ei in zip(miss, s, e):
                self.store[k] = (float(si), ei)
        return (np.array([self.store[k][0] for k in keys], np.float32), np.stack([self.store[k][1] for k in keys]))
