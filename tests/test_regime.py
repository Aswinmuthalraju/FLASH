from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from src.features.market_features import FEATURES, build_table
from src.models.hmm_regime import HMMRegime, group_lengths
from src.models.regime_pca_gmm import RegimePCA


def _prep(cfg, synth):
    bars, _, _ = synth
    t = build_table(bars, cfg)
    t = t[t.valid_x].reset_index(drop=True)
    return t


def test_pca_components_chosen_on_validation(cfg, synth):
    t = _prep(cfg, synth)
    tr, va = t[t.split == "train"], t[t.split == "val"]
    p = RegimePCA(0.90).fit(tr[FEATURES].to_numpy(), va[FEATURES].to_numpy())
    assert 1 <= p.n_components_ <= len(FEATURES)
    assert p.val_explained_ >= 0.90
    # one fewer component would NOT reach the target (it is the smallest)
    Xv = p._scale(va[FEATURES].to_numpy()); mu, comp = p.pca.mean_, p.pca.components_
    k = p.n_components_ - 1
    if k >= 1:
        Zv = (Xv - mu) @ comp.T
        ev = 1 - (((Xv - mu) - Zv[:, :k] @ comp[:k]) ** 2).sum() / ((Xv - mu) ** 2).sum()
        assert ev < 0.90
    e = p.export()
    Xs = np.clip((va[FEATURES].to_numpy() - e["mean"]) / e["scale"], -e["clip"], e["clip"])
    assert np.allclose(Xs @ e["P"] + e["c"], p.transform(va[FEATURES].to_numpy()))


def _fit_hmm(cfg, t):
    tr, va = t[t.split == "train"], t[t.split == "val"]
    p = RegimePCA(0.90).fit(tr[FEATURES].to_numpy(), va[FEATURES].to_numpy())
    Z = p.transform(tr[FEATURES].to_numpy())
    L = group_lengths((tr.symbol + tr.day_idx.astype(str)).to_numpy())
    return p, HMMRegime(4, seed=1, n_iter=15).fit(Z, L), Z, L


def test_forward_step_matches_hmmlearn_forward_lattice(cfg, synth):
    t = _prep(cfg, synth)
    p, h, Z, L = _fit_hmm(cfg, t)
    seq = Z[: L[0]]
    mine = h.filter_posteriors(seq, [len(seq)])
    m = h.model
    # independent reference: textbook log-space forward pass on hmmlearn's own emission log-probs
    lb = m._compute_log_likelihood(seq)
    la = np.log(m.startprob_) + lb[0]
    fw = [la]
    for t_ in range(1, len(seq)):
        la = logsumexp(la[:, None] + np.log(m.transmat_), axis=0) + lb[t_]
        fw.append(la)
    fw = np.array(fw)
    ref = np.exp(fw - logsumexp(fw, axis=1, keepdims=True))
    f = h.new_filter()
    for z in seq:
        f.step(z)
    assert np.isclose(f.loglik, m.score(seq), rtol=1e-9)        # scaling factors give the exact evidence
    assert np.allclose(mine, ref, atol=1e-8)
    assert np.allclose(mine.sum(1), 1.0)


def test_filter_is_causal(cfg, synth):
    """P(state_t | z_1..t) must not move when later observations change."""
    t = _prep(cfg, synth)
    p, h, Z, L = _fit_hmm(cfg, t)
    seq = Z[: L[0]].copy()
    a = h.filter_posteriors(seq, [len(seq)])
    seq2 = seq.copy()
    seq2[100:] += 3.0
    b = h.filter_posteriors(seq2, [len(seq2)])
    assert np.array_equal(a[:100], b[:100])
    assert not np.allclose(a[100:], b[100:])


def test_viterbi_is_not_used_in_live_or_feature_code():
    root = Path(__file__).resolve().parents[1] / "src"
    for sub in ("inference", "features"):
        for f in (root / sub).glob("*.py"):
            assert "viterbi" not in f.read_text(encoding="utf-8").lower(), f
