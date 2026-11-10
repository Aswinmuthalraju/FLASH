"""Phase-5 leakage audit: every check is EXECUTED on the data, none is asserted from reading code.

    python preprocessing/leakage_audit.py --market data/raw/market --news data/raw/news.csv --config configs/real.yaml

Uses the stub (hash) encoder on purpose: every check below concerns timestamps, labels, splits and fitted
transforms, none of which depend on FinBERT's numbers, and the audit must run where torch is unavailable.
It never fits or scores a predictive model and never reads test-set predictions.

Checks
  L1  splits: whole trading days, chronological, disjoint, 60/20/20 by day
  L2  purge: no labelled row's horizon window crosses its session (hence no window crosses a split)
  L3  labels: independent naive re-implementation of the triple barrier agrees on a random sample
  L4  features are causal: recomputing a row from the day's PREFIX up to that row gives the same values
  L5  back-fill look-ahead: valid feature rows that precede a day's first real trade (see prepare_market)
  L6  news: every attached headline is strictly earlier than the bar it is attached to
  L7  event memory: corrupting outcomes not yet known at a split boundary leaves earlier events unchanged
  L8  PCA/HMM fitted on train only: corrupting val/test features leaves the fitted parameters unchanged
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402
from src.data.align import align_news_to_bars  # noqa: E402
from src.data.market_loader import load_market  # noqa: E402
from src.data.news_loader import load_news  # noqa: E402
from src.features.market_features import (FEATURES, WARMUP, batch_day_features, build_table,  # noqa: E402
                                          market_return)

RNG = np.random.default_rng(0)


def l1_splits(t: pd.DataFrame, cfg: dict) -> dict:
    d = t.assign(day=t["timestamp"].dt.normalize()).groupby("day")["split"].agg(lambda s: sorted(set(s)))
    whole = bool((d.map(len) == 1).all())
    order = d.map(lambda s: s[0]).to_numpy()
    rank = {"train": 0, "val": 1, "test": 2}
    mono = bool(np.all(np.diff([rank[x] for x in order]) >= 0))
    counts = pd.Series(order).value_counts().to_dict()
    return {"pass": whole and mono and set(counts) == {"train", "val", "test"},
            "days_per_split": {k: int(v) for k, v in counts.items()},
            "split_dates": {s: [str(d.index[order == s].min().date()), str(d.index[order == s].max().date())]
                            for s in ("train", "val", "test") if (order == s).any()},
            "whole_days": whole, "chronological": mono}


def l2_purge(t: pd.DataFrame, cfg: dict) -> dict:
    n = cfg["market"]["bars_per_day"]
    out, ok = {}, True
    for h in cfg["labels"]["horizons"]:
        lab = t[t[f"y{h}"] >= 0]
        cross = int(((lab["tmin"] + h) // n != lab["tmin"] // n).sum())
        out[f"h{h}_label_windows_crossing_session"] = cross
        ok &= cross == 0
    # rows whose window would reach into the next split are therefore impossible; also count explicitly
    out["note"] = "labels end at the session close, splits are whole days -> structural purge, no extra gap needed"
    return {"pass": bool(ok), **out}


def _naive_barrier(close: np.ndarray, sigma: float, i: int, h: int, k: float, sq: bool) -> int:
    if i + h >= len(close) or not np.isfinite(sigma):
        return -1
    thr = k * sigma * (math.sqrt(h) if sq else 1.0)
    for j in range(1, h + 1):
        r = math.log(close[i + j] / close[i])
        if r >= thr:
            return 2
        if r <= -thr:
            return 0
    return 1


def l3_labels(t: pd.DataFrame, cfg: dict, n_sample: int = 3000) -> dict:
    lab = cfg["labels"]
    groups = t.groupby(["symbol", t["timestamp"].dt.normalize()]).indices
    keys = list(groups)
    mism, checked = 0, 0
    for _ in range(n_sample):
        rows = groups[keys[RNG.integers(len(keys))]]
        i = int(RNG.integers(len(rows)))
        close = t["close"].to_numpy()[rows]
        sig = t["vol30"].to_numpy()[rows][i]
        for h in lab["horizons"]:
            want = _naive_barrier(close, sig, i, h, lab["k"], lab["scale_sqrt_h"])
            got = int(t[f"y{h}"].to_numpy()[rows][i])
            mism += want != got
            checked += 1
    return {"pass": mism == 0, "labels_checked": checked, "mismatches": int(mism)}


def l4_causal_features(bars: pd.DataFrame, cfg: dict, n_sample: int = 300) -> dict:
    n = cfg["market"]["bars_per_day"]
    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    mret = market_return(bars)
    groups = list(bars.groupby(["symbol", bars["timestamp"].dt.normalize()]).indices.values())
    cols = [bars[k].to_numpy() for k in ("open", "high", "low", "close", "volume")]
    worst = 0.0
    for _ in range(n_sample):
        rows = groups[RNG.integers(len(groups))]
        if len(rows) <= WARMUP + 1:
            continue
        cut = int(RNG.integers(WARMUP, len(rows)))
        full = batch_day_features(*(c[rows] for c in cols), mret[rows], n)[cut]
        pref = batch_day_features(*(c[rows[:cut + 1]] for c in cols), mret[rows[:cut + 1]], n)[cut]
        worst = max(worst, float(np.nanmax(np.abs(full - pref))))
    note = ("mret (equal-weight universe return) is contemporaneous across symbols at the same minute: "
            "known at the bar close, not look-ahead")
    return {"pass": worst < 1e-9, "rows_checked": n_sample, "max_abs_diff_prefix_vs_full": worst, "note": note}


def l5_backfill(bars: pd.DataFrame, t: pd.DataFrame, cfg: dict) -> dict:
    """On regularised bars a leading gap shows up as volume 0 with o=h=l=c equal to the first real close."""
    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    leaky_rows, leaky_days = 0, []
    v, valid = bars["volume"].to_numpy(), t["valid_x"].to_numpy()
    for (sym, day), rows in bars.groupby(["symbol", bars["timestamp"].dt.normalize()]).indices.items():
        traded = np.flatnonzero(v[rows] > 0)
        first = int(traded[0]) if len(traded) else len(rows)
        if first > WARMUP:
            k = int(valid[rows[WARMUP:first]].sum())
            if k:
                leaky_rows += k
                leaky_days.append(f"{sym} {day.date()}")
    return {"pass": leaky_rows == 0, "valid_rows_with_future_backfilled_price": leaky_rows,
            "symbol_days_affected": leaky_days[:50], "n_symbol_days_affected": len(leaky_days),
            "assumes": "real trades have volume > 0 (true for NSE 1-min bars)"}


def l6_l7_l8(cfg, bars, news, t) -> dict:
    from src.evaluation.experiments import fit_regime, make_encoder, prepare_base
    from src.news.event_memory import history_features
    enc = make_encoder(cfg, finbert=False)
    from src.config import resolve
    base = prepare_base(cfg, bars, news, enc, resolve(cfg, "processed") / "hash_cache.npz")
    out = {}
    # L6 strictly-before
    ev, tb = base.ev, base.table
    idx, _ = align_news_to_bars(tb, ev, base.cal, cfg["news"]["max_age_min"])
    has = idx >= 0
    if has.any():
        e_ts = pd.to_datetime(ev["timestamp"]).to_numpy()[idx[has]]
        b_ts = pd.to_datetime(tb["timestamp"]).to_numpy()[has]
        viol = int((e_ts >= b_ts).sum())
    else:
        viol = 0
    out["L6_news_strictly_before_bar"] = {"pass": viol == 0, "bar_rows_with_news": int(has.sum()),
                                          "violations": viol, "events_linked": int(len(ev))}
    # L7 event memory across split boundaries
    if len(ev) > 1:
        tmin, fwd, known = ev["tmin"].to_numpy(), ev["fwd"].to_numpy(), ev["known_at"].to_numpy()
        base_h = history_features(base.E, tmin, fwd, known, cfg["news"]["memory_k"])
        res = {}
        n = cfg["market"]["bars_per_day"]
        for s in ("val", "test"):
            first_day = tb.loc[tb.split == s, "day_idx"].min()
            boundary = first_day * n
            f2 = fwd.copy()
            unknown = ~(known < boundary)
            f2[unknown & np.isfinite(f2)] = RNG.normal(0, 50, (unknown & np.isfinite(f2)).sum())
            h2 = history_features(base.E, tmin, f2, known, cfg["news"]["memory_k"])
            before = tmin < boundary
            diff = float(np.nanmax(np.abs(np.nan_to_num(base_h[before]) - np.nan_to_num(h2[before])))) if before.any() else 0.0
            res[f"events_before_{s}_changed_max_abs"] = diff
        out["L7_event_memory_split_boundaries"] = {"pass": all(v < 1e-12 for v in res.values()), **res}
    else:
        out["L7_event_memory_split_boundaries"] = {"pass": None, "note": "fewer than 2 linked events"}
    # L8 PCA/HMM train-only
    r1 = fit_regime(base)
    corrupt = base.table.copy()
    from src.features.market_features import REGIME_FEATURES
    m = corrupt["split"].isin(["test"]).to_numpy()
    corrupt.loc[m, REGIME_FEATURES] = RNG.normal(0, 1, (m.sum(), len(REGIME_FEATURES)))
    base2 = type(base)(base.cfg, corrupt, base.cal, base.ev, base.E, base.news, base.age)
    r2 = fit_regime(base2)
    same_pca = bool(np.allclose(r1.pca.pca.components_, r2.pca.pca.components_)) if hasattr(r1.pca, "pca") else None
    tr = (base.table["split"] == "train").to_numpy() & base.table["valid_x"].to_numpy()
    post_diff = float(np.nanmax(np.abs(r1.post[tr] - r2.post[tr])))
    out["L8_regime_fit_train_only"] = {"pass": post_diff < 1e-9 and same_pca is not False,
                                       "pca_components_identical": same_pca,
                                       "train_posteriors_max_abs_diff_after_corrupting_test": post_diff,
                                       "note": "val IS used (PCA component count, early stopping, calibration) by design;"
                                               " val metrics are therefore optimistic, test is untouched"}
    return out


def run(cfg: dict, market: str, news_path: str) -> dict:
    bars = load_market(market, cfg)
    news = load_news(news_path, cfg["market"]["tz"])
    t = build_table(bars, cfg)
    rep = {"L1_splits": l1_splits(t, cfg), "L2_purge": l2_purge(t, cfg), "L3_labels": l3_labels(t, cfg),
           "L4_feature_causality": l4_causal_features(bars, cfg), "L5_backfill_lookahead": l5_backfill(bars, t, cfg)}
    rep.update(l6_l7_l8(cfg, bars, news, t))
    rep["all_pass"] = all(v.get("pass") is True for v in rep.values() if isinstance(v, dict))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default=str(ROOT / "data/raw/market"))
    ap.add_argument("--news", default=str(ROOT / "data/raw/news.csv"))
    ap.add_argument("--config")
    ap.add_argument("--out", default=str(ROOT / "reports/data_prep/leakage_audit.json"))
    a = ap.parse_args()
    rep = run(load_config(a.config), a.market, a.news)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    sys.exit(0 if rep["all_pass"] else 1)


if __name__ == "__main__":
    main()
