"""Console demo: replay one test day of one stock and show how the prediction moves around news events.

    python scripts/demo.py [--symbol TCS] [--day N] [--horizon 15] [--stub-encoder]
"""
import _common  # noqa: F401
import argparse

import pandas as pd

from src.config import load_config, resolve
from src.data.align import split_days
from src.data.build import get_data
from src.evaluation.experiments import make_encoder
from src.inference.predictor import load_bundle, load_memory
from src.inference.session import ReplayContext, around_news, run_day
from src.news.sentiment import CachedEncoder


def load_demo_assets(cfg, stub=False):
    bars, news = get_data(cfg)
    ctx = ReplayContext(bars, news, cfg)
    cache = resolve(cfg, "processed") / ("hash_cache.npz" if stub else "finbert_cache.npz")
    enc = CachedEncoder(make_encoder(cfg, finbert=not stub), cache)
    return ctx, load_bundle(cfg), load_memory(cfg), enc


def pick_day_and_symbol(ctx, cfg, symbol=None, day_no=None):
    test_days = list(split_days(pd.DatetimeIndex(ctx.days()), cfg)["test"])
    cands = []
    for d in ([test_days[day_no]] if day_no is not None else test_days):
        t0 = d + pd.Timedelta(hours=9, minutes=45)
        t1 = d + pd.Timedelta(hours=14)
        n = ctx.news[(ctx.news["timestamp"] > t0) & (ctx.news["timestamp"] < t1)]
        for s in ([symbol] if symbol else cfg["symbols"]):
            from src.news.entities import EntityLinker
            lk = EntityLinker(market_wide_relevance=cfg["news"]["market_wide_relevance"])
            from src.news.event_rules import classify_event
            k = sum(1 for h in n["headline"] if s in lk.link(h, classify_event(h), cfg["symbols"]))
            cands.append((k, d, s))
    cands.sort(key=lambda x: -x[0])
    return cands[0][1], cands[0][2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol"); ap.add_argument("--day", type=int)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--stub-encoder", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    H = a.horizon or cfg["labels"]["primary_horizon"]
    ctx, bundle, mem, enc = load_demo_assets(cfg, a.stub_encoder)
    day, sym = pick_day_and_symbol(ctx, cfg, a.symbol, a.day)
    sess = run_day(ctx, bundle, mem, enc, sym, day)
    tl = sess.timeline
    print(f"\n{sym} on {day:%Y-%m-%d}: {len(tl)} predicted bars; horizon {H}; gate {sess.pred.gate}")
    print(f"hot-loop latency per bar (3 horizons): median {tl.latency_us.median():.0f} us, p95 {tl.latency_us.quantile(.95):.0f} us, "
          f"max {tl.latency_us.max():.0f} us")
    pd.set_option("display.width", 200)
    for _, n in sess.news[(sess.news.timestamp >= tl.timestamp.iloc[0])].head(3).iterrows():
        print(f"\n--- NEWS {n.timestamp:%H:%M:%S}  [{n.etype}]  FinBERT sentiment {n.sentiment:+.2f}  relevance {n.relevance:.1f}")
        print(f"    \"{n.headline}\"   (history: mean fwd {n.hist_mean:+.2f} sigma, similarity {n.hist_sim:.2f})")
        snap = around_news(sess, n.timestamp, H)
        if len(snap):
            print(snap.drop(columns="_k").round(3).to_string(index=False))
            k = int(snap.loc[snap.offset_min == 1, "_k"].iloc[0]) if (snap.offset_min == 1).any() else int(snap["_k"].iloc[0])
            ex = sess.pred.explain(H, x=sess.X[k])
            g = ", ".join(f"{g}={v:+.2f}" for g, v in ex["group_contribution"].items())
            print(f"    why ({ex['class']}, 1 min after): {g}")
            print("    top: " + "; ".join(f"{f['feature']}[{f['group']}] {f['contribution']:+.2f}" for f in ex["top_features"][:5]))


if __name__ == "__main__":
    main()
