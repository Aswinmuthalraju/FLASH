"""Latency benchmark of the real hot loop (CPU, single bar at a time, time.perf_counter_ns).

    python scripts/benchmark.py [--n 10000] [--warmup 2000]
"""
import _common  # noqa: F401
import argparse

import numpy as np

from src.config import load_config, resolve
from src.data.align import assign_split, split_days
from src.data.build import get_data
from src.evaluation.report import df_to_md
from src.evaluation.latency import benchmark, latency_table, make_stream
from src.features.news_features import raw_vector
from src.inference.predictor import Predictor, load_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--warmup", type=int, default=2_000)
    ap.add_argument("--market")
    ap.add_argument("--news")
    a = ap.parse_args()
    cfg = load_config()
    bars, _ = get_data(cfg, a.market, a.news)
    bundle = load_bundle(cfg)
    days = split_days(bars["timestamp"].dt.normalize().unique(), cfg)["test"]
    stream = make_stream(bars, cfg, days)
    news_raw = raw_vector(0.8, "Earnings", 1.0, (0.4, 0.5, 0.9))
    res = {}
    for label, mode, hs in [("A  logistic (3 horizons)", "A", None), ("D  full model (3 horizons)", "D", None),
                            ("D  full model (h=15 only)", "D", [15])]:
        p = Predictor(bundle, mode, hs)
        print(f"benchmarking {label} ...")
        res[label] = benchmark(p, stream, a.n, a.warmup, news_raw)
    tab = latency_table(res)
    print(f"\nPer-bar latency in microseconds ({a.n:,} bars after {a.warmup:,} warm-up; stages are summed over horizons)\n")
    print(tab.to_string(index=False))
    out = resolve(cfg, "reports")
    tab.to_csv(out / "latency.csv", index=False)
    (out / "latency.md").write_text(df_to_md(tab, "{:.2f}"), encoding="utf-8")
    ta = res["A  logistic (3 horizons)"]["total_incl_call"]
    td = res["D  full model (3 horizons)"]["total_incl_call"]
    print(f"\nD / A cost (median): {td['median'] / ta['median']:.1f}x   (p99: {td['p99'] / ta['p99']:.1f}x)")


if __name__ == "__main__":
    main()
