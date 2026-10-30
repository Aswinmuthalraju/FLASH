"""Train the final models (D = full fusion, A = logistic baseline) for every horizon and save the bundle.

    python scripts/train.py                       # synthetic data, real FinBERT
    python scripts/train.py --market data/raw/prices --news data/raw/news.csv
    python scripts/train.py --stub-encoder        # TESTING ONLY: hash encoder instead of FinBERT
"""
import _common  # noqa: F401
import argparse
import time

from src.config import load_config, resolve
from src.data.build import get_data
from src.evaluation.experiments import fit_variant, make_encoder, prepare, score
from src.features.news_features import memory_from_events
from src.inference.predictor import make_bundle, save_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market")
    ap.add_argument("--news")
    ap.add_argument("--rebuild", action="store_true", help="regenerate the synthetic data")
    ap.add_argument("--stub-encoder", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    t0 = time.time()
    bars, news = get_data(cfg, a.market, a.news, a.rebuild)
    print(f"data: {len(bars):,} bars, {bars.symbol.nunique()} symbols, {len(news):,} headlines")
    enc = make_encoder(cfg, finbert=not a.stub_encoder)
    ds = prepare(cfg, bars, news, enc, resolve(cfg, "processed") / ("hash_cache.npz" if a.stub_encoder else "finbert_cache.npz"))
    print(f"encoder device: {enc.device}")
    r = ds.regime
    print(f"PCA: {r.pca.n_components_} components, validation explained variance {r.pca.val_explained_:.3f}")
    print(f"HMM states: {r.names}")
    finals = {"D": {}, "A": {}}
    for h in cfg["labels"]["horizons"]:
        for name in ("A", "D"):
            f = fit_variant(ds, name, h)
            finals[name][h] = f
            m = score(ds, f)["all"]
            extra = f", {f.model.best_iteration} trees" if name == "D" else ""
            print(f"  h={h:>2} {name}: macroF1 {m['macro_f1']:.3f}  AUC {m['auc_ovr']:.3f}  Brier {m['brier']:.4f}{extra}")
    bundle = make_bundle(ds, finals)
    mem = memory_from_events(ds.base.ev, ds.base.E, cfg["news"]["memory_k"])
    path = save_bundle(bundle, mem, cfg)
    print(f"saved {path}  (event memory: {len(mem)} past events)   [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
