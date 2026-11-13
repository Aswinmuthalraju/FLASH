"""Build the FinBERT cache (data/processed_real/finbert_cache.npz) in a process that never imports LightGBM.

Why: on macOS, importing lightgbm/hmmlearn before torch loads two OpenMP runtimes and the process dies without a
traceback (seen in preprocessing/verify_dataset.py). This script touches only torch + transformers, so the later
training run can read the cache and never needs to load FinBERT.

The cache format and keys are exactly those of src.news.sentiment.encode_cached (sha1 of the headline text), and
it encodes every headline in the news file, as prepare() does. It is resumable: chunks already cached are skipped.

    python preprocessing/build_finbert_cache.py [--config configs/real.yaml] [--news data/raw/news.csv]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/real.yaml")
    ap.add_argument("--news", default="data/raw/news.csv")
    ap.add_argument("--chunk", type=int, default=2000)
    a = ap.parse_args()

    from src.config import load_config, resolve
    from src.data.news_loader import load_news
    from src.news.sentiment import FinBertEncoder, encode_cached

    cfg = load_config(a.config)
    assert "lightgbm" not in sys.modules, "lightgbm must not be loaded in this process"
    news = load_news(a.news, cfg["market"]["tz"])
    heads = news["headline"].tolist()
    cache = resolve(cfg, "processed") / "finbert_cache.npz"
    print(f"{len(heads):,} headlines -> {cache}")
    enc = FinBertEncoder(cfg["news"]["finbert"], cfg["news"]["batch_size"])
    print(f"device: {enc.device}")
    t0 = time.time()
    for i in range(0, len(heads), a.chunk):
        encode_cached(enc, heads[i:i + a.chunk], cache)      # saves after every chunk
        print(f"  {min(i + a.chunk, len(heads)):>6,}/{len(heads):,}  [{time.time() - t0:.0f}s]", flush=True)
    encode_cached(enc, heads, cache)                          # no-op when complete; verifies full coverage
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
