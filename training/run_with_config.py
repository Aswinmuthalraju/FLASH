"""Run an UNMODIFIED repo script (scripts/train.py, scripts/evaluate.py, ...) against another config file.

    python training/run_with_config.py configs/real.yaml scripts/train.py --market data/raw/market --news data/raw/news.csv

The original scripts call load_config() with no argument; we rebind that default before running them,
so every path (processed cache, models/, reports/) comes from the chosen config. No repo file is edited.
"""
import functools
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import src.config as C  # noqa: E402

cfg_path, script, *rest = sys.argv[1:]
C.load_config = functools.partial(C.load_config.__wrapped__ if hasattr(C.load_config, "__wrapped__") else C.load_config,
                                  str(ROOT / cfg_path))
sys.argv = [script, *rest]
runpy.run_path(str(ROOT / script), run_name="__main__")
