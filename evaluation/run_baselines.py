"""Score the reference baselines with the extended metrics + day-block bootstrap CIs.

    python evaluation/run_baselines.py --config configs/real.yaml --market data/raw/market      # validation split
    python evaluation/run_baselines.py --synthetic                                             # repo's synthetic data

TEST-SET LOCK: --split test is refused unless BOTH --unlock-test is passed AND the environment variable
FLASH_TEST_UNLOCK=I_HAVE_APPROVAL is set. The test split is meant to be scored once, after every decision is frozen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.baselines import Baselines, rows  # noqa: E402
from evaluation.metrics_ext import block_bootstrap, full_report  # noqa: E402
from src.config import load_config  # noqa: E402
from src.features.market_features import build_table  # noqa: E402


def check_split_allowed(split: str, unlock: bool) -> None:
    if split == "test" and not (unlock and os.environ.get("FLASH_TEST_UNLOCK") == "I_HAVE_APPROVAL"):
        raise SystemExit("refusing to score the TEST split: pass --unlock-test AND set FLASH_TEST_UNLOCK=I_HAVE_APPROVAL")


def run(cfg: dict, bars, split: str = "val", B: int = 1000) -> dict:
    t = build_table(bars, cfg)
    out = {"split": split, "horizons": {}}
    for h in cfg["labels"]["horizons"]:
        bl = Baselines().fit(t, h)
        idx = rows(t, split, h)
        y = t[f"y{h}"].to_numpy()[idx].astype(int)
        day = t["day_idx"].to_numpy()[idx]
        preds = bl.predict(t, idx)
        res = {"fit_on_train": bl.describe(), "models": {}}
        for name, P in preds.items():
            r = full_report(P, y)
            r["bootstrap"] = block_bootstrap(P, y, day, B=B, P_ref=preds["prior"] if name != "prior" else None)
            res["models"][name] = r
        out["horizons"][h] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--market")
    ap.add_argument("--synthetic", action="store_true", help="use the repo's committed synthetic bars")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--unlock-test", action="store_true")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    check_split_allowed(a.split, a.unlock_test)
    cfg = load_config(a.config)
    if a.synthetic:
        from src.data.store import load_parquet
        bars, src = load_parquet(cfg, "bars"), "SYNTHETIC (repo's committed data/processed/bars.parquet)"
    else:
        from src.data.market_loader import load_market
        bars, src = load_market(a.market, cfg), f"market dir {a.market}"
    rep = run(cfg, bars, a.split, a.B)
    rep["data_source"] = src
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2, default=str))
    for h, r in rep["horizons"].items():
        for n, m in r["models"].items():
            ci = m["bootstrap"]["ci"]
            print(f"h={h:>2} {n:9s} macroF1 {m['macro_f1']:.3f} [{ci['macro_f1'][0]:.3f},{ci['macro_f1'][1]:.3f}]  "
                  f"AUC {m['auc_ovr']:.3f}  Brier {m['brier']:.4f} [{ci['brier'][0]:.4f},{ci['brier'][1]:.4f}]  "
                  f"ECE {m['ece']:.3f}")


if __name__ == "__main__":
    main()
