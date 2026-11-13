"""FLASH D6 -- ONE-SHOT final test-split evaluation of the frozen D4 models (models_real/stage_d4_mac).
No fitting, no calibration, no threshold tuning on test. Refuses to run twice (single-shot guard).
Run once, with the project owner's approval (given in-session 2026-11-13)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import joblib, numpy as np
from sklearn.metrics import roc_auc_score
import evaluation.baselines as BL
from evaluation.metrics_ext import full_report
from src.config import load_config
from src.data.build import get_data
from src.evaluation import experiments as X
from src.models.fusion_lgbm import load_booster

OUT = Path(__file__).parent / "d6_test_metrics.json"
if OUT.exists():
    raise SystemExit("D6 is single-shot: output already exists")

class CacheOnly:
    device = "cache-only"
    def encode(self, h): raise RuntimeError("missing FinBERT cache entries")

def sm(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)

cfg = load_config(ROOT / "configs" / "real.yaml")
bundle = joblib.load(ROOT / "models_real" / "stage_d4_mac" / "bundle.joblib")
bars, news = get_data(cfg, "data/raw/market", "data/raw/news.csv")
ds = X.prepare(cfg, bars, news, CacheOnly(), ROOT / "data/processed_real/finbert_cache.npz")
t = ds.table
cols_A = X.variant_cols(ds, "A")
res = {"split": "test", "model_dir": "models_real/stage_d4_mac", "probabilities": "uncalibrated", "horizons": {}}
for h in cfg["labels"]["horizons"]:
    te = ds.rows("test", h); y = ds.y(te, h); Xt = ds.XF[te]
    a, d = bundle["A"][h], bundle["D"][h]
    P_A = sm(Xt[:, cols_A] @ a["W"].T + a["b"])
    lg = d["logit"]
    P_D = (1 - lg["w"]) * load_booster(d["model"]).predict(Xt) + lg["w"] * sm(np.nan_to_num(Xt) @ lg["W"].T + lg["b"])
    base = BL.Baselines().fit(t, h).predict(t, te)
    r = {"n": int(len(y)), "support": np.bincount(y, minlength=3).tolist(), "models": {}, "baselines": {}}
    mv = y != 1
    for name, P in [("A", P_A), ("D", P_D)]:
        rep = full_report(P, y)
        rep["pred_share"] = (np.bincount(P.argmax(1), minlength=3) / len(y)).tolist()
        rep["move_auc"] = float(roc_auc_score(mv, 1 - P[:, 1]))
        pu = P[mv, 2] / (P[mv, 2] + P[mv, 0] + 1e-12)
        rep["dir_auc"] = float(roc_auc_score(y[mv] == 2, pu))
        r["models"][name] = rep
    for n, P in base.items():
        r["baselines"][n] = full_report(P, y)
    res["horizons"][str(h)] = r
    print(h, {k: round(r["models"]["D"][k], 4) for k in ("auc_ovr", "brier", "macro_f1", "move_auc", "dir_auc")},
          "prior brier", round(r["baselines"]["prior"]["brier"], 4) if "prior" in r["baselines"] else list(r["baselines"]))
OUT.write_text(json.dumps(res, indent=2, default=str))
print("wrote", OUT)
