"""THE EDITABLE FILE of the FLASH autoresearch loop. Baseline == the current production recipe (variant D).

Contract: predict(split, ds) -> calibrated probabilities for split.Xdev, shape (n, 3), classes DOWN/NEUTRAL/UP.
Free to change: feature engineering on split.X*, model family, ensembling, hyperparameters, calibration.
Do not touch prepare.py, the test split, or the label definition.
"""
import numpy as np

from prepare import run
from src.models.calibration import Calibrator
from src.models.fusion_lgbm import FusionLGBM


def predict(sp, ds):
    cfg = ds.base.cfg
    m = FusionLGBM(cfg["fusion"]["lgbm"], cfg["seed"]).fit(sp.Xtr, sp.ytr, sp.Xfit, sp.yfit)
    cal = Calibrator(cfg["fusion"]["calibration"]).fit(m.predict_proba(sp.Xfit), sp.yfit)
    return cal.transform(m.predict_proba(sp.Xdev))


if __name__ == "__main__":
    run(predict)
