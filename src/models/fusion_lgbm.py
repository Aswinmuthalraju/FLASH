"""Unit 5 (ensemble): LightGBM multiclass fusion of market features, HMM posteriors, cached news features
and historical-event impact. Boosting = additive ensemble of trees fitted stage-wise on the multinomial deviance.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np


class FusionLGBM:
    def __init__(self, lgbm_cfg: dict, seed: int = 7):
        p = dict(lgbm_cfg)
        self.early = p.pop("early_stopping_rounds", 30)
        self.params = dict(objective="multiclass", num_class=3, random_state=seed, verbose=-1, n_jobs=-1, **p)
        self.model: lgb.LGBMClassifier | None = None
        self.booster: lgb.Booster | None = None

    def fit(self, Xtr, ytr, Xva, yva) -> "FusionLGBM":
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="multi_logloss",
                       callbacks=[lgb.early_stopping(self.early, verbose=False)])
        self.booster = self.model.booster_
        return self

    @property
    def best_iteration(self) -> int:
        return int(self.model.best_iteration_ or self.params["n_estimators"])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X, num_iteration=self.best_iteration)

    def model_string(self) -> str:
        return self.booster.model_to_string(num_iteration=self.best_iteration)

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """SHAP-style per-feature contributions via LightGBM pred_contrib: (n, 3 classes, n_features + 1 bias)."""
        c = self.booster.predict(np.atleast_2d(x), pred_contrib=True, num_iteration=self.best_iteration)
        return c.reshape(len(c), 3, -1)


def load_booster(model_str: str) -> lgb.Booster:
    return lgb.Booster(model_str=model_str)


class FusionBlend:
    """LightGBM + L2 logistic probability blend (the logistic leg adds a smooth, low-variance linear component).
    P = (1 - w) * P_lgbm + w * P_logit. Temperature calibration is applied afterwards by the Calibrator."""

    def __init__(self, fusion_cfg: dict, seed: int = 7):
        self.gbm = FusionLGBM(fusion_cfg["lgbm"], seed)
        self.w = float(fusion_cfg.get("blend_logit_weight", 0.0))
        self.C = float(fusion_cfg.get("logit_C", 0.05))
        self.logit = None

    def fit(self, Xtr, ytr, Xva, yva) -> "FusionBlend":
        from src.models.baseline_logistic import BaselineLogistic

        self.gbm.fit(Xtr, ytr, Xva, yva)
        if self.w > 0:
            self.logit = BaselineLogistic(C=self.C, max_iter=300).fit(np.nan_to_num(Xtr), ytr)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        P = self.gbm.predict_proba(X)
        if self.logit is not None:
            P = (1 - self.w) * P + self.w * self.logit.predict_proba(np.nan_to_num(X))
        return P

    @property
    def best_iteration(self) -> int:
        return self.gbm.best_iteration

    def model_string(self) -> str:
        return self.gbm.model_string()

    def contributions(self, x):
        return self.gbm.contributions(x)

    def export_logit(self) -> dict | None:
        return None if self.logit is None else {**self.logit.export(), "w": self.w}
