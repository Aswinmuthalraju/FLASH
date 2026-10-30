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
