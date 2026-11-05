"""Ablation: remove news, historical events, HMM and PCA one at a time from the full model D."""
from __future__ import annotations

import pandas as pd

from src.evaluation.experiments import Dataset, assemble, fit_regime, fit_variant, score, variant_cols


def run_ablation(ds: Dataset, h: int) -> pd.DataFrame:
    g = ds.groups
    full = variant_cols(ds, "D")
    def drop(*groups):
        gs = set(groups) | ({"derived_x"} if set(groups) & {"news", "history", "regime"} else set())
        return [c for c in full if not any(c in g[x] for x in gs)]
    rows = []

    def add(label, f, d):
        s = score(d, f)
        rows.append({"horizon": h, "variant": label, **{f"all_{k}": v for k, v in s["all"].items() if k != "n"},
                     **{f"shock_{k}": v for k, v in s["shock"].items() if k in ("macro_f1", "auc_ovr", "brier")}})

    add("D (full)", fit_variant(ds, "D", h), ds)
    add("- news", fit_variant(ds, "D", h, cols=drop("news")), ds)
    add("- historical events", fit_variant(ds, "D", h, cols=drop("history")), ds)
    add("- HMM regime", fit_variant(ds, "D", h, cols=drop("regime")), ds)
    # no PCA: the HMM is refitted on the standardised raw features (GMM still initialises it)
    ds_nopca = assemble(ds.base, fit_regime(ds.base, use_pca=False))
    add("- PCA", fit_variant(ds_nopca, "D", h), ds_nopca)
    return pd.DataFrame(rows)
