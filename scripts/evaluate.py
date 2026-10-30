"""Run every experiment and write reports/ (tables, plots, REPORT.md). Test-set numbers only.

    python scripts/evaluate.py [--stub-encoder] [--market ... --news ...]
"""
import _common  # noqa: F401
import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config, resolve
from src.data.build import get_data
from src.data.store import load_parquet
from src.evaluation.ablation import run_ablation
from src.evaluation.backtest import sweep
from src.evaluation.experiments import SHOCK_MINUTES, make_encoder, prepare, results_table, run_experiments
from src.evaluation.metrics import conf_matrix, plot_calibration, plot_confusion
from src.evaluation.report import df_to_md
from src.features.market_features import FEATURES, IDX, REGIME_FEATURES


def regime_analysis(ds, out_dir, cfg):
    """OFFLINE: Viterbi decoding of the test split (uses future data, so never a feature)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import adjusted_rand_score

    from src.models.hmm_regime import group_lengths

    t, hmm, pca = ds.table, ds.regime.hmm, ds.regime.pca
    m = ((t["split"] == "test") & t["valid_x"]).to_numpy()
    idx = np.flatnonzero(m)
    X = t[FEATURES].to_numpy()[idx]
    XR = t[REGIME_FEATURES].to_numpy()[idx]
    keys = (t["symbol"].astype(str) + "_" + t["day_idx"].astype(str)).to_numpy()[idx]
    lengths = group_lengths(keys)
    path = hmm.viterbi(pca.transform(XR), lengths)
    names = ds.regime.names
    runs = np.diff(np.flatnonzero(np.r_[True, path[1:] != path[:-1], True]))
    run_state = path[np.r_[True, path[1:] != path[:-1]]]
    rows = []
    for s, n in enumerate(names):
        sel = path == s
        rows.append({"state": n, "share": sel.mean(), "vol30_bps": X[sel, IDX["vol30"]].mean() * 1e4 if sel.any() else np.nan,
                     "drift_bps_per_bar": X[sel, IDX["mean20"]].mean() * 1e4 if sel.any() else np.nan,
                     "avg_run_bars": runs[run_state == s].mean() if (run_state == s).any() else np.nan})
    stats = pd.DataFrame(rows)
    ari = None
    try:
        truth = load_parquet(cfg, "truth")
        key = truth.assign(d=truth["day"].dt.normalize())
        day_ts = t["timestamp"].dt.normalize().to_numpy()[idx]
        mins = ((t["tmin"].to_numpy() % cfg["market"]["bars_per_day"]).astype(int))[idx]
        lut = {(d, mi): r for d, mi, r in zip(key["d"], key["minute"], key["regime"])}
        tr = np.array([lut[(pd.Timestamp(d), mi)] for d, mi in zip(day_ts, mins)])
        ari = float(adjusted_rand_score(tr, path))
    except Exception:
        pass
    # plot one test day of one symbol: price + Viterbi shading + filtered posteriors
    sym = cfg["symbols"][0]
    d0 = t.loc[(t["split"] == "test"), "day_idx"].iloc[0]
    sel = np.flatnonzero(((t["symbol"] == sym) & (t["day_idx"] == d0) & t["valid_x"]).to_numpy())
    vp = hmm.viterbi(pca.transform(t[REGIME_FEATURES].to_numpy()[sel]), [len(sel)])
    post = ds.regime.post[sel]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    cols = plt.cm.Set2(np.arange(len(names)))
    ax[0].plot(t["close"].to_numpy()[sel], color="k", lw=1)
    for k in range(len(sel)):
        ax[0].axvspan(k - .5, k + .5, color=cols[vp[k]], alpha=.35, lw=0)
    ax[0].set_title(f"{sym} test day {d0}: close with OFFLINE Viterbi regimes (shading) -- "
                    + ", ".join(f"{n}" for n in names), fontsize=9)
    for s, n in enumerate(names):
        ax[1].plot(post[:, s], color=cols[s], label=n)
    ax[1].set_ylabel("filtered P(state)"); ax[1].legend(ncol=len(names), fontsize=8); ax[1].set_xlabel("bar (after 30-bar warm-up)")
    fig.tight_layout(); fig.savefig(out_dir / "regimes_viterbi_offline.png", dpi=110); plt.close(fig)
    return stats, ari


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market"); ap.add_argument("--news")
    ap.add_argument("--stub-encoder", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    out = resolve(cfg, "reports")
    H = cfg["labels"]["primary_horizon"]
    bars, news = get_data(cfg, a.market, a.news)
    enc = make_encoder(cfg, finbert=not a.stub_encoder)
    ds = prepare(cfg, bars, news, enc, resolve(cfg, "processed") / ("hash_cache.npz" if a.stub_encoder else "finbert_cache.npz"))
    t = ds.table
    ntr, nva, nte = (int(((t["split"] == s) & t["valid_x"]).sum()) for s in ("train", "val", "test"))
    print(f"rows train/val/test = {ntr:,}/{nva:,}/{nte:,}; PCA {ds.regime.pca.n_components_} comps; states {ds.regime.names}")

    # ---------------------------------------------------------------- experiments A-D
    fitted, res = run_experiments(ds)
    all_t, shock_t = results_table(res, "all"), results_table(res, "shock")
    keep = ["horizon", "model", "n", "macro_f1", "auc_ovr", "brier", "brier_uncalibrated", "ece", "accuracy"]
    all_t[keep].to_csv(out / "experiments_all.csv", index=False)
    shock_t.to_csv(out / "experiments_shock.csv", index=False)
    print("\n=== Experiments A-D, all test bars ===\n" + all_t[keep].round(4).to_string(index=False))
    print(f"\n=== News-shock subset (bars <= {SHOCK_MINUTES} min after a news event) ===\n" + shock_t.round(4).to_string(index=False))

    # B vs D on shock subset
    bd = []
    for h in res:
        b, d = res[h]["B"]["shock"], res[h]["D"]["shock"]
        bd.append({"horizon": h, "n_shock": d["n"], "macroF1_B": b["macro_f1"], "macroF1_D": d["macro_f1"],
                   "AUC_B": b["auc_ovr"], "AUC_D": d["auc_ovr"], "Brier_B": b["brier"], "Brier_D": d["brier"]})
    bd = pd.DataFrame(bd)
    print("\n=== Shock subset, B vs D ===\n" + bd.round(4).to_string(index=False))
    bd.to_csv(out / "shock_B_vs_D.csv", index=False)

    # -------------------------------------------------------------------- plots (primary horizon)
    fD, fA = fitted[("D", H)], fitted[("A", H)]
    y = ds.y(fD.rows_test, H)
    plot_calibration({"D uncalibrated": fD.P_test_raw, "D calibrated": fD.P_test_cal, "A calibrated": fA.P_test_cal}, y,
                     out / f"calibration_h{H}.png", f"Reliability, horizon {H} (test)")
    plot_confusion(conf_matrix(fD.P_test_cal, y), out / f"confusion_D_h{H}.png", f"D, h={H}")
    plot_confusion(conf_matrix(fA.P_test_cal, y), out / f"confusion_A_h{H}.png", f"A, h={H}")

    # ---------------------------------------------------------------------------------- ablation
    abl = run_ablation(ds, H)
    abl.to_csv(out / "ablation.csv", index=False)
    print(f"\n=== Ablation (horizon {H}) ===\n" + abl.round(4).to_string(index=False))

    # ----------------------------------------------------------------------------------- backtest
    bt_cfg = cfg["backtest"]
    ths = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    bts = {}
    for name in ("B", "D"):
        f = fitted[(name, H)]
        bts[name] = sweep(t, f.rows_test, f.P_test_cal, H, ths, bt_cfg["cost_bps_per_side"], bt_cfg["entry_delay_bars"])
        bts[name].insert(0, "model", name)
        print(f"\n=== Backtest {name}, h={H}, cost {bt_cfg['cost_bps_per_side']} bps/side ===\n" + bts[name].round(4).to_string(index=False))
    pd.concat(bts.values()).to_csv(out / "backtest.csv", index=False)

    # ---------------------------------------------------------------------------------- regimes
    stats, ari = regime_analysis(ds, out, cfg)
    print("\n=== Regimes on test (Viterbi, OFFLINE) ===\n" + stats.round(3).to_string(index=False))
    if ari is not None:
        print(f"adjusted Rand index vs. the synthetic latent regimes: {ari:.3f}")
    stats.to_csv(out / "regimes.csv", index=False)

    # ----------------------------------------------------------------------------------- REPORT.md
    p = ds.regime.pca
    md = [f"# flash-market evaluation (test split only)\n",
          f"Rows train/val/test: {ntr:,}/{nva:,}/{nte:,}. Labels: triple-barrier, k={cfg['labels']['k']}, "
          f"sqrt(h)-scaled. PCA: {p.n_components_} components ({p.val_explained_:.1%} validation variance). "
          f"HMM states: {', '.join(ds.regime.names)}.\n",
          "## Experiments A-D (calibrated probabilities)\n", df_to_md(all_t[keep]),
          f"\n## News-shock subset (<= {SHOCK_MINUTES} min after a news event): B vs D\n", df_to_md(bd),
          f"\n## Ablation (horizon {H}, all test bars; shock_* = news-shock subset)\n", df_to_md(abl),
          f"\n## Backtest, horizon {H} (cost {bt_cfg['cost_bps_per_side']} bps per side, entry delay {bt_cfg['entry_delay_bars']} bar)\n",
          df_to_md(pd.concat(bts.values())),
          "\n## Regimes (offline Viterbi on test)\n", df_to_md(stats),
          (f"\nAdjusted Rand index vs synthetic latent regimes: {ari:.3f}\n" if ari is not None else ""),
          "\nLatency: run `python scripts/benchmark.py` (writes reports/latency.csv).\n",
          "\nPlots: calibration_h*.png, confusion_*_h*.png, regimes_viterbi_offline.png\n"]
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    with open(out / "results.json", "w") as f:
        json.dump({str(h): {m: {k: v for k, v in r.items()} for m, r in per.items()} for h, per in res.items()}, f, indent=1)
    print(f"\nwrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
