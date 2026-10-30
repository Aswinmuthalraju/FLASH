"""flash-market demo: replay a trading day and watch the prediction change before / after a news event.

    streamlit run app/streamlit_demo.py
Requires `python scripts/train.py` to have been run once (models/bundle.joblib).
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.data.align import split_days  # noqa: E402
from src.data.build import get_data  # noqa: E402
from src.evaluation.experiments import make_encoder  # noqa: E402
from src.inference.predictor import NO_SIGNAL, STAGES, load_bundle, load_memory  # noqa: E402
from src.inference.session import ReplayContext, around_news, run_day  # noqa: E402
from src.news.sentiment import CachedEncoder  # noqa: E402

st.set_page_config(page_title="flash-market", layout="wide")
CLS_COL = {"UP": "#1b9e77", "NEUTRAL": "#888888", "DOWN": "#d95f02"}


@st.cache_resource(show_spinner="Loading models and data ...")
def assets():
    cfg = load_config()
    bars, news = get_data(cfg)
    ctx = ReplayContext(bars, news, cfg)
    enc = CachedEncoder(make_encoder(cfg, finbert=True), resolve(cfg, "processed") / "finbert_cache.npz")
    return cfg, ctx, load_bundle(cfg), load_memory(cfg), enc


@st.cache_resource(show_spinner="Replaying the day through the pipeline ...")
def session(symbol: str, day: str):
    cfg, ctx, bundle, mem, enc = assets()
    return run_day(ctx, bundle, mem, enc, symbol, pd.Timestamp(day))


def signal_of(p_down, p_neutral, p_up, gate):
    P = np.stack([p_down, p_neutral, p_up], 1)
    k = P.argmax(1)
    names = np.array(["DOWN", "NEUTRAL", "UP"])[k]
    return np.where(P.max(1) < gate, NO_SIGNAL, names)


try:
    cfg, ctx, bundle, memory, encoder = assets()
except FileNotFoundError:
    st.error("No trained bundle found. Run `python scripts/train.py` first.")
    st.stop()

st.title("flash-market: news-aware short-horizon prediction (NSE, synthetic replay)")
st.caption("Synthetic prices and headlines -- this demonstrates the pipeline, not trading performance.")

test_days = [d for d in split_days(pd.DatetimeIndex(ctx.days()), cfg)["test"]]
with st.sidebar:
    symbol = st.selectbox("Stock", cfg["symbols"])
    day = st.selectbox("Test day", test_days, format_func=lambda d: f"{d:%Y-%m-%d}")
    H = st.selectbox("Horizon (bars)", bundle["horizons"], index=bundle["horizons"].index(cfg["labels"]["primary_horizon"]))
    gate = st.slider("Confidence gate", 0.34, 0.90, float(cfg["inference"]["gate_threshold"]), 0.01)

sess = session(symbol, str(day))
tl = sess.timeline.copy()
if tl.empty:
    st.warning("No predicted bars for this selection.")
    st.stop()
tl["signal"] = signal_of(tl[f"P_DOWN_{H}"].to_numpy(), tl[f"P_NEUTRAL_{H}"].to_numpy(), tl[f"P_UP_{H}"].to_numpy(), gate)
today_news = sess.news[(sess.news["timestamp"] >= tl["timestamp"].iloc[0]) & (sess.news["timestamp"] <= tl["timestamp"].iloc[-1])].reset_index(drop=True)

with st.sidebar:
    if len(today_news):
        i_ev = st.selectbox("News event", range(len(today_news)),
                            format_func=lambda i: f"{today_news.timestamp[i]:%H:%M} [{today_news.etype[i]}] {today_news.headline[i][:48]}")
    else:
        i_ev = None
        st.info("No news for this stock today.")

# ------------------------------------------------------------------ latency header
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latency / bar, median", f"{tl.latency_us.median():.0f} µs", help="whole hot loop, all 3 horizons, CPU, measured with perf_counter_ns")
c2.metric("p95", f"{tl.latency_us.quantile(.95):.0f} µs")
c3.metric("max", f"{tl.latency_us.max():.0f} µs")
c4.metric("Bars predicted", f"{len(tl)}")

ev_ts = today_news.timestamp[i_ev] if i_ev is not None else None
x = np.arange(len(tl))
k_ev = None
if ev_ts is not None:
    k_ev = int((tl["timestamp"] <= ev_ts).sum())          # first bar strictly after the news

# --------------------------------------------------------------------------- charts
fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]})
ax[0].plot(x, tl["close"], color="k", lw=1)
ax[0].set_ylabel("close")
ax[1].plot(x, tl[f"P_UP_{H}"], color=CLS_COL["UP"], label="P(UP)")
ax[1].plot(x, tl[f"P_DOWN_{H}"], color=CLS_COL["DOWN"], label="P(DOWN)")
ax[1].plot(x, tl[f"P_NEUTRAL_{H}"], color=CLS_COL["NEUTRAL"], lw=.8, label="P(NEUTRAL)")
ax[1].axhline(gate, color="b", ls=":", lw=1, label=f"gate {gate:.2f}")
ax[1].set_ylabel(f"calibrated, h={H}"); ax[1].legend(ncol=4, fontsize=8, loc="upper left"); ax[1].set_ylim(0, 1)
reg_cols = [c for c in tl.columns if c.startswith("regime_")]
ax[2].stackplot(x, *[tl[c] for c in reg_cols], labels=[c[7:] for c in reg_cols], alpha=.8)
ax[2].set_ylabel("filtered regime"); ax[2].legend(ncol=len(reg_cols), fontsize=8, loc="upper left"); ax[2].set_xlabel("minute of session (after warm-up)")
ticks = np.arange(0, len(tl), 30)
ax[2].set_xticks(ticks); ax[2].set_xticklabels([tl["timestamp"].iloc[i].strftime("%H:%M") for i in ticks], rotation=0, fontsize=8)
for _, n in today_news.iterrows():
    k = int((tl["timestamp"] <= n.timestamp).sum())
    for a in ax:
        a.axvline(k, color="purple", alpha=.25 if (ev_ts is not None and n.timestamp != ev_ts) else .9, lw=1.2)
if k_ev is not None:
    ax[0].set_title(f"{symbol} {day:%Y-%m-%d}   purple lines = news; bold = selected", fontsize=10)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ------------------------------------------------------------------ before / after
if ev_ts is not None:
    n = today_news.iloc[i_ev]
    st.subheader("Before / after the selected news")
    st.markdown(f"**{n.timestamp:%H:%M:%S}**  ·  `{n.etype}`  ·  FinBERT sentiment **{n.sentiment:+.2f}**  ·  relevance {n.relevance:.1f}  ·  "
                f"similar past events moved **{n.hist_mean:+.2f}σ** (cosine {n.hist_sim:.2f})  \n> {n.headline}")
    snap = around_news(sess, ev_ts, H)
    snap["signal"] = signal_of(snap.P_DOWN.to_numpy(), snap.P_NEUTRAL.to_numpy(), snap.P_UP.to_numpy(), gate)
    st.dataframe(snap.drop(columns="_k").style.format({"P_UP": "{:.3f}", "P_NEUTRAL": "{:.3f}", "P_DOWN": "{:.3f}", "latency_us": "{:.0f}"}),
                 hide_index=True, use_container_width=True)

    st.subheader("Why? (LightGBM pred_contrib, grouped)")
    offs = snap["offset_min"].tolist()
    default = offs.index(1) if 1 in offs else 0
    off = st.select_slider("Bar relative to the news (minutes)", options=offs, value=offs[default])
    k = int(snap.loc[snap.offset_min == off, "_k"].iloc[0])
    ex = sess.pred.explain(H, x=sess.X[k], top=8)
    g1, g2 = st.columns([1, 2])
    with g1:
        fig2, a2 = plt.subplots(figsize=(4, 3))
        names, vals = list(ex["group_contribution"]), list(ex["group_contribution"].values())
        a2.barh(names, vals, color=["#1b9e77" if v >= 0 else "#d95f02" for v in vals]); a2.axvline(0, color="k", lw=.8)
        a2.set_title(f"contribution to {ex['class']}", fontsize=10); fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)
    with g2:
        st.dataframe(pd.DataFrame(ex["top_features"]).style.format({"value": "{:.4g}", "contribution": "{:+.3f}"}),
                     hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- latency
st.subheader("Where the microseconds go (median per stage, µs)")
stage_cols = [f"us_{s}" for s in STAGES]
st.bar_chart(pd.DataFrame({"µs": [tl[c].median() for c in stage_cols]}, index=STAGES))
st.caption("Full benchmark (10k+ bars, p95/p99/max, A vs D): `python scripts/benchmark.py`.")
