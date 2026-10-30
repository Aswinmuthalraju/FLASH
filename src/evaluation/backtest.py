"""Simple threshold backtest.

Trade only when the calibrated max probability > threshold AND the top class is UP or DOWN.
Entry at the close of bar t+entry_delay (we cannot trade the close we just computed the signal from),
exit `horizon` bars later, one open position per symbol, round-trip cost = 2 * cost_bps_per_side.
Capital is split equally across the universe, so a day's return = sum(trade net returns) / n_symbols.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(table: pd.DataFrame, rows: np.ndarray, P: np.ndarray, horizon: int, threshold: float,
                 cost_bps: float, entry_delay: int = 1, n_symbols: int | None = None) -> dict:
    sym = table["symbol"].to_numpy()
    day = table["day_idx"].to_numpy()
    close = table["close"].to_numpy()
    n_symbols = n_symbols or table["symbol"].nunique()
    top = P.argmax(1)
    sig = (P.max(1) > threshold) & (top != 1)
    direction = np.where(top == 2, 1, -1)
    free_until = {}
    trades = []
    for k in np.flatnonzero(sig):
        r = rows[k]
        e, x = r + entry_delay, r + entry_delay + horizon
        if x >= len(table) or sym[x] != sym[r] or day[x] != day[r]:
            continue                                     # exit must fall inside the same session
        if r < free_until.get(sym[r], -1):
            continue                                     # already in a position
        free_until[sym[r]] = x + 1
        gross = direction[k] * np.log(close[x] / close[e])
        trades.append((sym[r], day[r], r, direction[k], gross, gross - 2 * cost_bps / 1e4))
    t = pd.DataFrame(trades, columns=["symbol", "day", "row", "dir", "gross", "net"])
    days = np.unique(day[rows])
    daily = (t.groupby("day")["net"].sum().reindex(days).fillna(0.0) / n_symbols) if len(t) else pd.Series(0.0, index=days)
    cum = daily.cumsum()
    mdd = float((cum.cummax() - cum).max()) if len(cum) else 0.0
    sd = daily.std(ddof=1)
    return {"threshold": threshold, "trades": int(len(t)), "trades_per_day": len(t) / max(len(days), 1),
            "hit_rate": float((t["gross"] > 0).mean()) if len(t) else float("nan"),
            "net_hit_rate": float((t["net"] > 0).mean()) if len(t) else float("nan"),
            "avg_gross_bps": float(t["gross"].mean() * 1e4) if len(t) else float("nan"),
            "avg_net_bps": float(t["net"].mean() * 1e4) if len(t) else float("nan"),
            "total_return": float(daily.sum()),
            "sharpe": float(daily.mean() / sd * np.sqrt(252)) if sd and sd > 0 else float("nan"),
            "max_drawdown": mdd,
            "turnover_per_day": float(2 * len(t) / n_symbols / max(len(days), 1))}, t


def sweep(table, rows, P, horizon, thresholds, cost_bps, entry_delay=1) -> pd.DataFrame:
    return pd.DataFrame([run_backtest(table, rows, P, horizon, th, cost_bps, entry_delay)[0] for th in thresholds])
