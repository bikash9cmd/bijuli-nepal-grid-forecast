"""Scoring: what the Load Dispatch Centre cares about, in MW, percent and rupees."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ABSORB_MW, COST_OVER_NPR_PER_MWH, COST_UNDER_NPR_PER_MWH


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0)
    return float(np.mean(np.abs(p[ok] - y[ok]) / y[ok]))


def bias(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0)
    return float(np.mean((p[ok] - y[ok]) / y[ok]))


def pinball(y, p, q: float) -> float:
    d = np.asarray(y, float) - np.asarray(p, float)
    return float(np.nanmean(np.maximum(q * d, (q - 1) * d)))


def error_cost_npr(y, bid) -> float:
    """Rupees lost to forecast error. NEA can absorb small misses by re-dispatching its peaking
    storage plants (Kulekhani I and II), so only the part of each hourly miss beyond
    ABSORB_MW is settled: shortfalls bought in the real-time market or cut, surplus bought day-ahead
    and resold at a loss. One forecast hour = MW * 1 h = MWh."""
    d = np.asarray(bid, float) - np.asarray(y, float)
    d = d[np.isfinite(d)]
    settled = np.clip(np.abs(d) - ABSORB_MW, 0, None)
    return float(np.sum(np.where(d < 0, settled * COST_UNDER_NPR_PER_MWH, settled * COST_OVER_NPR_PER_MWH)))


def daily_peak_mape(frame: pd.DataFrame, col: str) -> float:
    g = frame.groupby("target_day")
    return mape(g["y"].max(), g[col].max())


def summarise(frame: pd.DataFrame, col: str = "p50", bid_col: str | None = None) -> dict:
    """frame has y, target_day and the prediction columns."""
    ok = frame["y"].notna() & frame[col].notna()
    f = frame[ok]
    days = f["target_day"].nunique()
    out = {
        "hours": int(len(f)),
        "mape": mape(f["y"], f[col]),
        "bias": bias(f["y"], f[col]),
        "peak_mape": daily_peak_mape(f, col),
        "cost_npr_per_day": error_cost_npr(f["y"], f[bid_col or col]) / max(days, 1),
    }
    if col == "p50" and {"p10", "p90"} <= set(f.columns):
        out["coverage_80"] = float(((f["y"] >= f["p10"]) & (f["y"] <= f["p90"])).mean())
    return out
