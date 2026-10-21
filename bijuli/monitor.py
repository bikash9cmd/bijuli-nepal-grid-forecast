"""Monitoring: score yesterday's forecast against what happened, and decide whether the world moved.

Three different questions, three different detectors:
  data drift     do today's inputs look like the training data?     PSI per feature, weekly
  concept drift  is the model wrong in a consistent direction?        two-sided CUSUM on daily bias
  performance    is it simply worse than it promised?                 7-day MAPE vs offline MAPE

Bias is the one that matters most for a growing grid: a model that under-forecasts by 2% every day
has a fine-looking MAPE for weeks while NEA under-buys every evening.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CUSUM_H, CUSUM_K, DRIFT_PSI_ALERT

PSI_FEATURES = ["d1_mean", "fc_ktm_tmean", "fc_apparent_terai", "morning_ratio", "lag_w_ratio"]
PSI_BINS = 10


def reference_bins(frame: pd.DataFrame, features=PSI_FEATURES) -> dict:
    """Decile edges and frequencies of the training data, stored on the model card."""
    ref = {}
    for f in features:
        x = frame[f].dropna().to_numpy()
        edges = np.unique(np.quantile(x, np.linspace(0, 1, PSI_BINS + 1)))
        freq = np.histogram(np.clip(x, edges[0], edges[-1]), bins=edges)[0] / len(x)
        ref[f] = {"edges": edges.tolist(), "freq": freq.tolist()}
    return ref


def psi(reference: dict, frame: pd.DataFrame) -> dict[str, float]:
    """Population stability index per feature. Values outside the training range land in the edge
    bins, which is exactly where a growing demand level shows up first."""
    out = {}
    for f, r in reference.items():
        x = frame[f].dropna().to_numpy()
        if len(x) == 0:
            continue
        edges = np.asarray(r["edges"])
        cur = np.histogram(np.clip(x, edges[0], edges[-1]), bins=edges)[0] / len(x)
        ref = np.asarray(r["freq"])
        cur, ref = np.clip(cur, 1e-4, None), np.clip(ref, 1e-4, None)
        out[f] = float(np.sum((cur - ref) * np.log(cur / ref)))
    return out


@dataclass
class Cusum:
    """Two-sided CUSUM on the daily mean percentage error. Resets when a new champion is promoted."""

    k: float = CUSUM_K
    h: float = CUSUM_H
    pos: float = 0.0
    neg: float = 0.0
    history: list = field(default_factory=list)

    def update(self, daily_bias: float) -> bool:
        if not np.isfinite(daily_bias):
            return False
        self.pos = max(0.0, self.pos + daily_bias - self.k)
        self.neg = max(0.0, self.neg - daily_bias - self.k)
        self.history.append((self.pos, self.neg))
        return self.alarm

    @property
    def alarm(self) -> bool:
        return self.pos > self.h or self.neg > self.h

    def reset(self) -> None:
        self.pos = self.neg = 0.0


def score_day(day_frame: pd.DataFrame, col: str = "p50") -> dict:
    """Daily scorecard for one forecast (24 rows with y and predictions)."""
    f = day_frame.dropna(subset=["y", col])
    if f.empty:
        return {"mape": np.nan, "bias": np.nan, "peak_err": np.nan}
    e = (f[col] - f["y"]) / f["y"]
    return {
        "mape": float(e.abs().mean()),
        "bias": float(e.mean()),
        "peak_err": float((f[col].max() - f["y"].max()) / f["y"].max()),
    }


def drift_report(reference: dict, recent: pd.DataFrame) -> dict:
    scores = psi(reference, recent)
    worst = max(scores, key=scores.get) if scores else None
    return {"psi": scores, "worst": worst, "alert": bool(worst and scores[worst] > DRIFT_PSI_ALERT)}
