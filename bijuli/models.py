"""Baselines and the LightGBM quantile forecaster.

The model predicts log(demand / anchor), where the anchor is yesterday's load at the same hour.
Demand grows every month, and a tree cannot predict a level it has never seen, so it learns
"how different from yesterday" instead. That one decision is what keeps an old model from
collapsing as the grid grows (docs/CHALLENGES_AND_BUGS.md).
"""

from __future__ import annotations

import hashlib
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import BID_QUANTILE, QUANTILES
from .features import FEATURES

CALIB_DAYS = 42

LGB_PARAMS = dict(
    n_estimators=250, learning_rate=0.08, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
)


def baseline_yesterday(f: pd.DataFrame) -> np.ndarray:
    """Same hour on the last complete day (D-1)."""
    return f["lag_d1_h"].to_numpy()


def baseline_last_week(f: pd.DataFrame) -> np.ndarray:
    """Same hour, same weekday last week."""
    return f["lag_w_h"].to_numpy()


def baseline_scaled_week(f: pd.DataFrame) -> np.ndarray:
    """Last week's profile scaled by how this morning compares with yesterday morning and how
    yesterday compared with the same day a week before: a reasonable desk-forecaster's rule."""
    scale = (f["morning_ratio"] * f["lag_w_ratio"]).clip(0.8, 1.25).fillna(1.0)
    return (f["lag_w_h"] * scale).to_numpy()


BASELINES = {
    "Same hour yesterday": baseline_yesterday,
    "Same hour last week": baseline_last_week,
    "Last week, rescaled to this morning": baseline_scaled_week,
}


class QuantileForecaster:
    """Three LightGBM quantile models on the log-ratio target, plus the cost-optimal bid."""

    def __init__(self, params: dict | None = None, quantiles=QUANTILES, features=FEATURES):
        self.params = {**LGB_PARAMS, **(params or {})}
        self.quantiles = tuple(quantiles)
        self.features = list(features)
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.meta: dict = {}
        self.margin = 0.0

    def _fit_models(self, f: pd.DataFrame, seed: int) -> dict:
        x = f[self.features]
        y = np.log(f["y"] / f["anchor"])
        models = {}
        for q in self.quantiles:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=seed, **self.params)
            m.fit(x, y)
            models[q] = m
        return models

    def fit(self, f: pd.DataFrame, seed: int = 0, calib_days: int = CALIB_DAYS) -> QuantileForecaster:
        """Conformalised quantile regression: fit on all but the last `calib_days`, measure how far
        outcomes fall outside the raw p10-p90 band there, widen by that margin, then refit on
        everything. Raw LightGBM quantile bands covered 37% of outcomes instead of 80%."""
        self.margin = 0.0
        self.holdout_mape = float("nan")
        days = np.sort(f["target_day"].unique())
        if calib_days and len(days) > calib_days + 60:
            cut = days[-calib_days]
            fit_part, cal_part = f[f["target_day"] < cut], f[f["target_day"] >= cut]
            self.models = self._fit_models(fit_part, seed)
            raw = self._raw_log(cal_part)
            y = np.log(cal_part["y"] / cal_part["anchor"]).to_numpy()
            score = np.maximum(raw[self.quantiles[0]] - y, y - raw[self.quantiles[-1]])
            coverage = self.quantiles[-1] - self.quantiles[0]  # 0.8 for p10-p90
            self.margin = float(np.quantile(score, min(1.0, coverage * (1 + 1 / len(score)))))
            # Out-of-sample error on the held-out weeks: what this model "promises" in production.
            self.holdout_mape = float(np.mean(np.abs(np.exp(raw[0.5] - y) - 1)))
        self.models = self._fit_models(f, seed)
        self.meta = {
            "train_rows": int(len(f)),
            "train_start": str(pd.Timestamp(f["target_day"].min()).date()),
            "train_end": str(pd.Timestamp(f["target_day"].max()).date()),
            "data_sha256": frame_hash(f),
            "features": self.features,
            "params": self.params,
            "conformal_margin": round(self.margin, 5),
            "holdout_mape": round(self.holdout_mape, 5),
        }
        return self

    def _raw_log(self, f: pd.DataFrame) -> dict:
        x = f[self.features]
        return {q: self.models[q].predict(x) for q in self.quantiles}

    def predict(self, f: pd.DataFrame) -> pd.DataFrame:
        anchor = f["anchor"].to_numpy()
        raw = self._raw_log(f)
        raw[self.quantiles[0]] = raw[self.quantiles[0]] - self.margin
        raw[self.quantiles[-1]] = raw[self.quantiles[-1]] + self.margin
        preds = {q: anchor * np.exp(raw[q]) for q in self.quantiles}
        lo, mid, hi = (preds[q] for q in self.quantiles)
        # Quantile models are trained separately and can cross; sort so p10 <= p50 <= p90.
        stacked = np.sort(np.vstack([lo, mid, hi]), axis=0)
        out = pd.DataFrame({"p10": stacked[0], "p50": stacked[1], "p90": stacked[2]}, index=f.index)
        out["bid"] = bid_from_quantiles(out)
        return out


def bid_from_quantiles(q: pd.DataFrame, target: float = BID_QUANTILE) -> np.ndarray:
    """The MW to buy day-ahead. With under-forecasting 1.6x as expensive as over-forecasting,
    the cost-minimising bid is the 0.62 quantile, not the median. Linear between p50 and p90."""
    w = (target - 0.5) / (0.9 - 0.5)
    return (q["p50"] + w * (q["p90"] - q["p50"])).to_numpy()


def frame_hash(f: pd.DataFrame) -> str:
    """Content hash of the training data: lineage for the registry, so 'which data made this model?'
    always has an answer."""
    cols = [c for c in ("target_day", "hour", "y", "anchor") if c in f]
    h = hashlib.sha256(pd.util.hash_pandas_object(f[cols], index=False).values.tobytes())
    return h.hexdigest()[:16]


def feature_importance(model: QuantileForecaster, top: int = 12) -> pd.Series:
    m = model.models[0.5]
    imp = pd.Series(m.booster_.feature_importance("gain"), index=model.features)
    return (imp / imp.sum()).sort_values(ascending=False).head(top)


def to_json_meta(model: QuantileForecaster) -> str:
    return json.dumps(model.meta, sort_keys=True, default=str)
