"""Day-ahead features. One row per (target day, hour), built only from what NEA's Load Dispatch
Centre knows at 10:00 on the issue day: SCADA load up to 09:00 today, observed weather up to
yesterday, a weather forecast for tomorrow, and the calendar.

The leakage rule is enforced by construction (everything is shifted relative to the issue day) and
tested in tests/test_features.py by corrupting post-cutoff data and checking nothing changes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import calendar_np as cal
from .config import LAST_KNOWN_HOUR

WEATHER_FC_SIGMA = {"ktm_tmean": 1.5, "ktm_tmin": 1.8, "ktm_tmax": 1.8, "terai_tmax": 1.6, "terai_dew": 2.0}

FEATURES = [
    "hour", "hours_after_sunset", "dow", "is_saturday", "doy_sin", "doy_cos",
    "days_to_dashami", "days_to_laxmi_puja", "dashain_closure", "tihar",
    "fc_ktm_tmean", "fc_ktm_tmin", "fc_apparent_terai", "obs_ktm_tmean_d1", "fc_minus_obs_ktm",
    "lag_d1_h", "lag_w_h", "d1_mean", "d1_peak", "w_mean", "morning_ratio", "d1_is_saturday",
    "d1_was_dashain", "lag_w_ratio",
]


def weather_forecast(daily_weather: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Stand-in for a day-ahead NWP forecast: the observed value plus autocorrelated error sized to
    typical day-ahead 2 m temperature error. Archived NWP forecasts for Nepal were not reachable for
    this backtest; in production this frame comes from NOAA GFS on AWS Open Data (docs/ARCHITECTURE.md)."""
    rng = np.random.default_rng(seed)
    fc = daily_weather[list(WEATHER_FC_SIGMA)].copy()
    n = len(fc)
    for col, sigma in WEATHER_FC_SIGMA.items():
        e = np.empty(n)
        e[0] = rng.normal(0, sigma)
        for i in range(1, n):
            e[i] = 0.5 * e[i - 1] + rng.normal(0, sigma * np.sqrt(1 - 0.25))
        fc[col] = fc[col] + e
    return fc


def load_matrix(load: pd.Series) -> pd.DataFrame:
    """Hourly series -> one row per day, 24 columns. Missing hours stay NaN."""
    df = load.to_frame("y")
    df["date"] = df.index.normalize()
    df["hour"] = df.index.hour
    m = df.pivot_table(index="date", columns="hour", values="y", aggfunc="mean").reindex(columns=range(24))
    # pivot_table silently drops days with no data at all; a feed outage must stay visible as NaN rows.
    return m.reindex(pd.date_range(df["date"].min(), df["date"].max(), freq="D"))


def build(load: pd.Series, daily_weather: pd.DataFrame, forecast: pd.DataFrame,
          target_days: pd.DatetimeIndex | None = None, with_target: bool = True) -> pd.DataFrame:
    """Rows for every target day T (issue day D = T - 1) and hour 0..23."""
    m = load_matrix(load)
    days = m.index if target_days is None else pd.DatetimeIndex(target_days).normalize()
    one = pd.Timedelta(days=1)
    issue = days - one  # D
    d1 = issue - one  # D-1: the last complete day
    wk = days - pd.Timedelta(days=7)  # same weekday last week (T-7 = D-6)

    def rows(idx):
        return m.reindex(idx).to_numpy()

    m_d1, m_wk, m_issue = rows(d1), rows(wk), rows(issue)
    known = slice(0, LAST_KNOWN_HOUR + 1)  # 00:00 .. 09:00 on the issue day
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN days (feed outages) give NaN, by design
        morning_ratio = np.nanmean(m_issue[:, known], axis=1) / np.nanmean(m_d1[:, known], axis=1)
        d1_mean = np.nanmean(m_d1, axis=1)
        d1_peak = np.nanmax(m_d1, axis=1)
        w_mean = np.nanmean(m_wk, axis=1)
        lag_w_ratio = np.nanmean(m_d1, axis=1) / np.nanmean(rows(d1 - pd.Timedelta(days=7)), axis=1)

    c_t = cal.calendar_frame(days)
    c_d1 = cal.calendar_frame(d1)
    fc = forecast.reindex(days)
    obs = daily_weather.reindex(d1)
    apparent = fc["terai_tmax"].values + 0.35 * np.clip(fc["terai_dew"].values - 18.0, 0, None)
    sunset = cal.sunset_hour(days)

    n = len(days)
    hour = np.tile(np.arange(24), n)
    rep = lambda a: np.repeat(np.asarray(a, dtype=float), 24)  # noqa: E731
    out = pd.DataFrame(
        {
            "target_day": np.repeat(days.values, 24),
            "hour": hour,
            "hours_after_sunset": hour + 0.5 - rep(sunset),
            "dow": rep(c_t["dow"]),
            "is_saturday": rep(c_t["is_saturday"]),
            "doy_sin": rep(np.sin(2 * np.pi * c_t["doy"] / 365.25)),
            "doy_cos": rep(np.cos(2 * np.pi * c_t["doy"] / 365.25)),
            "days_to_dashami": rep(c_t["days_to_dashami"]),
            "days_to_laxmi_puja": rep(c_t["days_to_laxmi_puja"]),
            "dashain_closure": rep(c_t["dashain_closure"]),
            "tihar": rep(c_t["tihar"]),
            "fc_ktm_tmean": rep(fc["ktm_tmean"]),
            "fc_ktm_tmin": rep(fc["ktm_tmin"]),
            "fc_apparent_terai": rep(apparent),
            "obs_ktm_tmean_d1": rep(obs["ktm_tmean"]),
            "fc_minus_obs_ktm": rep(fc["ktm_tmean"].values - obs["ktm_tmean"].values),
            "lag_d1_h": m_d1.reshape(-1),
            "lag_w_h": m_wk.reshape(-1),
            "d1_mean": rep(d1_mean),
            "d1_peak": rep(d1_peak),
            "w_mean": rep(w_mean),
            "morning_ratio": rep(morning_ratio),
            "d1_is_saturday": rep(c_d1["is_saturday"]),
            "d1_was_dashain": rep(c_d1["dashain_closure"]),
            "lag_w_ratio": rep(lag_w_ratio),
        }
    )
    out["anchor"] = out["lag_d1_h"]
    if with_target:
        out["y"] = m.reindex(days).to_numpy().reshape(-1)
    return out


def usable(frame: pd.DataFrame) -> pd.Series:
    """Rows a model can be trained or scored on: anchor and target present."""
    ok = frame["anchor"].notna() & (frame["anchor"] > 0)
    if "y" in frame:
        ok &= frame["y"].notna()
    return ok
