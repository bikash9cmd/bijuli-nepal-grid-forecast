"""Grid twin: hourly national electricity demand for Nepal, driven by real weather and calendar and
calibrated to NEA's published FY 2081/82 monthly energy and peak demand.

Why a twin: NEA does not publish an open, machine-readable hourly load series that this project could
reach (see data/README.md). The twin is built from documented components so every number it produces
can be explained, and it is checked against 40 real NEA daily operational reports it was never fitted to.
It is labelled as simulated everywhere it appears. Swapping in real SCADA data is a loader change:
anything that returns the same `load` frame (time, demand_mw) works with the rest of the pipeline.

Components (all MW, national demand requirement = demand served + interruption):
  organic   base level growing 8% a year, times a smooth seasonal curve fitted to NEA's monthly energy
  weather   heating load from Kathmandu cold, cooling load from Terai heat and humidity (real NOAA data)
  shape     morning cooking peak, industrial daytime plateau, an evening peak that follows sunset
  calendar  Saturdays, the Dashain shutdown, Tihar lights
  drift     EV charging that grows every month, and two new industrial feeders (dated, so detection
            lag can be measured)
  noise     day-level and hour-level autocorrelated noise
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import calendar_np as cal
from .config import NEA_DIR

FY_START, FY_END = pd.Timestamp("2024-07-16"), pd.Timestamp("2025-07-16")  # FY 2081/82, 1 Shrawan to 1 Shrawan
FISCAL_ORDER = [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3]  # Shrawan .. Ashadh as BS month numbers


@dataclass
class TwinParams:
    growth: float = 0.08  # organic growth per year
    heat_base_c: float = 14.0  # Kathmandu daily mean below which heaters come on
    heat_mw_per_c: float = 26.0
    cool_base_c: float = 27.0  # Terai apparent afternoon temperature above which fans and ACs ramp
    cool_mw_per_c: float = 34.0
    saturday_factor: float = 0.955
    dashain_factor: float = 0.80
    tihar_evening_boost: float = 0.07
    ev_mw_start: float = 6.0  # average EV charging load, Jan 2022
    ev_mw_end: float = 48.0  # Aug 2025
    feeders: list[tuple[str, float]] = field(
        default_factory=lambda: [("2024-11-20", 60.0), ("2025-04-10", 45.0)]
    )
    daily_noise: float = 0.012
    hourly_noise: float = 0.010
    seed: int = 2081


def nea_monthly() -> pd.DataFrame:
    return pd.read_csv(NEA_DIR / "monthly_fy2081_82.csv")


def nea_daily_reports() -> pd.DataFrame:
    df = pd.read_csv(NEA_DIR / "daily_reports_2024-12_2025-01.csv", parse_dates=["date"])
    return df.set_index("date")


def _fill_weather(w: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    w = w.reindex(dates)
    clim = w.groupby(w.index.dayofyear).transform("mean")
    for c in ("ktm_tmean", "ktm_tmin", "ktm_tmax", "terai_tmax", "terai_dew"):
        w[c] = w[c].fillna(clim[c]).interpolate(limit_direction="both")
    return w


def _shape(hours: np.ndarray, sunset: np.ndarray, evening_amp: np.ndarray) -> np.ndarray:
    """Relative hourly profile (mean about 1) for each row: night trough, morning cooking peak at 08:00,
    a daytime industrial plateau and a sharp evening peak ~40 minutes after sunset."""
    h = hours.astype(float) + 0.5
    night = -0.30 * np.exp(-0.5 * ((h - 3.5) / 2.2) ** 2)
    morning = 0.16 * np.exp(-0.5 * ((h - 8.2) / 1.3) ** 2)
    day = 0.05 * np.exp(-0.5 * ((h - 13.0) / 3.0) ** 2)
    peak_at = sunset + 0.7
    evening = evening_amp * np.exp(-0.5 * ((h - peak_at) / 1.25) ** 2)
    late = 0.10 * np.exp(-0.5 * ((h - (peak_at + 2.5)) / 1.6) ** 2)
    return 1.0 + night + morning + day + evening + late


def _evening_amp(doy: np.ndarray) -> np.ndarray:
    """Evening peak height: largest in winter (lights + cooking + heating at once), smallest in the
    monsoon, when cooling keeps the whole day high and the peak flattens (NEA: peak/mean 1.42 in Poush,
    1.16 in Ashadh)."""
    return 0.22 + 0.12 * np.cos(2 * np.pi * (doy - 5) / 365.25)


def _ev_profile(hours: np.ndarray) -> np.ndarray:
    """Home charging after work and overnight, a smaller midday public-charger hump. Mean = 1."""
    h = hours.astype(float) + 0.5
    p = 0.35 + 1.6 * np.exp(-0.5 * ((h - 21.0) / 2.0) ** 2) + 1.2 * np.exp(-0.5 * ((h - 0.5) / 2.0) ** 2) \
        + 0.4 * np.exp(-0.5 * ((h - 13.0) / 2.0) ** 2)
    return p


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _components(weather: pd.DataFrame, start: str, end: str, p: TwinParams, season: np.ndarray | None):
    dates = pd.date_range(start, end, freq="D")
    w = _fill_weather(weather, dates)
    c = cal.calendar_frame(dates)
    years = ((dates - FY_START).days / 365.25).values
    organic_level = np.exp(np.log1p(p.growth) * years)  # 1.0 at the start of FY 2081/82
    doy = dates.dayofyear.values
    if season is None:
        season = np.ones(len(dates))

    apparent = w["terai_tmax"].values + 0.35 * np.clip(w["terai_dew"].values - 18.0, 0, None)
    heat = p.heat_mw_per_c * np.clip(p.heat_base_c - w["ktm_tmean"].values, 0, None) * organic_level
    cool = p.cool_mw_per_c * np.clip(apparent - p.cool_base_c, 0, None) * organic_level

    t = np.linspace(0, 1, len(dates))
    ev = p.ev_mw_start * (p.ev_mw_end / p.ev_mw_start) ** t
    feeder = np.zeros(len(dates))
    for when, mw in p.feeders:
        feeder += np.where(dates >= pd.Timestamp(when), mw, 0.0)

    day = pd.DataFrame(
        {
            "organic": organic_level * season,
            "heat": heat,
            "cool": cool,
            "ev": ev,
            "feeder": feeder,
            "sunset": cal.sunset_hour(dates),
            "evening_amp": _evening_amp(doy),
            "is_saturday": c["is_saturday"].values,
            "dashain": c["dashain_closure"].values,
            "pre_dashain": c["days_to_dashami"].between(-8, -4).values.astype(int),
            "tihar": c["tihar"].values,
        },
        index=dates,
    )
    return day


def _hourly(day: pd.DataFrame, base_mw: float, p: TwinParams) -> pd.DataFrame:
    rng = np.random.default_rng(p.seed)
    idx = pd.date_range(day.index[0], day.index[-1] + pd.Timedelta(hours=23), freq="h")
    d = day.reindex(idx.normalize()).set_axis(idx)
    hours = idx.hour.values
    shape = _shape(hours, d["sunset"].values, d["evening_amp"].values)
    h = hours + 0.5
    heat_shape = 1.0 + 0.9 * np.exp(-0.5 * ((h - 7.5) / 1.8) ** 2) + 1.1 * np.exp(-0.5 * ((h - 19.5) / 2.0) ** 2) \
        - 0.6 * np.exp(-0.5 * ((h - 14.0) / 2.5) ** 2)
    heat_shape /= heat_shape.reshape(-1, 24).mean(axis=1).repeat(24)
    cool_shape = 1.0 + 0.6 * np.exp(-0.5 * ((h - 15.0) / 3.0) ** 2) + 0.3 * np.exp(-0.5 * ((h - 22.0) / 2.5) ** 2) \
        - 0.4 * np.exp(-0.5 * ((h - 6.0) / 2.0) ** 2)
    cool_shape /= cool_shape.reshape(-1, 24).mean(axis=1).repeat(24)
    ev_shape = _ev_profile(hours)
    ev_shape = ev_shape / ev_shape.reshape(-1, 24).mean(axis=1).repeat(24)

    organic = base_mw * d["organic"].values * shape
    # Saturdays and Dashain hit the industrial daytime load hardest
    daytime = ((h >= 9) & (h <= 18)).astype(float)
    sat = np.where(d["is_saturday"].values == 1, p.saturday_factor - 0.03 * daytime, 1.0)
    dash = np.where(d["dashain"].values == 1, p.dashain_factor - 0.05 * daytime, 1.0)
    pre = np.where(d["pre_dashain"].values == 1, 0.97, 1.0)
    tihar = np.where(d["tihar"].values == 1, 1.0 + p.tihar_evening_boost * np.exp(-0.5 * ((h - 19.5) / 1.5) ** 2), 1.0)
    organic = organic * sat * dash * pre * tihar

    feeder = d["feeder"].values * np.where(d["is_saturday"].values == 1, 0.5, 1.0) * np.where(d["dashain"].values == 1, 0.2, 1.0)
    demand = organic + d["heat"].values * heat_shape + d["cool"].values * cool_shape + d["ev"].values * ev_shape + feeder

    n_days = len(day)
    daily_noise = np.repeat(_ar1(n_days, 0.7, p.daily_noise, rng), 24)
    hourly_noise = _ar1(len(idx), 0.85, p.hourly_noise, rng)
    demand = demand * (1 + daily_noise) * (1 + hourly_noise)

    return pd.DataFrame(
        {
            "demand_mw": demand.round(1),
            "ev_mw": (d["ev"].values * ev_shape).round(1),
            "feeder_mw": feeder.round(1),
        },
        index=pd.DatetimeIndex(idx, name="time"),
    )


def _fy_month_key(index: pd.DatetimeIndex) -> np.ndarray:
    return np.array([cal.to_bs(t.date())[1] for t in index.normalize()])


def _fourier(doy: np.ndarray, k: int = 3) -> np.ndarray:
    cols = [np.ones_like(doy, dtype=float)]
    for i in range(1, k + 1):
        cols += [np.cos(2 * np.pi * i * doy / 365.25), np.sin(2 * np.pi * i * doy / 365.25)]
    return np.column_stack(cols)


def calibrate(weather: pd.DataFrame, p: TwinParams | None = None, iterations: int = 4):
    """Fit the base level and the smooth seasonal curve so FY 2081/82 monthly energy matches NEA."""
    p = p or TwinParams()
    nea = nea_monthly()
    target = dict(zip(FISCAL_ORDER, nea["national_energy_gwh"], strict=True))
    dates = pd.date_range(FY_START, FY_END - pd.Timedelta(days=1), freq="D")
    season_coef = np.zeros(7)
    season_coef[0] = 1.0
    base_mw = 1200.0
    for _ in range(iterations):
        season = _fourier(dates.dayofyear.values) @ season_coef
        day = _components(weather, str(dates[0].date()), str(dates[-1].date()), p, season)
        hourly = _hourly(day, base_mw, p)
        months = _fy_month_key(hourly.index)
        sim = hourly.groupby(months)["demand_mw"].sum() / 1000.0  # GWh
        # Solve for the organic share only: weather, EVs and feeders are physical and stay fixed.
        organic_hourly = (base_mw * day["organic"]).reindex(hourly.index.normalize()).values
        organic_gwh = pd.Series(organic_hourly).groupby(months).sum() / 1000.0
        ratio = {m: 1 + (target[m] - sim[m]) / organic_gwh[m] for m in FISCAL_ORDER}
        mid = {m: dates[np.array([cal.to_bs(x.date())[1] for x in dates]) == m] for m in FISCAL_ORDER}
        x = np.concatenate([mid[m].dayofyear.values for m in FISCAL_ORDER])
        cur = np.concatenate([_fourier(mid[m].dayofyear.values) @ season_coef for m in FISCAL_ORDER])
        y = np.concatenate([np.full(len(mid[m]), ratio[m]) for m in FISCAL_ORDER]) * cur
        season_coef, *_ = np.linalg.lstsq(_fourier(x), y, rcond=None)
    return base_mw, season_coef, p


def generate(weather: pd.DataFrame, start: str, end: str, p: TwinParams | None = None):
    """Hourly demand from `start` to `end` (inclusive dates), calibrated on FY 2081/82."""
    base_mw, coef, p = calibrate(weather, p)
    dates = pd.date_range(start, end, freq="D")
    day = _components(weather, start, end, p, _fourier(dates.dayofyear.values) @ coef)
    return _hourly(day, base_mw, p)


def calibration_report(load: pd.DataFrame) -> pd.DataFrame:
    """Twin vs NEA FY 2081/82, month by month: energy (fitted) and peak (not fitted)."""
    nea = nea_monthly()
    fy = load.loc[FY_START: FY_END - pd.Timedelta(hours=1)]
    months = _fy_month_key(fy.index)
    g = fy.groupby(months)["demand_mw"]
    rows = []
    for m, (_, r) in zip(FISCAL_ORDER, nea.iterrows(), strict=True):
        rows.append(
            {
                "bs_month": r["bs_month"],
                "nea_energy_gwh": r["national_energy_gwh"],
                "twin_energy_gwh": round(g.sum()[m] / 1000.0),
                "nea_peak_mw": r["national_peak_mw"],
                "twin_peak_mw": round(g.max()[m]),
            }
        )
    return pd.DataFrame(rows)


def validate_against_daily_reports(load: pd.DataFrame) -> dict:
    """The honest check: 40 real NEA daily operational reports (Dec 2024 - Jan 2025) the twin never saw."""
    real = nea_daily_reports()
    daily = load["demand_mw"].groupby(load.index.normalize())
    twin = pd.DataFrame({"energy_mwh": daily.sum(), "peak_mw": daily.max(), "peak_hour": daily.idxmax().dt.hour})
    j = real.join(twin, how="inner")
    e_err = (j["energy_mwh"] - j["energy_requirement_mwh"]) / j["energy_requirement_mwh"]
    p_err = (j["peak_mw"] - j["peak_requirement_mw"]) / j["peak_requirement_mw"]
    real_hour = pd.to_datetime(j["peak_time"], format="%H:%M")
    return {
        "days": int(len(j)),
        "energy_mape": float(e_err.abs().mean()),
        "energy_bias": float(e_err.mean()),
        "energy_corr": float(j["energy_mwh"].corr(j["energy_requirement_mwh"])),
        "peak_mape": float(p_err.abs().mean()),
        "peak_bias": float(p_err.mean()),
        "peak_corr": float(j["peak_mw"].corr(j["peak_requirement_mw"])),
        "real_peak_time_range": f"{real_hour.min():%H:%M}-{real_hour.max():%H:%M}",
        "twin_peak_hour_range": f"{int(j['peak_hour'].min()):02d}:00-{int(j['peak_hour'].max()):02d}:59",
        "table": j[["energy_requirement_mwh", "energy_mwh", "peak_requirement_mw", "peak_mw"]].round(0),
    }
