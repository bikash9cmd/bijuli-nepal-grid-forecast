"""Data contracts: checks every feed must pass before it can reach a model, for training or for a forecast.

A failed check never crashes the pipeline. It quarantines the bad hours (set to NaN, so LightGBM
treats them as missing and the training set drops them) and returns findings for the alarm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_MW, MAX_MW = 300.0, 4500.0  # Nepal's national demand has never left this band; peak so far ~2,400 MW
MAX_HOURLY_JUMP = 0.35  # a 35% change in one hour is a telemetry fault, not consumers
FLATLINE_HOURS = 4  # SCADA historians repeat the last value when an RTU stops reporting


@dataclass
class Finding:
    check: str
    hours: int
    detail: str


def check_load(load: pd.Series, reference_median: float | None = None) -> tuple[pd.Series, list[Finding]]:
    """Validate an hourly MW series. Returns (cleaned series, findings)."""
    s = load.astype(float).copy()
    findings: list[Finding] = []

    def quarantine(mask: pd.Series, check: str, detail: str):
        n = int(mask.sum())
        if n:
            s[mask] = np.nan
            findings.append(Finding(check, n, detail))

    # Unit mix-ups first: a kW day sits ~1000x above the rest, a GW day 1000x below.
    if reference_median is None:
        reference_median = float(np.nanmedian(s))
    daily_median = s.groupby(s.index.normalize()).transform("median")
    ratio = daily_median / reference_median
    quarantine((ratio > 50) | (ratio < 0.02), "unit", "whole days off by ~1000x: kW or GW written as MW")

    quarantine(~s.between(MIN_MW, MAX_MW) & s.notna(), "range", f"outside {MIN_MW:.0f}-{MAX_MW:.0f} MW")

    run_id = (s != s.shift()).cumsum()
    flat = (s.groupby(run_id).transform("size") >= FLATLINE_HOURS) & s.notna()
    quarantine(flat, "flatline", f">= {FLATLINE_HOURS} identical consecutive readings (stuck RTU)")

    rel = (s / s.shift(1) - 1).abs()
    spike = (rel > MAX_HOURLY_JUMP) & ((s / s.shift(-1) - 1).abs() > MAX_HOURLY_JUMP)
    quarantine(spike, "spike", f"isolated jump > {MAX_HOURLY_JUMP:.0%} that reverses the next hour")
    return s, findings


def check_weather(daily: pd.DataFrame) -> list[Finding]:
    """Weather is never quarantined (the forecast falls back to imputed values); the alarm says
    when the model is running on estimates instead of observations."""
    findings = []
    imputed = int(daily.get("terai_imputed", pd.Series(dtype=bool)).sum())
    if imputed:
        findings.append(Finding("terai_missing", imputed * 24, "no Terai station reported; afternoon temperature estimated from Kathmandu"))
    if "ktm_tmean" not in daily:
        findings.append(Finding("kathmandu_missing", len(daily) * 24, "Kathmandu daily mean column absent from the feed"))
        return findings
    missing = int(daily["ktm_tmean"].isna().sum())
    if missing:
        findings.append(Finding("kathmandu_missing", missing * 24, "Kathmandu daily mean missing"))
    return findings


def freshness_ok(load: pd.Series, issue_time: pd.Timestamp, max_lag_hours: int = 3) -> bool:
    """At issue time (10:00) the feed must reach at least 06:00 today, or the morning signal is stale.

    A last reading after the issue time is not fresh either: it means a clock or time zone fault
    (UTC written as Nepal time puts readings 5h45m in the future), so the feed cannot be trusted.
    """
    last = load.dropna().index.max()
    if last is pd.NaT:
        return False
    lag = issue_time - last
    return pd.Timedelta(0) <= lag <= pd.Timedelta(hours=max_lag_hours + 1)
