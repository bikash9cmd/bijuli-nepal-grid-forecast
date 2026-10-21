"""Real weather for Nepal's load centres: download (pinned), parse, convert to Nepal time, fill gaps honestly.

Kathmandu reports every hour. The Terai stations, where the summer cooling load lives, only report
3-hourly daytime synops (05:45 to 17:45 NPT) and have a network-wide hole from December 2023 to
March 2024. Every rule here exists because of something in these files: see docs/CHALLENGES_AND_BUGS.md.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ISD_BUCKET, ISD_SHA256, RAW, STATIONS, TZ, YEARS

ISD_COLUMNS = ["year", "month", "day", "hour", "temp", "dewpoint", "slp", "wind_dir", "wind_speed", "sky", "p1", "p6"]
MISSING = -9999
AFTERNOON_UTC_HOUR = 9  # 14:45 NPT: the synop closest to the daily maximum
MAX_INTERP_HOURS = 3


def isd_url(usaf: str, year: int) -> str:
    return f"https://{ISD_BUCKET}.s3.amazonaws.com/isd-lite/data/{year}/{usaf}-99999-{year}.gz"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(raw_dir: Path = RAW, force: bool = False) -> list[Path]:
    """Fetch every pinned file once. A checksum mismatch is an error, never a silent overwrite."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for usaf in STATIONS:
        for year in YEARS:
            key = f"{usaf}-{year}"
            path = raw_dir / f"{key}.gz"
            if force or not path.exists():
                with urllib.request.urlopen(isd_url(usaf, year), timeout=60) as resp:  # noqa: S310 (fixed https host)
                    path.write_bytes(resp.read())
            got = sha256(path)
            if got != ISD_SHA256[key]:
                raise ValueError(f"{path.name}: sha256 {got[:12]} does not match the pinned {ISD_SHA256[key][:12]}")
            paths.append(path)
    return paths


def parse_isd_lite(raw: bytes) -> pd.DataFrame:
    """One ISD-Lite file -> rows indexed by *Nepal* time. Values are tenths of a degree; -9999 is missing."""
    text = gzip.decompress(raw).decode("ascii")
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None, names=ISD_COLUMNS)
    utc = pd.to_datetime(df[["year", "month", "day", "hour"]], utc=True)
    out = pd.DataFrame(
        {
            "time_utc": utc,
            "temp_c": df["temp"].where(df["temp"] != MISSING) / 10.0,
            "dewpoint_c": df["dewpoint"].where(df["dewpoint"] != MISSING) / 10.0,
        }
    )
    out["time"] = out["time_utc"].dt.tz_convert(TZ).dt.tz_localize(None)
    # Plausibility for Nepal's inhabited stations; anything else is a decoding or sensor fault.
    out.loc[~out["temp_c"].between(-15, 47), "temp_c"] = np.nan
    out.loc[out["dewpoint_c"] > out["temp_c"] + 0.5, "dewpoint_c"] = np.nan
    return out


def load_station(usaf: str, raw_dir: Path = RAW) -> pd.DataFrame:
    frames = [parse_isd_lite((raw_dir / f"{usaf}-{y}.gz").read_bytes()) for y in YEARS]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("time_utc").sort_values("time_utc")
    df["usaf"] = usaf
    return df.reset_index(drop=True)


def kathmandu_hourly(raw_dir: Path = RAW) -> pd.DataFrame:
    """Kathmandu hourly temperature on the NPT clock: observations land at hh:45, so they are
    assigned to the following clock hour (07:45 -> 08:00), then short gaps are interpolated."""
    df = load_station("444540", raw_dir)
    df["hour"] = (df["time"] + pd.Timedelta(minutes=15)).dt.floor("h")
    s = df.groupby("hour")["temp_c"].mean()
    full = pd.date_range(s.index.min().normalize(), s.index.max().floor("D") - pd.Timedelta(hours=1), freq="h")
    s = s.reindex(full)
    gap = s.isna()
    filled = s.interpolate(limit=MAX_INTERP_HOURS, limit_area="inside")
    out = pd.DataFrame({"ktm_temp": filled, "ktm_temp_imputed": gap & filled.notna()})
    out.index.name = "time"
    return out


def terai_daily(raw_dir: Path = RAW) -> pd.DataFrame:
    """Afternoon temperature and dew point across the Terai stations, one row per NPT date."""
    rows = []
    for usaf, (_, role) in STATIONS.items():
        if role != "terai":
            continue
        df = load_station(usaf, raw_dir)
        aft = df[df["time_utc"].dt.hour == AFTERNOON_UTC_HOUR].copy()
        aft["date"] = aft["time"].dt.normalize()
        rows.append(aft[["date", "usaf", "temp_c", "dewpoint_c"]])
    t = pd.concat(rows).dropna(subset=["temp_c"])
    daily = t.groupby("date").agg(
        terai_tmax=("temp_c", "mean"), terai_dew=("dewpoint_c", "mean"), terai_n=("usaf", "nunique")
    )
    return daily


def daily_weather(raw_dir: Path = RAW) -> pd.DataFrame:
    """One row per date with every weather input the twin and the model use, plus imputation flags.

    When no Terai station reported, the Terai afternoon temperature is estimated from Kathmandu's
    afternoon temperature with a per-month linear fit learned on days where both exist. It is flagged,
    never passed off as an observation.
    """
    ktm = kathmandu_hourly(raw_dir)
    d = ktm["ktm_temp"].groupby(ktm.index.normalize()).agg(["min", "max", "mean", "count"])
    d.columns = ["ktm_tmin", "ktm_tmax", "ktm_tmean", "ktm_hours"]
    d.loc[d["ktm_hours"] < 18, ["ktm_tmin", "ktm_tmax", "ktm_tmean"]] = np.nan
    d[["ktm_tmin", "ktm_tmax", "ktm_tmean"]] = d[["ktm_tmin", "ktm_tmax", "ktm_tmean"]].interpolate(limit=2)
    terai = terai_daily(raw_dir)
    d = d.join(terai, how="left")
    d["terai_n"] = d["terai_n"].fillna(0).astype(int)
    d["terai_imputed"] = d["terai_tmax"].isna()

    both = d.dropna(subset=["terai_tmax", "ktm_tmax"])
    for month in range(1, 13):
        m = both[both.index.month == month]
        if len(m) < 20:
            continue
        a, b = np.polyfit(m["ktm_tmax"], m["terai_tmax"], 1)
        a_d, b_d = np.polyfit(m["ktm_tmax"], m["terai_dew"].fillna(m["terai_dew"].mean()), 1)
        mask = d["terai_imputed"] & (d.index.month == month) & d["ktm_tmax"].notna()
        d.loc[mask, "terai_tmax"] = a * d.loc[mask, "ktm_tmax"] + b
        d.loc[mask, "terai_dew"] = a_d * d.loc[mask, "ktm_tmax"] + b_d
    d["terai_dew"] = d["terai_dew"].fillna(d["terai_dew"].rolling(7, min_periods=1, center=True).mean())
    d.index.name = "date"
    return d.drop(columns=["ktm_hours"])


def quality_report(raw_dir: Path = RAW) -> pd.DataFrame:
    """Share of expected observations present, per station and year (Kathmandu: 24/day, Terai: 1 afternoon synop/day)."""
    out = []
    for usaf, (name, role) in STATIONS.items():
        df = load_station(usaf, raw_dir)
        df = df.dropna(subset=["temp_c"])
        for year in YEARS:
            y = df[df["time"].dt.year == year]
            days = 366 if year % 4 == 0 else 365
            if year == YEARS[-1]:
                days = int((pd.Timestamp("2025-08-24") - pd.Timestamp("2025-01-01")).days)
            if role == "valley":
                present = y["time"].dt.floor("h").nunique() / (24 * days)
            else:
                present = (y["time_utc"].dt.hour == AFTERNOON_UTC_HOUR).sum() / days
            out.append({"station": name, "year": year, "completeness": round(float(present), 3)})
    return pd.DataFrame(out).pivot(index="station", columns="year", values="completeness")
