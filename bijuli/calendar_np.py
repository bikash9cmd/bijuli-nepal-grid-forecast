"""Nepal's calendar as the grid sees it: Bikram Sambat months, the fiscal year, Saturdays and festivals."""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import nepali_datetime as nd
import numpy as np
import pandas as pd

BS_MONTHS = ["Baishakh", "Jestha", "Ashadh", "Shrawan", "Bhadra", "Ashwin",
             "Kartik", "Mangsir", "Poush", "Magh", "Falgun", "Chaitra"]

# Vijaya Dashami and Laxmi Puja in Nepal (Gregorian). Lunar festivals move by up to three weeks a year,
# which is exactly why a model with only "month" and "day of year" gets October and November wrong.
DASHAMI = {2022: "2022-10-05", 2023: "2023-10-24", 2024: "2024-10-12", 2025: "2025-10-02"}
LAXMI_PUJA = {2022: "2022-10-24", 2023: "2023-11-12", 2024: "2024-11-01", 2025: "2025-10-20"}


@lru_cache(maxsize=4096)
def to_bs(date: dt.date) -> tuple[int, int, int]:
    d = nd.date.from_datetime_date(date)
    return d.year, d.month, d.day


def fiscal_year(date: dt.date) -> str:
    """Nepal's fiscal year starts on 1 Shrawan (mid-July): 2081/82 runs 16 Jul 2024 to 16 Jul 2025."""
    y, m, _ = to_bs(date)
    start = y if m >= 4 else y - 1
    return f"{start}/{str(start + 1)[-2:]}"


def festival_offsets(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Days to the nearest Dashami and Laxmi Puja (clipped to +-20). Signed offsets let a model
    learn the shape of the shutdown, not just a 0/1 flag."""
    out = {}
    for name, table in (("dashami", DASHAMI), ("laxmi_puja", LAXMI_PUJA)):
        anchors = pd.to_datetime(list(table.values()))
        diff = (dates.values[:, None] - anchors.values[None, :]).astype("timedelta64[D]").astype(int)
        nearest = diff[np.arange(len(dates)), np.abs(diff).argmin(axis=1)]
        out[f"days_to_{name}"] = np.clip(nearest, -20, 20)
    return pd.DataFrame(out, index=dates)


def calendar_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features for a set of dates (normalised to midnight)."""
    dates = pd.DatetimeIndex(dates).normalize()
    bs = [to_bs(d.date()) for d in dates]
    df = pd.DataFrame(index=dates)
    df["bs_year"] = [b[0] for b in bs]
    df["bs_month"] = [b[1] for b in bs]
    df["dow"] = dates.dayofweek  # Saturday = 5 is Nepal's weekly holiday
    df["is_saturday"] = (df["dow"] == 5).astype(int)
    df["doy"] = dates.dayofyear
    df = df.join(festival_offsets(dates))
    # Dashain closure: Phulpati (Dashami - 3) to Dashami + 4, when industry and offices shut.
    df["dashain_closure"] = df["days_to_dashami"].between(-3, 4).astype(int)
    # Tihar: Kag Tihar (Laxmi Puja - 2) to Bhai Tika (+2)
    df["tihar"] = df["days_to_laxmi_puja"].between(-2, 2).astype(int)
    return df


def sunset_hour(dates: pd.DatetimeIndex) -> np.ndarray:
    """Approximate Kathmandu sunset in decimal NPT hours (about 17:05 in early December, 19:05 in early July).
    The evening peak follows sunset: lights go on while cooking is still running."""
    doy = pd.DatetimeIndex(dates).dayofyear.values
    return 18.1 + 1.0 * np.cos(2 * np.pi * (doy - 180) / 365.25)
