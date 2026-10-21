import numpy as np
import pandas as pd
import pytest

from bijuli import features
from bijuli.config import RAW


def synthetic_load(start="2024-01-01", days=120, seed=0) -> pd.Series:
    """A plausible hourly MW series: evening peak, Saturday dip, slow growth, noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    h = idx.hour.to_numpy()
    shape = 1 + 0.35 * np.exp(-0.5 * ((h - 18.5) / 1.4) ** 2) - 0.25 * np.exp(-0.5 * ((h - 3.5) / 2) ** 2)
    level = 1400 * (1 + 0.0002 * np.arange(len(idx)) / 24)
    sat = np.where(idx.dayofweek == 5, 0.95, 1.0)
    return pd.Series(level * shape * sat * (1 + rng.normal(0, 0.01, len(idx))), index=idx, name="demand_mw")


def synthetic_weather(start="2023-12-01", days=200, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days, freq="D")
    doy = idx.dayofyear.to_numpy()
    base = 19 - 7 * np.cos(2 * np.pi * (doy - 15) / 365)
    df = pd.DataFrame(index=idx)
    df["ktm_tmean"] = base + rng.normal(0, 1, days)
    df["ktm_tmin"] = df["ktm_tmean"] - 6
    df["ktm_tmax"] = df["ktm_tmean"] + 6
    df["terai_tmax"] = df["ktm_tmax"] + 5
    df["terai_dew"] = 15 + rng.normal(0, 2, days)
    df["terai_imputed"] = False
    df.index.name = "date"
    return df


@pytest.fixture(scope="session")
def small_frame():
    load = synthetic_load()
    w = synthetic_weather()
    f = features.build(load, w, w)
    return f[features.usable(f)], load, w


requires_data = pytest.mark.skipif(not (RAW / "444540-2024.gz").exists(), reason="run `python -m bijuli download` first")
