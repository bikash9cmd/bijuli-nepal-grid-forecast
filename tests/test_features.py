import numpy as np
import pandas as pd

from bijuli import features

from .conftest import synthetic_load, synthetic_weather


def test_nothing_after_the_cutoff_leaks_into_a_forecast():
    """Corrupt everything that happens after 09:00 on the issue day; tomorrow's features must not change."""
    load, w = synthetic_load(), synthetic_weather()
    target = pd.Timestamp("2024-03-15")
    issue = target - pd.Timedelta(days=1)
    before = features.build(load, w, w, target_days=pd.DatetimeIndex([target]), with_target=False)

    tampered = load.copy()
    tampered[issue + pd.Timedelta(hours=10):] = 99999.0
    w2 = w.copy()
    w2.loc[issue:, ["ktm_tmean", "ktm_tmin", "ktm_tmax", "terai_tmax", "terai_dew"]] = -50.0  # observed weather
    after = features.build(tampered, w2, w, target_days=pd.DatetimeIndex([target]), with_target=False)
    pd.testing.assert_frame_equal(before[features.FEATURES], after[features.FEATURES])


def test_outage_days_stay_visible_as_missing_rows():
    load = synthetic_load(days=40)
    load["2024-01-20":"2024-01-21"] = np.nan
    m = features.load_matrix(load)
    assert pd.Timestamp("2024-01-20") in m.index and m.loc["2024-01-20"].isna().all()
    f = features.build(load, synthetic_weather(), synthetic_weather(), target_days=pd.DatetimeIndex(["2024-01-22"]))
    assert f["anchor"].isna().all()  # yesterday (21st) is gone ...
    assert f["lag_w_h"].notna().all()  # ... but last week's same hour is there for the fallback


def test_one_row_per_hour_with_sunset_aligned_hours():
    load, w = synthetic_load(), synthetic_weather()
    f = features.build(load, w, w, target_days=pd.DatetimeIndex(["2024-02-10"]))
    assert len(f) == 24 and list(f["hour"]) == list(range(24))
    assert f["is_saturday"].iloc[0] == 1  # 10 Feb 2024 was a Saturday
    assert f["hours_after_sunset"].between(-19, 7).all()
