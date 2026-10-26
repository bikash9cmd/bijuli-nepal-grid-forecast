import gzip

import numpy as np
import pandas as pd
import pytest

from bijuli import contracts, weather

from .conftest import requires_data, synthetic_load


def _isd(lines: list[str]) -> bytes:
    return gzip.compress(("\n".join(lines) + "\n").encode())


def test_isd_lite_is_converted_to_nepal_time_and_missing_codes_dropped():
    raw = _isd([
        "2024 01 01 00    70    60 -9999 -9999    10 -9999 -9999 -9999",
        "2024 01 01 09 -9999 -9999 -9999 -9999    10 -9999 -9999 -9999",
        "2024 01 01 10   990    60 -9999 -9999    10 -9999 -9999 -9999",
    ])
    df = weather.parse_isd_lite(raw)
    assert df["time"].iloc[0] == pd.Timestamp("2024-01-01 05:45")  # UTC+05:45, not +05:00 or +06:00
    assert df["temp_c"].iloc[0] == 7.0  # tenths of a degree
    assert np.isnan(df["temp_c"].iloc[1])  # -9999 is missing, never -999.9 degrees
    assert np.isnan(df["temp_c"].iloc[2])  # 99 C is a decoding fault


def test_checksum_mismatch_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "STATIONS", {"444540": ("Kathmandu", "valley")})
    monkeypatch.setattr(weather, "YEARS", (2024,))
    (tmp_path / "444540-2024.gz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        weather.download(tmp_path)


@requires_data
def test_real_terai_gap_is_imputed_and_flagged():
    d = weather.daily_weather()
    gap = d.loc["2024-01-01":"2024-03-15"]
    assert gap["terai_imputed"].mean() > 0.9  # the network-wide hole in the NOAA feed
    assert gap["terai_tmax"].notna().all()
    assert d.loc["2022-04-01":"2022-04-30", "terai_imputed"].mean() <= 0.15  # 3 days missing in a normal month


def test_unit_glitch_is_quarantined_not_trained_on():
    load = synthetic_load(days=30)
    bad = load.copy()
    bad["2024-01-10"] = bad["2024-01-10"] * 1000  # kW written into the MW column
    clean, findings = contracts.check_load(bad)
    assert clean["2024-01-10"].isna().all()
    assert clean["2024-01-11"].notna().all()
    assert any(f.check == "unit" and f.hours == 24 for f in findings)


def test_flatline_and_spike_are_caught():
    load = synthetic_load(days=10)
    bad = load.copy()
    bad.iloc[50:56] = bad.iloc[50]  # stuck RTU for six hours
    bad.iloc[100] = bad.iloc[100] * 1.8  # one-hour spike
    clean, findings = contracts.check_load(bad)
    checks = {f.check: f.hours for f in findings}
    assert checks["flatline"] == 6 and clean.iloc[50:56].isna().all()
    assert checks["spike"] == 1 and np.isnan(clean.iloc[100])
    assert clean.iloc[57:99].notna().all()


def test_freshness_needs_the_morning_hours():
    load = synthetic_load(days=3)
    issue = pd.Timestamp(load.index[-1].normalize() + pd.Timedelta(hours=10))
    assert contracts.freshness_ok(load[: issue - pd.Timedelta(hours=4)], issue)  # reaches 06:00
    assert not contracts.freshness_ok(load[: issue - pd.Timedelta(hours=6)], issue)  # stops at 04:00
    assert not contracts.freshness_ok(load * np.nan, issue)  # nothing reported at all


def test_future_stamped_feed_is_not_fresh():
    load = synthetic_load(days=3)
    issue = pd.Timestamp(load.index[-1].normalize() + pd.Timedelta(hours=10))
    shifted = load[: issue - pd.Timedelta(hours=4)].copy()
    shifted.index = shifted.index + pd.Timedelta(hours=5, minutes=45)  # UTC labelled as Nepal time
    assert not contracts.freshness_ok(shifted, issue)


def test_weather_findings_report_estimates_and_gaps():
    days = pd.date_range("2024-01-01", periods=5, freq="D")
    daily = pd.DataFrame({"ktm_tmean": [10.0, np.nan, 11.0, 12.0, 9.0],
                          "terai_imputed": [False, True, True, False, False]}, index=days)
    found = {f.check: f.hours for f in contracts.check_weather(daily)}
    assert found == {"terai_missing": 48, "kathmandu_missing": 24}
    assert contracts.check_weather(daily.drop(columns="terai_imputed")) == [
        f for f in contracts.check_weather(daily) if f.check == "kathmandu_missing"]


def test_weather_without_kathmandu_column_is_a_finding_not_a_crash():
    days = pd.date_range("2024-01-01", periods=2, freq="D")
    found = contracts.check_weather(pd.DataFrame({"terai_imputed": [False, False]}, index=days))
    assert [(f.check, f.hours) for f in found] == [("kathmandu_missing", 48)]
