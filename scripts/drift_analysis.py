"""What kinds of drift hurt this model, and which detectors see them.

1. Level shifts (two new industrial feeders): how many days of extra error before the model adapts?
2. Shape drift (EV charging growing overnight): bias by time of day for a model trained once.
3. Data drift (PSI): how often does each monitored feature cross 0.25 against the training data?

Writes artifacts/reports/drift_analysis.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bijuli import features, monitor, twin, weather  # noqa: E402
from bijuli.config import DATA_END, DATA_START, REPORTS  # noqa: E402
from bijuli.models import QuantileForecaster  # noqa: E402

TRAIN_END = pd.Timestamp("2023-07-01")


def main():
    w = weather.daily_weather()
    load = twin.generate(w, DATA_START, DATA_END)
    frame = features.build(load["demand_mw"], w, features.weather_forecast(w))
    frame = frame[features.usable(frame)]
    train = frame[(frame["target_day"] < TRAIN_END) & (frame["target_day"] >= TRAIN_END - pd.Timedelta(days=365))]
    test = frame[frame["target_day"] >= TRAIN_END].copy()
    model = QuantileForecaster().fit(train)
    test["p50"] = model.predict(test)["p50"].to_numpy()
    test["err"] = (test["p50"] - test["y"]) / test["y"]
    out = {}

    # 1. level shifts: daily MAPE around each feeder connection
    daily = test.groupby("target_day")["err"].agg(lambda e: e.abs().mean())
    steps = {}
    for when, mw in twin.TwinParams().feeders:
        t = pd.Timestamp(when)
        steps[f"{when} (+{mw:.0f} MW)"] = {
            "mape_week_before": float(daily[t - pd.Timedelta(days=7): t - pd.Timedelta(days=1)].mean()),
            "mape_day_of": float(daily.get(t, np.nan)),
            "mape_day_after": float(daily.get(t + pd.Timedelta(days=1), np.nan)),
            "mape_days_2_to_7": float(daily[t + pd.Timedelta(days=2): t + pd.Timedelta(days=7)].mean()),
        }
    out["level_shift_static_model"] = steps

    # 2. shape drift: bias by time of day, per half-year, for the model trained once
    band = pd.cut(test["hour"], [-1, 5, 16, 18, 23], labels=["00-05 night", "06-16 day", "17-18 peak", "19-23 evening"])
    half = pd.to_datetime(test["target_day"]).dt.year.astype(str) + np.where(
        pd.to_datetime(test["target_day"]).dt.month <= 6, "-H1", "-H2")
    out["bias_by_band_static_model"] = (
        test.groupby([half, band], observed=True)["err"].mean().unstack().round(4).to_dict("index"))
    ev = load["ev_mw"].groupby(load.index.year).mean().round(1).to_dict()
    out["ev_mean_mw_by_year"] = {str(k): v for k, v in ev.items()}

    # 3. PSI on 28-day windows of daily values (day-level features repeat 24 times a day; counting
    #    them per hour would fake precision). Two references: the whole training year, and the same
    #    28 days one year earlier.
    feats = monitor.PSI_FEATURES
    day_train = train.drop_duplicates("target_day")
    day_all = frame.drop_duplicates("target_day").set_index("target_day")
    ref_year = monitor.reference_bins(day_train, feats)
    vs_year, vs_season = [], []
    for start in pd.date_range(TRAIN_END, test["target_day"].max() - pd.Timedelta(days=28), freq="28D"):
        cur = day_all.loc[start: start + pd.Timedelta(days=27)]
        last = day_all.loc[start - pd.Timedelta(days=365): start - pd.Timedelta(days=338)]
        vs_year.append(monitor.psi(ref_year, cur))
        vs_season.append(monitor.psi(monitor.reference_bins(last, feats), cur))
    for name, rows in (("vs_training_year", vs_year), ("vs_same_season_last_year", vs_season)):
        psi = pd.DataFrame(rows)
        out[f"psi_{name}_share_of_windows_above_0.25"] = (psi > 0.25).mean().round(2).to_dict()
        out[f"psi_{name}_median"] = psi.median().round(2).to_dict()

    # Null check: no drift at all, same distribution, only sample size varies.
    rng = np.random.default_rng(0)
    null_ref = monitor.reference_bins(pd.DataFrame({"x": rng.normal(size=365)}), ["x"])
    out["psi_no_drift_share_above_0.25_by_sample_size"] = {
        n: float(np.mean([monitor.psi(null_ref, pd.DataFrame({"x": rng.normal(size=n)}))["x"] > 0.25
                          for _ in range(300)]))
        for n in (28, 90, 365)
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "drift_analysis.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
