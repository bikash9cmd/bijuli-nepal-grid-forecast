"""Walk-forward backtest: for each month from January 2023, train on the 365 days before it and
forecast every day of the month, 10:00 the day before, exactly as the daily job would."""

from __future__ import annotations

import pandas as pd

from . import features, metrics
from .config import REPORTS
from .models import BASELINES, QuantileForecaster, feature_importance

START, END = "2023-01-01", "2025-08-01"


def run(daily_weather: pd.DataFrame, load: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    frame = features.build(load["demand_mw"], daily_weather, forecast)
    frame = frame[features.usable(frame)]
    parts = []
    model = None
    for month in pd.date_range(START, END, freq="MS"):
        train = frame[(frame["target_day"] < month) & (frame["target_day"] >= month - pd.Timedelta(days=365))]
        test = frame[(frame["target_day"] >= month) & (frame["target_day"] < month + pd.offsets.MonthBegin(1))].copy()
        model = QuantileForecaster().fit(train)
        test = test.join(model.predict(test))
        for name, fn in BASELINES.items():
            test[name] = fn(test)
        parts.append(test)
    res = pd.concat(parts)
    REPORTS.mkdir(parents=True, exist_ok=True)
    res.to_parquet(REPORTS / "backtest_predictions.parquet", index=False)

    rows = {name: metrics.summarise(res, name) for name in BASELINES}
    rows["LightGBM quantile (Bijuli)"] = metrics.summarise(res, "p50", bid_col="bid")
    res["month"] = pd.to_datetime(res["target_day"]).dt.to_period("M").astype(str)
    by_month = res.groupby("month").apply(
        lambda g: pd.Series({"bijuli": metrics.mape(g["y"], g["p50"]),
                             "yesterday": metrics.mape(g["y"], g["Same hour yesterday"])}), include_groups=False)
    festival = res[(res["dashain_closure"] == 1) | (res["tihar"] == 1)]
    summary = {
        "period": f"{START} to {res['target_day'].max():%Y-%m-%d}",
        "forecast_hours": int(len(res)),
        "results": rows,
        "festival_days": {"bijuli": metrics.mape(festival["y"], festival["p50"]),
                          "yesterday": metrics.mape(festival["y"], festival["Same hour yesterday"])},
        "by_month": by_month.round(4).to_dict("index"),
        "feature_importance_last_model": feature_importance(model).round(4).to_dict(),
    }
    (REPORTS / "backtest_summary.json").write_text(pd.Series(summary).to_json(indent=2))
    return summary
