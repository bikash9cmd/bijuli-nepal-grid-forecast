"""Streamlit demo of Bijuli: tomorrow's hourly demand forecast for Nepal's grid, and 26 months of operations.

    streamlit run app.py

Needs the NOAA weather files and a champion in artifacts/registry; if they are missing they are
fetched (checksummed) and trained once, like `make train` (well under a minute).
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

import brand
from bijuli import features, weather
from bijuli.__main__ import _inputs
from bijuli.config import REGISTRY_DIR, REPORTS
from bijuli.registry import Registry

brand.page("bijuli")


@st.cache_resource(show_spinner="Fetching NOAA weather and training the model (first visit only)...")
def load():
    weather.download()
    reg = Registry(REGISTRY_DIR)
    if not reg.champion():
        from argparse import Namespace

        from bijuli.__main__ import cmd_train

        cmd_train(Namespace(until="2025-08-01"))
        reg = Registry(REGISTRY_DIR)
    w, demand, fc = _inputs()
    frame = features.build(demand["demand_mw"], w, fc)
    version = reg.champion()
    return frame, demand, reg.load(version), version, reg.card(version)


frame, demand, model, version, card = load()


forecast_tab, ops_tab, twin_tab = st.tabs(["Day-ahead forecast", "26 months of operations", "The grid twin"])

with forecast_tab:
    days = sorted(frame.loc[features.usable(frame), "target_day"].dt.date.unique())
    target = st.date_input("Forecast for", value=date(2025, 1, 8), min_value=days[0], max_value=days[-1])
    rows = frame[frame["target_day"] == pd.Timestamp(target)]
    if rows.empty or rows["anchor"].isna().any():
        st.warning("Not enough history before that day to forecast it.")
        st.stop()
    pred = model.predict(rows).set_index(rows["hour"].to_numpy())
    actual = demand["demand_mw"].loc[str(target)].to_numpy()
    pred["twin actual"] = actual if len(actual) == 24 else None
    peak = int(pred["p50"].idxmax())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evening peak (median)", f"{pred['p50'].max():,.0f} MW", help=f"at {peak:02d}:00")
    c2.metric("Uncertainty at the peak", f"{pred.loc[peak, 'p10']:,.0f} to {pred.loc[peak, 'p90']:,.0f} MW")
    c3.metric("Day-ahead purchase basis", f"{pred['bid'].sum():,.0f} MWh", help="the 0.62 quantile: a shortfall costs 1.6x an excess")
    if len(actual) == 24:
        mape = (abs(pred["p50"] - actual) / actual).mean()
        c4.metric("Error against the twin", f"{mape:.1%}")
    st.line_chart(
        pred.rename(columns={"p10": "p10", "p50": "median", "p90": "p90", "bid": "purchase basis"}), x_label="hour of day", y_label="MW"
    )
    st.caption(
        f"Champion {version}, trained on {card.get('train_start', '?')} to {card.get('train_end', '?')}. "
        "Days after the training window are out of sample."
    )
    with st.expander("The API response for this day"):
        st.json(
            {
                "target_day": str(target),
                "version": version,
                "degraded": False,
                **{c: [round(float(x)) for x in pred[c]] for c in ("p50", "p10", "p90", "bid")},
            }
        )

with ops_tab:
    st.write(
        "Six policies replayed day by day from July 2023 to August 2025, with real faults injected: "
        "data outages, a broken feature pipeline, new industrial feeders and EV growth."
    )
    summary = json.loads((REPORTS / "replay_summary.json").read_text())
    st.dataframe(pd.DataFrame([{k: v for k, v in r.items() if k != "events"} for r in summary]), hide_index=True, width="stretch")
    daily = {}
    for p in sorted(REPORTS.glob("replay_*.csv")):
        d = pd.read_csv(p, parse_dates=["issue_day"]).set_index("issue_day")["mape"]
        daily[p.stem.removeprefix("replay_").replace("_", " ")] = d.rolling(28, min_periods=7).mean()
    if daily:
        st.subheader("28-day rolling error by policy")
        st.line_chart(pd.DataFrame(daily), y_label="MAPE")
    img = Path("docs/img/replay_timeline.png")
    if img.exists():
        st.image(str(img))

with twin_tab:
    st.write(
        "Hourly national demand isn't published, so Bijuli forecasts a calibrated twin of Nepal's grid, "
        "checked against 40 real NEA daily reports."
    )
    for name in ("twin_vs_nea.png", "example_week.png", "backtest_by_month.png"):
        p = Path("docs/img") / name
        if p.exists():
            st.image(str(p))

brand.footer()
