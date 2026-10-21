"""Figures for the README, drawn from artifacts/reports (run twin, backtest and replay first)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORTS, IMG = ROOT / "artifacts" / "reports", ROOT / "docs" / "img"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK, MUTED = "#8c8c86", "#262624", "#6b6a64"

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#e6e5df", "grid.linewidth": 0.8, "lines.linewidth": 2, "figure.dpi": 130,
    "legend.frameon": False, "axes.titleweight": "bold", "axes.titlesize": 11,
})


def twin_vs_nea():
    v = json.loads((REPORTS / "twin_validation.json").read_text())
    d = pd.DataFrame(v["daily"]).assign(date=lambda x: pd.to_datetime(x["date"]))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(d["date"], d["energy_requirement_mwh"] / 1000, color=INK, label="NEA daily report (real)")
    ax[0].plot(d["date"], d["energy_mwh"] / 1000, color=BLUE, label="Grid twin")
    ax[0].set_title(f"Daily energy, GWh (error {v['energy_mape']:.1%})")
    ax[1].plot(d["date"], d["peak_requirement_mw"], color=INK, label="NEA daily report (real)")
    ax[1].plot(d["date"], d["peak_mw"], color=BLUE, label="Grid twin")
    ax[1].set_title(f"Daily peak demand, MW (error {v['peak_mape']:.1%})")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    ax[0].legend(loc="lower right")
    fig.suptitle("The twin against 40 real NEA operational reports it was not fitted to (Dec 2024 - Jan 2025)",
                 fontsize=10, color=MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(IMG / "twin_vs_nea.png")


def backtest_by_month():
    s = json.loads((REPORTS / "backtest_summary.json").read_text())
    m = pd.DataFrame(s["by_month"]).T
    m.index = pd.PeriodIndex(m.index, freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(m.index, m["yesterday"] * 100, color=GRAY, label="Same hour yesterday")
    ax.plot(m.index, m["bijuli"] * 100, color=BLUE, label="Bijuli (LightGBM quantile)")
    ax.set_ylabel("Hourly MAPE, %")
    ax.set_ylim(0, None)
    ax.set_title("Day-ahead error by month, walk-forward backtest")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "backtest_by_month.png")


def example_week():
    p = pd.read_parquet(REPORTS / "backtest_predictions.parquet")
    p["time"] = pd.to_datetime(p["target_day"]) + pd.to_timedelta(p["hour"], unit="h")
    w = p[(p["time"] >= "2025-01-06") & (p["time"] < "2025-01-13")]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.fill_between(w["time"], w["p10"], w["p90"], color=BLUE, alpha=0.15, linewidth=0, label="p10 - p90")
    ax.plot(w["time"], w["y"], color=INK, label="Demand (twin)")
    ax.plot(w["time"], w["p50"], color=BLUE, label="Forecast p50, issued 10:00 the day before")
    ax.plot(w["time"], w["bid"], color=ORANGE, linewidth=1.2, linestyle="--", label="Purchase basis (q0.62)")
    ax.set_ylabel("MW")
    ax.set_title("A winter week: evening peaks after sunset, Saturday 11 January lower")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / "example_week.png")


def replay_timeline():
    files = {
        "Train once": ("replay_train_once_never_retrain.csv", GRAY),
        "Every 4 weeks, auto-deploy": ("replay_retrain_every_4_weeks_auto-deploy.csv", ORANGE),
        "Every 4 weeks or on bias drift, gated (Bijuli)": ("replay_every_4_weeks_or_on_bias_drift_gated.csv", BLUE),
    }
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for name, (f, color) in files.items():
        d = pd.read_csv(REPORTS / f, parse_dates=["issue_day"]).set_index("issue_day")
        ax.plot(d.index, d["mape"].rolling(14, min_periods=7).mean() * 100, color=color, label=name)
    for x, label in (("2024-08-15", "bad release\n(Fahrenheit feed)"), ("2024-11-20", "+60 MW feeder"),
                     ("2025-04-10", "+45 MW feeder")):
        ax.axvline(pd.Timestamp(x), color=MUTED, linewidth=0.8, linestyle=":")
        ax.text(pd.Timestamp(x), ax.get_ylim()[1] * 0.97, " " + label, fontsize=8, color=MUTED, va="top")
    ax.set_ylabel("14-day rolling MAPE, %")
    ax.set_title("26 months of daily operations, same days and same faults, three policies")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / "replay_timeline.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    for fn in (twin_vs_nea, backtest_by_month, example_week, replay_timeline):
        try:
            fn()
            print("ok", fn.__name__)
        except FileNotFoundError as e:
            print("skipped", fn.__name__, e)
