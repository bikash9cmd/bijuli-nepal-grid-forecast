"""Command line: python -m bijuli <command>. Every number in the README comes from one of these."""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from . import features, ops, twin, weather
from .config import DATA_END, DATA_START, REGISTRY_DIR, REPORTS


def _inputs():
    w = weather.daily_weather()
    load = twin.generate(w, DATA_START, DATA_END)
    return w, load, features.weather_forecast(w)


def _write(name: str, obj) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / name).write_text(json.dumps(obj, indent=2, default=str))
    print(f"wrote {REPORTS / name}")


def cmd_download(_):
    paths = weather.download()
    print(f"{len(paths)} NOAA ISD-Lite files present and verified")


def cmd_quality(_):
    print("Share of expected observations present (Kathmandu: hourly; Terai/Pokhara: one afternoon synop a day)")
    print(weather.quality_report().to_string())
    d = weather.daily_weather()
    print(f"\nDays with no Terai observation (estimated from Kathmandu): {int(d['terai_imputed'].sum())}")
    gaps = d.index[d["terai_imputed"]]
    if len(gaps):
        runs = (gaps.to_series().diff() != pd.Timedelta(days=1)).cumsum()
        longest = gaps.to_series().groupby(runs.values).agg(["min", "max", "count"]).sort_values("count").iloc[-1]
        print(f"Longest Terai gap: {longest['min']:%Y-%m-%d} to {longest['max']:%Y-%m-%d} ({longest['count']} days)")


def cmd_twin(_):
    w, load, _ = _inputs()
    cal = twin.calibration_report(load)
    val = twin.validate_against_daily_reports(load)
    print(cal.to_string(index=False))
    print({k: v for k, v in val.items() if k != "table"})
    _write("twin_validation.json", {"calibration_fy2081_82": cal.to_dict("records"),
                                    **{k: v for k, v in val.items() if k != "table"},
                                    "daily": val["table"].reset_index().astype({"date": str}).to_dict("records")})


def cmd_backtest(_):
    from .backtest import run

    t = time.time()
    summary = run(*_inputs())
    print(json.dumps(summary, indent=1))
    print(f"{time.time() - t:.0f} s")


def cmd_replay(args):
    w, load, fc = _inputs()
    clean, raw, truth, findings = ops.prepare(load["demand_mw"], w, fc, features.build)
    print("contracts:", [f.__dict__ for f in findings])
    path = REPORTS / "replay_summary.json"
    previous = {r["policy"]: r for r in json.loads(path.read_text())} if path.exists() else {}
    policies = ops.POLICIES if not args.policy else [ops.POLICIES[i] for i in args.policy]
    for p in policies:
        t = time.time()
        r = ops.replay(p, clean, raw, truth, args.start, args.end)
        print(f"{p.name}: {r.summary} ({time.time() - t:.0f} s)", flush=True)
        r.daily.to_csv(REPORTS / f"replay_{p.name.split(' (')[0].replace(' ', '_').replace(',', '').lower()}.csv")
        previous[p.name] = {**r.summary, "events": r.events}
    order = [p.name for p in ops.POLICIES]
    _write("replay_summary.json", sorted(previous.values(), key=lambda r: order.index(r["policy"]) if r["policy"] in order else 99))


def cmd_train(args):
    """Train today's model on the trailing year and register it as champion in the local registry."""
    from .models import QuantileForecaster
    from .monitor import reference_bins
    from .registry import ModelCard, Registry

    w, load, fc = _inputs()
    frame = features.build(load["demand_mw"], w, fc)
    end = pd.Timestamp(args.until)
    f = frame[features.usable(frame) & (frame["target_day"] < end) & (frame["target_day"] >= end - pd.Timedelta(days=365))]
    m = QuantileForecaster().fit(f)
    reg = Registry(REGISTRY_DIR)
    v = reg.next_version()
    reg.register(m, ModelCard(version=v, created=str(end.date()), status="candidate", reason="manual train",
                              train_start=m.meta["train_start"], train_end=m.meta["train_end"],
                              data_sha256=m.meta["data_sha256"],
                              offline={"holdout_mape": m.holdout_mape, "conformal_margin": m.margin},
                              reference=reference_bins(f), params=m.params))
    reg.promote(v, str(end.date()), "manual train")
    print(f"registered and promoted {v}: holdout MAPE {m.holdout_mape:.2%}, data {m.meta['data_sha256']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bijuli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").set_defaults(fn=cmd_download)
    sub.add_parser("quality").set_defaults(fn=cmd_quality)
    sub.add_parser("twin").set_defaults(fn=cmd_twin)
    sub.add_parser("backtest").set_defaults(fn=cmd_backtest)
    r = sub.add_parser("replay")
    r.add_argument("--start", default="2023-07-01")
    r.add_argument("--end", default="2025-08-21")
    r.add_argument("--policy", type=int, nargs="*")
    r.set_defaults(fn=cmd_replay)
    t = sub.add_parser("train")
    t.add_argument("--until", default="2025-08-01")
    t.set_defaults(fn=cmd_train)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
