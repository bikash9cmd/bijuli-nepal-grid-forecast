"""Lambda entry points. Same logic as the local replay, wired to S3, DynamoDB, SSM, SNS and EventBridge.

Daily state machine (10:00 NPT):  ingest -> forecast -> monitor -> (maybe) start the retrain state machine
Retrain state machine:            build_dataset -> SageMaker training (spot) -> offline_gate -> register_shadow
Read path:                        api (HTTP API -> forecasts table)

The champion and shadow model versions live in two SSM parameters, so promotion and rollback are
single writes that every function sees on its next invocation, and the audit trail is the
parameter history plus the events in the registry index.
"""

from __future__ import annotations

import io
import json
import os
from datetime import date, timedelta

import boto3
import joblib
import numpy as np
import pandas as pd
from boto3.dynamodb.conditions import Key

from .. import contracts, features, monitor
from ..config import (
    PROMOTE_MIN_IMPROVEMENT,
    ROLLBACK_TOLERANCE,
    ROLLBACK_WINDOW_DAYS,
    SHADOW_DAYS,
)

BUCKET = os.environ.get("BUCKET", "bijuli-local")
FORECAST_TABLE = os.environ.get("FORECAST_TABLE", "bijuli-forecasts")
STATE_TABLE = os.environ.get("STATE_TABLE", "bijuli-state")
CHAMPION_PARAM = os.environ.get("CHAMPION_PARAM", "/bijuli/champion")
SHADOW_PARAM = os.environ.get("SHADOW_PARAM", "/bijuli/shadow")
TOPIC_ARN = os.environ.get("TOPIC_ARN", "")
EVENT_BUS = os.environ.get("EVENT_BUS", "default")
HISTORY_DAYS = 16  # features reach back 15 days (same weekday last week, and the week before)
NONE = "none"

_model_cache: dict[str, object] = {}


def _s3():
    return boto3.client("s3")


def _ssm():
    return boto3.client("ssm")


def _table(name):
    return boto3.resource("dynamodb").Table(name)


def _day(event, key="issue_day") -> date:
    return date.fromisoformat(event[key]) if key in event else date.today()


def _read_csv(key: str, **kw) -> pd.DataFrame | None:
    try:
        body = _s3().get_object(Bucket=BUCKET, Key=key)["Body"].read()
    except _s3().exceptions.NoSuchKey:
        return None
    return pd.read_csv(io.BytesIO(body), **kw)


def _write_csv(key: str, df: pd.DataFrame) -> None:
    _s3().put_object(Bucket=BUCKET, Key=key, Body=df.to_csv(index=False).encode())


def _param(name: str) -> str | None:
    v = _ssm().get_parameter(Name=name)["Parameter"]["Value"]
    return None if v == NONE else v


def _set_param(name: str, value: str | None) -> None:
    _ssm().put_parameter(Name=name, Value=value or NONE, Type="String", Overwrite=True)


def _load_model(version: str):
    """Cached across warm invocations: the model is downloaded once per container, not per call."""
    if version not in _model_cache:
        body = _s3().get_object(Bucket=BUCKET, Key=f"models/{version}/model.joblib")["Body"].read()
        _model_cache[version] = joblib.load(io.BytesIO(body))
    return _model_cache[version]


def _clean_load(days: list[date]) -> pd.Series:
    frames = [_read_csv(f"clean/load/date={d}/load.csv", parse_dates=["time"]) for d in days]
    frames = [f for f in frames if f is not None]
    if not frames:
        return pd.Series(dtype=float)
    df = pd.concat(frames)
    return df.set_index("time")["demand_mw"].sort_index()


def _state() -> dict:
    item = _table(STATE_TABLE).get_item(Key={"pk": "monitor"}).get("Item")
    return json.loads(item["doc"]) if item else {"cusum_pos": 0.0, "cusum_neg": 0.0, "scores": {}}


def _save_state(state: dict) -> None:
    _table(STATE_TABLE).put_item(Item={"pk": "monitor", "doc": json.dumps(state)})


def _emit(detail_type: str, detail: dict) -> None:
    boto3.client("events").put_events(Entries=[{
        "Source": "bijuli", "DetailType": detail_type, "Detail": json.dumps(detail), "EventBusName": EVENT_BUS,
    }])


# ---------------------------------------------------------------------------------------------
def ingest(event, context=None):
    """Validate the SCADA export for one day and write the clean copy. Quarantined hours become NaN."""
    d = _day(event, "date")
    raw = _read_csv(f"raw/scada/date={d}/load.csv", parse_dates=["time"])
    if raw is None:
        return {"date": str(d), "status": "missing"}
    ref = _clean_load([d - timedelta(days=i) for i in range(1, 29)])
    ref_median = float(ref.median()) if len(ref) else None
    clean, findings = contracts.check_load(raw.set_index("time")["demand_mw"], reference_median=ref_median)
    _write_csv(f"clean/load/date={d}/load.csv", clean.rename("demand_mw").reset_index())
    report = [f.__dict__ for f in findings]
    _s3().put_object(Bucket=BUCKET, Key=f"quality/date={d}/findings.json", Body=json.dumps(report).encode())
    boto3.client("cloudwatch").put_metric_data(Namespace="Bijuli", MetricData=[
        {"MetricName": "QuarantinedHours", "Value": float(sum(f.hours for f in findings)), "Unit": "Count"}])
    return {"date": str(d), "status": "ok", "findings": report}


def forecast(event, context=None):
    """Forecast all 24 hours of tomorrow with the champion, and with the shadow if there is one."""
    d = _day(event)
    target = d + timedelta(days=1)
    load = _clean_load([d - timedelta(days=i) for i in range(HISTORY_DAYS)])
    if load.dropna().empty:
        _emit("ForecastRefused", {"target_day": str(target), "reason": "no load data in the last 16 days"})
        return {"target_day": str(target), "status": "refused"}
    weather = _read_csv("weather/daily.csv", parse_dates=["date"]).set_index("date")
    obs = weather[[c for c in weather.columns if not c.startswith("fc_")]]
    fc = weather[[c for c in weather.columns if c.startswith("fc_")]].rename(columns=lambda c: c[3:])
    rows = features.build(load, obs, fc, target_days=pd.DatetimeIndex([pd.Timestamp(target)]), with_target=False)
    missing = rows["anchor"].isna()
    rows.loc[missing, "anchor"] = rows.loc[missing, "lag_w_h"]  # fallback: last week's same hour
    if rows["anchor"].isna().any():
        _emit("ForecastRefused", {"target_day": str(target), "reason": "no recent load data"})
        return {"target_day": str(target), "status": "refused"}

    out = {}
    for role, param in (("champion", CHAMPION_PARAM), ("shadow", SHADOW_PARAM)):
        version = _param(param)
        if not version:
            continue
        pred = _load_model(version).predict(rows)
        item = {
            "target_day": str(target), "version": version, "role": role, "issued": str(d),
            "degraded": bool(missing.any()),
            **{c: json.dumps([round(float(x), 1) for x in pred[c]]) for c in ("p10", "p50", "p90", "bid")},
        }
        _table(FORECAST_TABLE).put_item(Item=item)
        out[role] = {"version": version, "peak_mw": round(float(pred["p50"].max())),
                     "peak_hour": int(np.argmax(pred["p50"].to_numpy())), "bid_mwh": round(float(pred["bid"].sum()))}
    if TOPIC_ARN and "champion" in out:
        c = out["champion"]
        msg = (f"Bijuli day-ahead {target}: peak ~{c['peak_mw']} MW at {c['peak_hour']:02d}:00. "
               f"Suggested day-ahead purchase basis: {c['bid_mwh']} MWh (0.62 quantile).")
        boto3.client("sns").publish(TopicArn=TOPIC_ARN, Subject=f"Bijuli forecast {target}", Message=msg)
    return {"target_day": str(target), "status": "ok", **out}


def _score(version: str, day: date, actual: pd.Series) -> dict | None:
    item = _table(FORECAST_TABLE).get_item(Key={"target_day": str(day), "version": version}).get("Item")
    if not item or actual.isna().all():
        return None
    p50 = np.array(json.loads(item["p50"]))
    y = actual.reindex(pd.date_range(pd.Timestamp(day), periods=24, freq="h")).to_numpy()
    e = (p50 - y) / y
    return {"mape": float(np.nanmean(np.abs(e))), "bias": float(np.nanmean(e))}


def _window_mape(state, version, days) -> float:
    vals = [state["scores"].get(version, {}).get(str(x), {}).get("mape") for x in days]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def monitor_daily(event, context=None):
    """Score yesterday, update the CUSUM, then decide: promote the shadow, roll back, or ask for a retrain."""
    d = _day(event)
    y_day = d - timedelta(days=1)
    actual = _clean_load([y_day])
    state = _state()
    champ, shadow = _param(CHAMPION_PARAM), _param(SHADOW_PARAM)
    decisions = []

    for v in {champ, shadow, state.get("previous")} - {None}:
        s = _score(v, y_day, actual)
        if s:
            state["scores"].setdefault(v, {})[str(y_day)] = s
    today = state["scores"].get(champ, {}).get(str(y_day))
    if today:
        cus = monitor.Cusum(pos=state["cusum_pos"], neg=state["cusum_neg"])
        cus.update(today["bias"])
        state["cusum_pos"], state["cusum_neg"] = cus.pos, cus.neg
        boto3.client("cloudwatch").put_metric_data(Namespace="Bijuli", MetricData=[
            {"MetricName": "DailyMAPE", "Value": today["mape"] * 100, "Unit": "Percent"},
            {"MetricName": "DailyBias", "Value": today["bias"] * 100, "Unit": "Percent"},
            {"MetricName": "CusumMax", "Value": max(cus.pos, cus.neg), "Unit": "None"}])

    # Rollback watch: the previous champion keeps forecasting for a few days after a promotion.
    if state.get("previous") and state.get("promoted_on"):
        since = date.fromisoformat(state["promoted_on"])
        if (d - since).days >= ROLLBACK_WINDOW_DAYS:
            days = [since + timedelta(days=i) for i in range(1, (d - since).days)]
            new, old = _window_mape(state, champ, days), _window_mape(state, state["previous"], days)
            if new > old * ROLLBACK_TOLERANCE:
                _set_param(CHAMPION_PARAM, state["previous"])
                decisions.append({"action": "rollback", "from": champ, "to": state["previous"], "new": new, "old": old})
                champ = state["previous"]
            state["previous"], state["promoted_on"] = None, None

    # Shadow evaluation
    if shadow and state.get("shadow_since"):
        since = date.fromisoformat(state["shadow_since"])
        if (d - since).days > SHADOW_DAYS:
            days = [since + timedelta(days=i) for i in range(2, (d - since).days)]
            cand, cur = _window_mape(state, shadow, days), _window_mape(state, champ, days)
            alarm = max(state["cusum_pos"], state["cusum_neg"]) > monitor.CUSUM_H
            if cand <= cur * (1 - PROMOTE_MIN_IMPROVEMENT) or (alarm and cand <= cur):
                _set_param(CHAMPION_PARAM, shadow)
                state.update(previous=champ, promoted_on=str(d), cusum_pos=0.0, cusum_neg=0.0)
                decisions.append({"action": "promote", "version": shadow, "shadow_mape": cand, "champion_mape": cur})
            else:
                decisions.append({"action": "reject", "version": shadow, "shadow_mape": cand, "champion_mape": cur})
            _set_param(SHADOW_PARAM, None)
            state["shadow_since"] = None

    if max(state["cusum_pos"], state["cusum_neg"]) > monitor.CUSUM_H and not _param(SHADOW_PARAM):
        decisions.append({"action": "retrain", "reason": "bias drift"})
        _emit("RetrainRequested", {"issue_day": str(d), "reason": "bias drift"})

    for dec in decisions:
        if dec["action"] != "retrain":
            _emit(f"Model{dec['action'].title()}", {"issue_day": str(d), **dec})
    _save_state(state)
    return {"issue_day": str(d), "decisions": decisions}


def build_dataset(event, context=None):
    """First step of the retrain state machine: the trailing training window as one parquet file,
    built from clean data only, with its content hash for lineage."""
    from ..config import TRAIN_WINDOW_DAYS
    from ..models import frame_hash

    d = _day(event)
    days = [d - timedelta(days=i) for i in range(TRAIN_WINDOW_DAYS + 8)]
    load = _clean_load(days)
    weather = _read_csv("weather/daily.csv", parse_dates=["date"]).set_index("date")
    obs = weather[[c for c in weather.columns if not c.startswith("fc_")]]
    fc = weather[[c for c in weather.columns if c.startswith("fc_")]].rename(columns=lambda c: c[3:])
    frame = features.build(load, obs, fc)
    frame = frame[features.usable(frame) & (frame["target_day"] < pd.Timestamp(d))]
    key = f"training/issue_day={d}/train.parquet"
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    _s3().put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    return {"issue_day": str(d), "train_s3": f"s3://{BUCKET}/{key}", "rows": int(len(frame)), "sha256": frame_hash(frame)}


def publish_model(event, context=None):
    """Unpack SageMaker's model.tar.gz into models/<version>/ so serving reads plain files."""
    import tarfile

    version, artifact = event["version"], event["model_artifact"]  # s3://bucket/key/model.tar.gz
    bucket, key = artifact[5:].split("/", 1)
    body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        for name in ("model.joblib", "card.json"):
            member = tar.extractfile(name)
            _s3().put_object(Bucket=BUCKET, Key=f"models/{version}/{name}", Body=member.read())
    return {"version": version, "issue_day": event["issue_day"]}


def offline_gate(event, context=None):
    """After training: the candidate's held-out MAPE must not be worse than what the champion has
    actually delivered over the last 14 days. Writes the model card either way."""
    version, d = event["version"], _day(event)
    card = json.loads(_s3().get_object(Bucket=BUCKET, Key=f"models/{version}/card.json")["Body"].read())
    state = _state()
    champ = _param(CHAMPION_PARAM)
    live = _window_mape(state, champ, [d - timedelta(days=i) for i in range(1, 15)])
    cand = card["holdout_mape"]
    passed = not np.isfinite(live) or cand <= live
    card.update(status="candidate" if passed else "rejected", gate={"candidate": cand, "champion_live": live})
    _s3().put_object(Bucket=BUCKET, Key=f"models/{version}/card.json", Body=json.dumps(card).encode())
    return {"version": version, "issue_day": str(d), "passed": bool(passed), "candidate": cand, "champion_live": live}


def register_shadow(event, context=None):
    """End of the retrain state machine: the candidate passed the offline gate, start its shadow period."""
    _set_param(SHADOW_PARAM, event["version"])
    state = _state()
    state["shadow_since"] = event["issue_day"]
    _save_state(state)
    return {"shadow": event["version"]}


def api(event, context=None):
    """GET /forecast/{day}: the champion's forecast for that day (HTTP API, payload v2)."""
    day = (event.get("pathParameters") or {}).get("day")
    try:
        date.fromisoformat(day or "")
    except ValueError:
        return {"statusCode": 400, "body": json.dumps({"error": "day must be YYYY-MM-DD"})}
    items = _table(FORECAST_TABLE).query(
        KeyConditionExpression=Key("target_day").eq(day)).get("Items", [])
    champ = [i for i in items if i["role"] == "champion"]
    if not champ:
        return {"statusCode": 404, "body": json.dumps({"error": f"no forecast for {day}"})}
    i = champ[-1]
    body = {"target_day": day, "version": i["version"], "degraded": i["degraded"],
            **{c: json.loads(i[c]) for c in ("p10", "p50", "p90", "bid")}}
    return {"statusCode": 200, "headers": {"content-type": "application/json"}, "body": json.dumps(body)}
