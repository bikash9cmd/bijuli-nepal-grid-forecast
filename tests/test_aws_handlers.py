"""The Lambda handlers against mocked AWS (moto): a full day, then a shadow promotion and a rollback."""

import io
import json
import os

import boto3
import joblib
import pandas as pd
import pytest
from moto import mock_aws

from bijuli.models import QuantileForecaster

os.environ.setdefault("AWS_DEFAULT_REGION", "ap-south-1")
for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
    os.environ[k] = "testing"


@pytest.fixture
def aws(small_frame):
    f, load, w = small_frame
    with mock_aws():
        from bijuli.aws import handlers as h

        h._model_cache.clear()
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket=h.BUCKET, CreateBucketConfiguration={"LocationConstraint": "ap-south-1"})
        ddb = boto3.client("dynamodb")
        ddb.create_table(TableName=h.FORECAST_TABLE, BillingMode="PAY_PER_REQUEST",
                         AttributeDefinitions=[{"AttributeName": "target_day", "AttributeType": "S"},
                                               {"AttributeName": "version", "AttributeType": "S"}],
                         KeySchema=[{"AttributeName": "target_day", "KeyType": "HASH"},
                                    {"AttributeName": "version", "KeyType": "RANGE"}])
        ddb.create_table(TableName=h.STATE_TABLE, BillingMode="PAY_PER_REQUEST",
                         AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
                         KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}])
        ssm = boto3.client("ssm")
        for name in (h.CHAMPION_PARAM, h.SHADOW_PARAM):
            ssm.put_parameter(Name=name, Value="none", Type="String")

        for v, seed in (("v001", 0), ("v002", 1)):
            m = QuantileForecaster(params={"n_estimators": 30}).fit(f, seed=seed, calib_days=14)
            buf = io.BytesIO()
            joblib.dump(m, buf)
            s3.put_object(Bucket=h.BUCKET, Key=f"models/{v}/model.joblib", Body=buf.getvalue())
            s3.put_object(Bucket=h.BUCKET, Key=f"models/{v}/card.json",
                          Body=json.dumps({"holdout_mape": m.holdout_mape}).encode())
        ssm.put_parameter(Name=h.CHAMPION_PARAM, Value="v001", Type="String", Overwrite=True)

        wx = w.drop(columns=["terai_imputed"])
        wx = wx.join(wx.add_prefix("fc_")).reset_index()
        s3.put_object(Bucket=h.BUCKET, Key="weather/daily.csv", Body=wx.to_csv(index=False).encode())
        for day, g in load.groupby(load.index.normalize()):
            s3.put_object(Bucket=h.BUCKET, Key=f"raw/scada/date={day.date()}/load.csv",
                          Body=g.rename("demand_mw").rename_axis("time").reset_index().to_csv(index=False).encode())
        yield h, load


def _day(h, d):
    h.ingest({"date": d})
    out = h.forecast({"issue_day": d})
    mon = h.monitor_daily({"issue_day": d})
    return out, mon


def test_a_day_end_to_end_and_the_api(aws):
    h, load = aws
    for d in pd.date_range("2024-02-01", "2024-02-20"):
        h.ingest({"date": str(d.date())})
    out, _ = _day(h, "2024-02-21")
    assert out["status"] == "ok" and out["champion"]["version"] == "v001"
    resp = h.api({"pathParameters": {"day": "2024-02-22"}})
    body = json.loads(resp["body"])
    assert resp["statusCode"] == 200 and len(body["p50"]) == 24 and not body["degraded"]
    assert h.api({"pathParameters": {"day": "22-02-2024"}})["statusCode"] == 400
    assert h.api({"pathParameters": {"day": "2030-01-01"}})["statusCode"] == 404


def test_ingest_quarantines_a_kw_day(aws):
    h, load = aws
    for d in pd.date_range("2024-02-01", "2024-02-09"):
        h.ingest({"date": str(d.date())})
    s3 = boto3.client("s3")
    g = load["2024-02-10"] * 1000
    s3.put_object(Bucket=h.BUCKET, Key="raw/scada/date=2024-02-10/load.csv",
                  Body=g.rename("demand_mw").rename_axis("time").reset_index().to_csv(index=False).encode())
    res = h.ingest({"date": "2024-02-10"})
    assert res["findings"][0]["check"] == "unit"
    clean = pd.read_csv(io.BytesIO(s3.get_object(Bucket=h.BUCKET, Key="clean/load/date=2024-02-10/load.csv")["Body"].read()))
    assert clean["demand_mw"].isna().all()


def test_forecast_falls_back_then_refuses_when_load_data_is_gone(aws):
    h, _ = aws
    for d in pd.date_range("2024-02-01", "2024-02-20"):
        if d != pd.Timestamp("2024-02-19"):  # yesterday's export never arrived
            h.ingest({"date": str(d.date())})
    out = h.forecast({"issue_day": "2024-02-20"})
    assert out["status"] == "ok"
    item = boto3.resource("dynamodb").Table(h.FORECAST_TABLE).get_item(Key={"target_day": "2024-02-21", "version": "v001"})["Item"]
    assert item["degraded"] is True
    assert h.forecast({"issue_day": "2024-06-01"})["status"] == "refused"


def test_shadow_is_promoted_then_rolled_back_by_the_monitor(aws, monkeypatch):
    h, _ = aws
    for d in pd.date_range("2024-02-01", "2024-02-10"):
        h.ingest({"date": str(d.date())})
    h.register_shadow({"version": "v002", "issue_day": "2024-02-11"})
    # Make the shadow look better than the champion, then make it look much worse after promotion.
    real_score = h._score
    phase = {"v": "promote"}

    def fake_score(version, day, actual):
        s = real_score(version, day, actual)
        if s and phase["v"] == "promote":
            s["mape"] = 0.01 if version == "v002" else 0.05
        elif s:
            s["mape"] = 0.09 if version == "v002" else 0.03
        return s

    monkeypatch.setattr(h, "_score", fake_score)
    decisions = []
    for d in pd.date_range("2024-02-11", "2024-02-28"):
        out, mon = _day(h, str(d.date()))
        decisions += mon["decisions"]
        if any(x["action"] == "promote" for x in mon["decisions"]):
            phase["v"] = "rollback"
    actions = [x["action"] for x in decisions if x["action"] != "retrain"]
    assert actions[:2] == ["promote", "rollback"]
    assert boto3.client("ssm").get_parameter(Name=h.CHAMPION_PARAM)["Parameter"]["Value"] == "v001"
