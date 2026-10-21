"""Monthly AWS cost estimate for Bijuli in ap-south-1 (Mumbai), and the designs it was compared with.

Prices are approximate on-demand list prices, rounded; check the AWS pricing pages before relying on
them. Free tier is ignored, so these are the costs of a second-year account.
"""

DAYS = 30
P = {
    "lambda_gb_s_arm": 0.0000133334,
    "lambda_req": 0.20 / 1e6,
    "sfn_transition": 0.025 / 1000,
    "sm_m5_large_hour": 0.134,
    "spot_discount": 0.70,
    "s3_gb": 0.025,
    "ddb_wru": 1.4225 / 1e6,
    "ddb_rru": 0.285 / 1e6,
    "cw_metric": 0.30,
    "cw_alarm": 0.10,
    "cw_logs_gb": 0.67,
    "ecr_gb": 0.10,
    "httpapi_req": 1.20 / 1e6,
}


def bijuli(retrains_per_month: float = 2.0, api_requests: int = 20_000) -> dict[str, float]:
    # daily run: ingest 5 s @1 GB, forecast 8 s @2 GB, monitor 4 s @1 GB
    lam_daily = (5 * 1 + 8 * 2 + 4 * 1) * DAYS * P["lambda_gb_s_arm"]
    # each retrain: build_dataset 30 s @2 GB, publish + gate + register 10 s @1 GB
    lam_retrain = retrains_per_month * (30 * 2 + 10) * P["lambda_gb_s_arm"]
    lam_api = api_requests * (0.2 * 0.5 * P["lambda_gb_s_arm"] + P["lambda_req"] + P["httpapi_req"])
    # a 3-minute fit, ~6 billable minutes with container start, on spot
    training = retrains_per_month * (6 / 60) * P["sm_m5_large_hour"] * (1 - P["spot_discount"])
    sfn = (8 * DAYS + 10 * retrains_per_month) * P["sfn_transition"]
    return {
        "Lambda (daily run, retrain steps, API)": lam_daily + lam_retrain + lam_api,
        "SageMaker training (managed spot)": training,
        "Step Functions": sfn,
        "S3 (~2 GB with versions)": 2 * P["s3_gb"],
        "DynamoDB on-demand": DAYS * 60 * P["ddb_wru"] + (DAYS * 200 + api_requests) * P["ddb_rru"],
        "CloudWatch (3 custom metrics, 4 alarms, logs)": 3 * P["cw_metric"] + 4 * P["cw_alarm"] + 0.1 * P["cw_logs_gb"],
        "ECR (two images)": 1.2 * P["ecr_gb"],
    }


def alternatives() -> dict[str, float]:
    hour = 730
    return {
        "Bijuli (serverless, batch, spot training)": sum(bijuli().values()),
        "+ SageMaker real-time endpoint for the forecast (ml.m5.large, 24/7)": sum(bijuli().values()) + hour * 0.134,
        "+ SageMaker endpoint with Model Monitor data capture and hourly schedule":
            sum(bijuli().values()) + hour * 0.134 + 24 * DAYS * (5 / 60) * 0.134,
        "One t4g.medium EC2 running cron + MLflow (24/7)": hour * 0.0224 + 30 * 0.0912 + 1.0,
        "Daily retrain instead of drift-triggered (30 spot jobs)": sum(bijuli(retrains_per_month=30).values()),
    }


if __name__ == "__main__":
    items = bijuli()
    for k, v in items.items():
        print(f"{k:55s} ${v:7.2f}")
    print(f"{'Total per month':55s} ${sum(items.values()):7.2f}\n")
    for k, v in alternatives().items():
        print(f"{k:75s} ${v:8.2f}")
