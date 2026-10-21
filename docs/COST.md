# Cost

## Monthly AWS estimate (ap-south-1, from `python scripts/cost_model.py`)

Approximate on-demand list prices, no free tier (so this is a second-year account). Check current prices before relying on them.

| Item | Per month |
|---|---:|
| Lambda (daily run, retrain steps, API at 20k requests) | $0.07 |
| SageMaker training (2 jobs, ~6 billable minutes each, managed spot) | $0.01 |
| Step Functions (standard) | $0.01 |
| S3 (~2 GB including old versions) | $0.05 |
| DynamoDB on-demand | $0.01 |
| CloudWatch (3 custom metrics, 4 alarms, logs) | $1.37 |
| ECR (two images) | $0.12 |
| **Total** | **~$1.63** |

**The biggest line is monitoring, not ML.** Three custom metrics and four alarms cost twenty times more than training and inference together. That is the right way round: the alarms are what notice a pipeline that silently stopped.

## Against the alternatives

| Design | Per month |
|---|---:|
| **Bijuli** (serverless, batch, spot training, drift + schedule) | **$1.63** |
| Same, retraining every day instead | $1.77 |
| One t4g.medium EC2 with cron and MLflow, 24/7 | ~$20 |
| + a SageMaker real-time endpoint (ml.m5.large) for the forecast | ~$99 |
| + Model Monitor data capture and an hourly monitoring schedule | ~$107 |

Two things this table says that I did not expect before building it:

1. **Retraining frequency is not a cost decision.** Retraining every day on spot costs $0.14 a month more than retraining twice. The reason not to retrain daily is risk: every promotion is a chance to put a bad model in charge of a purchase plan (README, "What retraining policy?").
2. **An endpoint would be ~60x the whole system** to answer one request a day.

## The cost that actually matters: forecast error

Using stated assumptions (day-ahead price ~INR 5.5/kWh, a shortfall 1.6x as expensive as a surplus, NEA absorbing the first 50 MW of any miss with its own storage plants; `bijuli/config.py`), forecast error in the backtest costs roughly NPR 3.9 million a day with Bijuli against NPR 9.3 million for "same hour yesterday". The absolute rupee figures are only as good as those assumptions; the ratio between methods is the useful part. Either way, AWS is a rounding error next to one bad forecast day.

## Cost decisions in the design

| Decision | Saves |
|---|---|
| Batch inference in Lambda, no endpoint | ~$98/month |
| Managed spot for training | ~70% of training cost (and SageMaker checkpoints are unnecessary: a 1-minute job just restarts) |
| Step Functions + EventBridge Scheduler instead of MWAA | ~$350/month |
| S3 + SSM registry instead of a managed MLflow tracking server | the tracking server's hourly charge |
| arm64 (Graviton) Lambdas | ~20% per GB-second |
| One container image for all functions | one ECR image to store and scan |
| On-demand DynamoDB | nothing provisioned for a table written 2x a day |
| S3 lifecycle: raw to IA at 30 days, Glacier IR at 180; training snapshots expire at 120 days | storage grows slowly instead of linearly |
| Model cached in the Lambda container | one S3 read per cold start, not per call |
| Gate before training the real candidate is cheap (no conformal step) | half the compute of a rejected retrain |
| Budget alarm at 80% of $15 | catches a runaway loop (a retrain trigger that fires every hour) |

## What would change the picture

- **Per-substation forecasts** (~200 series): one global model still trains in minutes on one instance. Lambda costs scale linearly and stay under $5.
- **SMS to dispatchers instead of email:** each message costs more than the whole daily run.
- **Real-time market bidding** (15-minute products): inference every 15 minutes is still ~3,000 Lambda calls a month, a few cents.
