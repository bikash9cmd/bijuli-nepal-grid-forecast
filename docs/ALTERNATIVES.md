# Alternatives considered

Each decision as a short record: the options, what I chose, why, and what would make me change my mind.

## ADR-001: Which problem

**Options:** hydropower generation forecasting per plant; national load forecasting for the day-ahead market; outage prediction on distribution feeders; tariff/theft analytics.

**Chosen:** day-ahead national demand forecasting for NEA's cross-border trading desk.

**Why:** It is a decision someone makes every day with real money: since Nepal started trading on India's day-ahead market, every evening shortfall in the dry season is bought across the border, and every wet-season surplus is offered. Truth arrives the next day, so the system can measure itself daily, which is what makes it a good MLOps problem rather than a one-off model. Plant-level generation needs river-flow data that is not open; feeder outages need NEA's distribution data, which is not open either.

**Would change if:** NEA published plant-level dispatch or hydrology. A run-of-river generation forecast is the other half of the bid and the natural next project.

## ADR-002: Real hourly load vs a calibrated twin

**Options:** (a) wait for real hourly SCADA data; (b) use another country's open load data; (c) a twin calibrated to NEA's published numbers, driven by real Nepal weather.

**Chosen:** (c), labelled as simulated everywhere.

**Why:** NEA's 2016-17 hourly load profile on Open Data Nepal and its daily operational reports could not be reached from the build environment. Another country's load would make the project a clone with a Nepali name. The twin reproduces what NEA *does* publish: FY 2081/82 monthly energy (fitted) and monthly peaks (used to set the evening-peak height), and it matches 40 real daily reports it was not fitted to within 2.8% on energy and 2.4% on peak. For an MLOps project the twin has one real advantage: drift events (EV growth, two new industrial feeders) have known dates, so detection lag is measurable instead of argued.

**Would change if:** real SCADA data is available. The rest of the pipeline only needs a `time, demand_mw` series; swapping the source is a loader change.

## ADR-003: Model

**Options:** naive baselines; SARIMAX; Prophet; LightGBM; a neural network (N-BEATS, TFT, LSTM).

**Chosen:** LightGBM quantile regression on `log(demand / yesterday's same hour)`, with conformal calibration of the p10-p90 band.

**Why:** Three years of hourly data is ~30,000 rows: plenty for gradient boosting, thin for a deep sequence model, and the evening-peak/festival/weather interactions are what trees do well. Predicting the ratio to yesterday instead of the level keeps the model valid as demand grows (a tree cannot predict a level above anything it saw in training). Quantile models give the purchase basis directly; raw LightGBM quantiles were badly over-confident (37% coverage for a nominal 80%), so a conformal margin from the last 6 weeks of each training window fixes the band.

**Would change if:** the problem became multi-region with shared patterns (a global deep model starts to pay off), or if NWP ensembles became an input (probabilistic weather would want a model that consumes distributions).

## ADR-004: What triggers retraining

**Options:** never; fixed schedule (daily, weekly, 4-weekly); performance trigger; data-drift (PSI) trigger; concept-drift (bias CUSUM) trigger.

**Chosen:** retrain every 4 weeks, plus an early retrain when the CUSUM on daily bias crosses its threshold; every candidate goes through the gated rollout (ADR-005). PSI is reported but does not trigger.

**Why:** Measured in the 26-month replay (README, "What retraining policy?"), and the result was not what I expected:

| Policy | MAPE | Worst week | Trainings | Days on the bad release |
|---|---:|---:|---:|---:|
| 4-weekly, auto-deploy | 3.42% | 9.8% | 28 | 28 |
| 4-weekly, gated | 3.36% | 7.6% | 42 | 0 |
| Drift only, gated | 3.43% | 8.0% | 50 | 0 |
| 4-weekly or bias drift, gated | 3.39% | 7.45% | 61 | 5 |

The gates matter. The trigger barely does: every gated policy is within 0.07 percentage points of the others, which is inside the noise of one replay. Drift triggers did not beat a calendar; they bought more trainings. I keep the CUSUM because it gives the best worst week and reacts within days to the kind of change the replay injected (feeder steps, EV growth), but I would not claim the replay proves it is needed.

Compute cost is irrelevant: retraining daily on spot would cost about $0.14 a month more. The real cost of retraining is *change*: every promotion is a chance to ship a bad model. The auto-deployed schedule spent 28 days buying power on one; the hybrid, which trains most often, was the only gated policy whose shadow let the bad release through (on a noisy week), and the rollback watch removed it after 5 days. More trainings means more rolls of that dice.

Bias is the signal that matters for a growing grid, because a model that is 2% low every evening looks fine on MAPE for weeks. PSI fires on almost every 28-day window even without drift (CHALLENGES_AND_BUGS #14), so it is information for a human, not a trigger.

**Would change if:** labels arrived late (e.g. monthly metering): then data drift would be the only early signal and PSI on scale-free features over longer windows would have to trigger. If a second replay with a different seed showed the CUSUM adds nothing, I would drop it and keep the plain 4-weekly gated schedule.

## ADR-005: How a new model is rolled out

**Options:** replace immediately; A/B or canary split; shadow then promote; manual approval in the model registry.

**Chosen:** offline gate, then 7 days of shadow on the serving path, then automatic promotion with a 5-day reverse-shadow rollback watch.

**Why:** There is one consumer and one forecast a day: no traffic to split, so canary is meaningless (ARCHITECTURE.md). The offline gate catches models that are simply worse and costs one extra training. Shadow catches what offline evaluation cannot: training-serving skew. Rollback is one SSM write. Manual approval would be right if the desk wanted it, and the SageMaker Model Registry's `PendingManualApproval` status is the place to add it.

## ADR-006: Where models are registered

**Options:** SageMaker Model Registry; MLflow on EC2 or SageMaker managed MLflow; an S3 + SSM registry.

**Chosen:** S3 model cards + an SSM champion pointer (`bijuli/registry.py` locally).

**Why:** One model family, one team, a handful of versions a year. The registry needs four things: versioned artifacts, lineage (data hash, training window), status history, and an atomic "which version is live". S3 versioning plus one SSM parameter gives all four for free. SageMaker Model Registry is the upgrade path and maps one-to-one (model package group, approval status); managed MLflow has an hourly tracking-server charge that is out of proportion here.

**Would change if:** several teams or model families, or a compliance requirement for approval workflows.

## ADR-007: Where training and inference run

**Options:** SageMaker real-time endpoint; SageMaker batch transform; Lambda; ECS Fargate; one EC2 instance with cron.

**Chosen:** inference in Lambda (container image), training as a SageMaker training job on managed spot.

**Why:** Inference is 24 rows once a day: a real-time endpoint would be ~$98 a month idle for one call a day, and batch transform spins up an instance for a sub-second job. Training runs a couple of times a month for a few minutes; SageMaker training jobs give spot pricing, a clean container boundary, logs and metrics without managing anything, and keep the door open to bigger models. Lambda could also train (it fits in under a minute today), but it would put the 15-minute limit between me and the next model.

**Would change if:** many consumers needed sub-second custom forecasts (endpoint), or training outgrew one instance (distributed training job).

## ADR-008: Orchestration

**Options:** EventBridge rule straight to Lambdas; Step Functions; SageMaker Pipelines; Amazon MWAA (Airflow).

**Chosen:** two Step Functions state machines (daily, retrain) started by EventBridge Scheduler (in the Asia/Kathmandu time zone) and an EventBridge rule.

**Why:** Retries with backoff, a failure branch that alerts, a native `createTrainingJob.sync` integration, and a visual history of every run, for cents. SageMaker Pipelines would cover only the retrain half. MWAA's smallest environment is roughly $350 a month.

## ADR-009: Monitoring

**Options:** SageMaker Model Monitor; a custom daily scoring Lambda; Evidently/WhyLabs.

**Chosen:** custom: daily score, CUSUM, PSI, CloudWatch metrics and alarms.

**Why:** Model Monitor is built around endpoint data capture and is strongest on data quality and drift of request payloads; Bijuli's most important signal is ground-truth performance the next morning, which is a 40-line function. The alarms that matter are on *absence*: "no forecast scored for two days" catches a pipeline that silently stopped, which an error alarm never sees.

## ADR-010: Infrastructure as code

**Options:** console; CloudFormation; SAM; CDK; Terraform.

**Chosen:** SAM. Serverless functions, state machines and schedules are first-class, it is plain YAML that `cfn-lint` checks in CI, and `sam build` builds the container image.
