# What breaks, and what happens when it does

The question I designed around: *at 10:00 on a winter morning, the desk needs tomorrow's 24 numbers by 12:15. What has to go right, and what happens when each piece doesn't?*

The replay (`python -m bijuli replay`) runs 26 months of daily operations, from July 2023 to August 2025, with real faults injected into the feed. These are the faults it injects, and what the system did about each one. All numbers come from `artifacts/reports/replay_*.csv`.

## Measured in the replay

### 1. The SCADA export switches to kW for three days (5-7 February 2024)

Every value is 1,000 times too large. This kind of fault is common: a units change in an export template.

| | With data contracts (Bijuli) | Same policy, contracts off |
|---|---:|---:|
| Forecast MAPE on the three affected days | 4.6%, 0.8%, 2.7% (fallback anchor, flagged `degraded`) | **91,878%, 91,865%, 88,617%** |
| Median daily MAPE over 26 months | 2.63% | 2.66% |
| Mean daily MAPE over 26 months | 3.39% | 352% |
| Trainings | 61 | 86 |

**What happened without contracts:** this went further than three bad forecasts.

- **The monitor was poisoned for 13 months.** One day with a bias of +91,000% pushed the bias CUSUM to about 2,700. It decays by only 0.02 a day, so it stayed in alarm until a promotion reset it in March 2025.
- **Retraining ran almost every week.** The policy kept retraining "for drift", and the offline gate rejected nearly all of those candidates.
- **The 1,000x values were in the training window.** One candidate that reached shadow scored 85.4% MAPE (v009). Shadow rejected it.

**What happened with contracts:** the unit check (the median ratio to the last 4 weeks is above 50) quarantined the three days at ingest. The forecasts that needed them fell back to last week's same hour, flagged `degraded`.

**Lesson, and a fix I would make next:** the median hides a disaster like this; look at the mean and the worst week. Also, the CUSUM should clip its input (for example to ±50% a day), so that one impossible day cannot hold the monitor in alarm for a year. Contracts are the first line of defence. Clipping would be the second.

### 2. The SCADA feed goes dark for three days (10-12 June 2024)

The ingest finds no file. `features.load_matrix` keeps those days as NaN rows (bug #10). The forecast then falls back from "yesterday's same hour" to "last week's same hour" as its anchor, and flags the forecast `degraded`.

| Forecasts that depended on the missing days | MAPE |
|---|---:|
| Day 1 | 6.0% |
| Day 2 | 10.6% |
| Day 3 | 2.9% |
| Normal June days around them | 3 to 9% (monsoon onset) |

The system never refused a forecast in the replay. It refuses only when last week is missing too. That case is tested in `test_forecast_falls_back_then_refuses_when_load_data_is_gone`, and it emits `ForecastRefused`, which is emailed. **Refusing beats inventing a number** for a bid worth tens of millions of rupees.

### 3. A bad release (first retrain on or after 15 August 2024)

The retraining job reads weather from a new provider that defaults to Fahrenheit, while the forecast job still reads Celsius. Offline, the model is indistinguishable from a clean one. Live, its MAPE is 4.36% against 3.61% (bug #13).

| Policy | Days the bad model bought power |
|---|---:|
| Retrain every 4 weeks, auto-deploy | **28** |
| Every 4 weeks, gated (offline gate + shadow) | 0 (shadow rejected it: 2.9% vs 2.4%) |
| Drift only, gated | 0 (shadow rejected it) |
| Bijuli (4-weekly or bias drift, gated + rollback) | 5 (passed a noisy shadow week, then rolled back: 5.0% vs 2.0%) |

### 4. New industrial feeders connect (+60 MW on 20 November 2024, +45 MW on 10 April 2025)

The step shows up as one bad day, then the model is back to normal, with no retraining needed. The model predicts the ratio to yesterday, so once yesterday includes the new feeder, so does the forecast. In the replay, the forecast covering the April step was 15.2% off. The next day's was 2.5%.

### 5. EV charging grows from 6 to 48 MW over the period

This is slow concept drift. It produces no single bad day, just a bias that creeps. A model that is never retrained drifts from -0.6% bias to -2.6% over two years (`scripts/drift_analysis.py`). The bias CUSUM exists for exactly this. In the Bijuli policy it fired 13 times in 26 months: 6 of those candidates reached shadow, and the offline gate stopped 7.

### 6. Five Terai weather stations go silent for 130 days (real, November 2023 to April 2024)

This gap is real, in NOAA's data. During it, Terai afternoon temperature is estimated from Kathmandu, and every estimated day is flagged (`terai_imputed`). The backtest covers this window. Its winter error stays in line with the other winters, because winter demand is driven by Kathmandu heating and darkness, not by Terai heat.

## Every failure mode I could think of

| What fails | How it shows up | How it's detected | What the system does | Residual risk |
|---|---|---|---|---|
| SCADA export late or missing | No `raw/scada/date=D` file at 10:00 | Ingest returns `missing` | Forecast uses last week's same hour as anchor, flagged `degraded` | Error roughly doubles on those days (measured above) |
| SCADA in wrong units | Values 1,000x too large or small | Unit contract: median ratio to the last 4 weeks above 50 or below 0.02 | Whole day quarantined to NaN, then as above; `QuarantinedHours` metric | Unit changes smaller than 50x (e.g. MW vs 10 MW blocks) get through |
| Stuck meter | Same value 4 or more hours in a row | Flatline contract | Those hours quarantined | A meter stuck at a plausible, slowly changing value |
| Single-hour spike | A jump of more than 35% that reverses the next hour | Spike contract | Hour quarantined | A real sudden event (a big trip) is also removed. It is rare at national level, and the hour returns as truth the next day |
| Load-shedding or a grid trip | Demand drops because supply failed, not because people used less | Not detected automatically | The model learns it as demand | NEA's daily report lists interruptions. Joining it in as a mask is listed as next work |
| Weather feed down | No `fc_` columns for tomorrow | Feature is NaN | LightGBM handles NaN; the forecast still runs | Hot or cold surprises are missed |
| Weather feed changes units (bad release) | Training and serving disagree | Not offline. Shadow on the serving path, then the rollback watch | Rejected in shadow, or rolled back after 5 days (measured above) | 5 days of worse forecasts in the unlucky case |
| Retrain produces a worse model | Candidate's held-out MAPE is worse than the champion's live MAPE | Offline gate | Rejected; SNS email | The champion gets staler. The 4-weekly schedule tries again |
| Promoted model is worse live | 5-day MAPE more than 15% above the previous champion's | Rollback watch (the previous champion keeps forecasting in shadow) | One write to `/bijuli/champion`, then an email | Up to 5 days on the worse model |
| Slow drift (EVs, growth) | Bias creeps, MAPE barely moves | Two-sided CUSUM on daily bias, plus the `BiasAlarm` | Retrain requested; normal gated rollout | 1 to 2 weeks of a 1 to 2% bias |
| One impossible day poisons the CUSUM | Monitor stays in alarm for months | Not detected (found in the contracts-off replay) | Contracts stop it reaching the monitor | Fix: clip the CUSUM input. Not built yet |
| PSI says everything drifted | PSI above 0.25 almost every 4-week window | n/a | PSI is reported, never a trigger (bug #14) | A real covariate shift is spotted late. Truth arrives daily, so the bias catches it |
| Daily run silently not scheduled | Nothing runs, **so nothing errors** | `NoScoreAlarm`: no `DailyMAPE` datapoint for 2 days (`TreatMissingData: breaching`) | Email | The failure an error-only alarm misses |
| Daily run fails | Step Functions `ExecutionsFailed` | `DailyRunFailedAlarm` | Retries (2) on the scheduler; failures go to the DLQ, and `FailureQueueAlarm` fires | The desk falls back to its manual forecast for that day |
| Spot capacity unavailable for training | Training job waits | `MaxWaitTimeInSeconds` on the job | Job fails; the retrain state machine emails; the champion keeps serving | A retrain is delayed by a day. Nothing user-facing |
| Two retrains overlap (schedule + drift on the same day) | Two candidates race for the shadow slot | Not prevented. Both are gated, and the second overwrites `/bijuli/shadow` | The later candidate is the one shadowed | Wasted training job ($0.01) |
| Alert topic encrypted with the AWS-managed key | EventBridge cannot publish, **so alerts silently fail** | Found in design review (bug #18) | Topic left unencrypted; messages hold no secrets | None now |
| Cost runaway | Bill grows | AWS Budget at 80% of $15 | Email | Budget data lags by hours |
| ap-south-1 outage | Everything down | AWS Health | Nothing: single region by choice | The desk still has its own forecast. A second region would double the moving parts for one day a year |

## Known gaps (not built, deliberately listed)

1. **CUSUM input clipping** (above): about two lines of code, but the replay numbers would need to be regenerated.
2. **A load-shedding mask** from NEA's daily interruption report, so the model doesn't learn supply failures as low demand.
3. **A weather ingest Lambda** for NOAA GFS. The backtest uses observed weather plus realistic forecast error instead (`features.weather_forecast`).
4. **A manual approval step** before promotion. The desk might want one. SageMaker Model Registry's `PendingManualApproval` would be the place for it.
