# Challenges and bugs

What went wrong while building this, roughly in the order I hit it, with the evidence and the fix. Every number can be reproduced from the repo.

## Data

### 1. The data I wanted was not reachable

**Problem:** NEA publishes an hourly national load profile (BS 2073, on Open Data Nepal) and a daily operational report (PDF). From the environment I built this in, both sites were blocked, and no mirror of any hourly Nepal load series existed.

**Choice:** a grid twin driven by real weather and calendar, calibrated to what NEA *does* publish, validated against 40 real daily reports it was not fitted to, and labelled as simulated everywhere ([ADR-002](ALTERNATIVES.md#adr-002-real-hourly-load-vs-a-calibrated-twin)). The loader is one function, so real SCADA data replaces it without touching the rest.

**Lesson:** say it on the first screen. A reviewer who discovers simulated data on their own stops trusting everything else.

### 2. Nepal is UTC+05:45, and the observations land at quarter to the hour

**Symptom:** the first Kathmandu temperature curve peaked at 08:00 UTC, which looked plausible, and 13:45 local, which also looked plausible.

**Cause:** ISD timestamps are UTC. Nepal's offset is 5 hours 45 minutes, not 5 or 6, so every observation lands at hh:45 local time. Rounding down put the 13:45 reading in the 13:00 bucket, describing a temperature 45 minutes before it was measured.

**Fix:** convert to Asia/Kathmandu, then assign hh:45 observations to the *next* clock hour. **Test:** `test_isd_lite_is_converted_to_nepal_time_and_missing_codes_dropped`. The same offset came back later as the planted bad release (#13).

### 3. The Terai stations only report in daylight

**Symptom:** a "daily mean temperature" for Biratnagar 3 to 5 °C above Kathmandu's in winter, more than climatology says.

**Cause:** outside Kathmandu, the stations send 3-hourly synops between 00 and 12 UTC only: 05:45 to 17:45 local. A mean of those is a daytime mean.

**Fix:** use the 09 UTC (14:45 local) observation as the afternoon temperature, close to the daily maximum, which is what drives fans and air conditioners anyway.

### 4. Five stations went silent on the same day

**Symptom:** completeness for every non-Kathmandu station drops to ~55% in 2024.

**Cause:** all five stop reporting together from 26 November 2023 to 3 April 2024 (130 days), and restart together. Five thermometers do not fail on the same day: this is a feed problem upstream.

**Fix:** estimate Terai afternoon temperature from Kathmandu with a per-month linear fit learned on days when both exist, flag every estimated day (`terai_imputed`), and raise a weather finding so the forecast says it is running on estimates. **Test:** `test_real_terai_gap_is_imputed_and_flagged`.

### 5. The first twin had evening peaks 8.7% too high

**Symptom:** against the 40 real NEA days, daily energy was within 2.8% but the daily peak was 8.7% high, every day.

**Cause:** I had guessed the height of the evening peak. NEA's monthly table says the peak is 1.42x the monthly mean in Poush but only 1.16x in Ashadh.

**Fix:** set the peak height against NEA's *monthly* peak table (calibration data), which brought the 40-day peak error to 2.4%. **Honest caveat:** I had already seen the 40-day result when I made that change, so the peak validation is not fully independent; the energy validation (which did not change) is.

### 6. The twin still over-peaks in the monsoon

Ashadh's twin peak is 13% above NEA's: the cooling term raises the evening in the humid months more than reality does. It is documented rather than tuned away, because the 40 validation days are in winter and there is nothing to check a monsoon fix against.

## Modelling

### 7. The 80% band covered 37% of outcomes

**Symptom:** first backtest: hourly MAPE 3.2%, excellent; p10-p90 coverage 36.6%.

**Cause:** LightGBM quantile models fit their training data closely, so the quantiles describe in-sample residuals, which are much smaller than tomorrow's.

**Fix:** conformalised quantile regression. Fit on all but the last 6 weeks, measure how far those weeks fall outside the band, widen by that margin, refit on everything. Coverage went to 79%. The held-out MAPE from those 6 weeks also became the model's "promise" for monitoring (#11).

### 8. My first cost figure was NPR 13.5 million a day

**Cause:** I priced every MW of error at market price. NEA does not: it re-dispatches its storage plants (Kulekhani) to absorb small misses, and only the rest is settled.

**Fix:** only the part of each hourly miss above 50 MW is priced. The relative ranking of methods did not change; the absolute number became believable. The assumptions are in `config.py` and the README says the rupee figures are only as good as them.

### 9. Predicting the level would have failed on a growing grid

Demand grows every month, and a tree can only predict values between the ones it saw. The model predicts `log(demand / yesterday's same hour)`. The drift analysis shows what this buys: when a 60 MW feeder connects, error spikes on that one day (13.5% when the 45 MW feeder connected) and is back to normal the next day, with no retraining. It also shows what it does not buy: the average bias of a never-retrained model still creeps from -0.6% to -2.6% over two years.

## MLOps

### 10. Days with no data disappeared instead of showing up as missing

**Symptom:** the first replay "refused" 7 forecasts for days that had perfectly good fallback data.

**Cause:** `pivot_table` drops rows that are entirely NaN. A three-day SCADA outage removed three rows from the day x hour matrix, so the features for the following days could not find "same hour last week" either.

**Fix:** reindex the matrix to every calendar day. **Test:** `test_outage_days_stay_visible_as_missing_rows`.

### 11. The drift trigger fired every week

**Cause:** the trigger compared the live 7-day MAPE with the model's *in-sample* error on its last two training weeks. In-sample error is always lower than live error, so the model looked degraded from day one.

**Fix:** compare with the conformal held-out MAPE (#7), which is out of sample by construction.

### 12. Scoring the wrong forecast after a promotion

**Symptom:** the auto-deploy policy showed 61 "refused" days.

**Cause:** each morning I scored *the current champion's* forecast for yesterday. On the day after a promotion, the new champion had not made yesterday's forecast; the old one had, and it was the one that bought power.

**Fix:** record which version's forecast was served for each day and score that one.

### 13. Two of my three planted bad releases were harmless

To test the rollout, the replay ships one bad model per policy: the first model trained on or after 15 August 2024 has a training-serving skew. I tried three bugs before I found one worth testing against.

1. **Hour features from UTC instead of Nepal time** (the #2 bug). Expected: a disaster. Measured: 2.78% MAPE against 2.85% for the clean model. The model uses yesterday's same hour as its anchor, so a constant shift in the `hour` feature is mostly absorbed.
2. **A leaky "this morning vs yesterday morning" ratio** built from hours that are not known at 10:00. Measured: 3.42% live against 3.61% clean. The leaked hours correlate 0.98 with the hours that *are* known, so the leak carried almost no extra information.
3. **The weather feed switched to Fahrenheit for the training data only.** This one hurts: 4.36% live against 3.61% clean, +21%, and **offline evaluation is identical to the clean model**, because training and gate evaluation both see Fahrenheit. Only serving sees Celsius.

Bug 3 is the one in the replay (`ops.fahrenheit_release`). In the gated policies the 7-day shadow rejected it (2.9% against 2.4% for the champion). In the hybrid policy the shadow week happened to be noisy, the bad model passed (3.0% against 3.3%), was promoted on 17 September 2024, and the 5-day rollback watch removed it on 22 September (5.0% against 2.0%). Defense in depth, measured.

On the way, my own offline gate "caught" the UTC bug with a MAPE of 19.1%. It could not have: the gate evaluates through the training pipeline, where the bug is consistent. The 19.1% came from my gate code looking up the truth by the buggy hour. Fixed: truth is looked up by the real hour.

**Lessons:** when a test catches something it shouldn't be able to see, the test is the bug. And a planted failure must be checked to actually fail; two of mine did not.

### 14. PSI said everything was drifting, all the time

**Symptom:** every monitored feature had a PSI above 0.25 in every 4-week window, including the scale-free ones, and also when compared with the same weeks one year earlier.

**Cause:** two things. Daily features repeat 24 times a day, so per-hour counts fake a large sample. And with one value per day, a 28-day window has 28 points for 10 bins: with **no drift at all**, a 28-point sample exceeds 0.25 in 71% of draws (3% at 90 points, 0% at 365). Weather and the demand level genuinely move between windows on top of that.

**Decision:** PSI is reported, not used as a trigger. With truth arriving every morning, the model's own bias is a direct measurement of what PSI would only hint at ([ADR-004](ALTERNATIVES.md#adr-004-what-triggers-retraining)).

### 15. The replay took 40 CPU-minutes for one policy

**Cause:** profiling showed 96% of the time in LightGBM boosting, from a trigger that fired too often (#11) and a gate that trained a full conformal model just to compare.

**Fix:** the gate copy skips the conformal step, trees went from 400 at 0.05 to 250 at 0.08 (same accuracy within noise), and the trigger fix removed most of the trainings. A three-month replay went from 76 s to 12 s.

### 16. The forecast Lambda crashed on an empty feed

A moto test that deletes all load data found `forecast` crashing inside pandas on an empty series instead of refusing. It now emits `ForecastRefused` and returns. **Test:** `test_forecast_falls_back_then_refuses_when_load_data_is_gone`.

### 17. The flatline detector undercounted

The test planted a 6-hour stuck reading; the detector quarantined 4. The run-length logic marked hours from the fourth repeat onward. Rewritten as "group runs of equal values, quarantine runs of 4 or more". **Test:** `test_flatline_and_spike_are_caught`.

### 18. Encrypting the alert topic would have silenced the alerts

Found in design review, not by deploying. I had encrypted the SNS alert topic with the AWS-managed key (`alias/aws/sns`), and later added an EventBridge rule that sends refusals, promotions and rollbacks to that topic. EventBridge cannot use the AWS-managed SNS key, so those publishes would fail, and a failed alert does not alert anyone. The choices were a customer-managed KMS key with a policy for EventBridge ($1 a month, a 60% increase on this system's bill) or no encryption on a topic whose messages are forecast summaries and model version names. I chose no encryption and wrote the reason next to the resource.

**Lesson:** a security default can break reliability; check what each service can actually use before ticking the box.
