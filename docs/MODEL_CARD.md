# Model card: Bijuli day-ahead demand forecaster

**What it predicts:** Nepal's national hourly electricity demand (MW, demand requirement including load that was interrupted) for each hour of tomorrow, issued at 10:00 Nepal time: p10, p50, p90 and a purchase basis at the 0.62 quantile.

**Model:** three LightGBM quantile regressors on `log(demand / yesterday's same hour)`, trained on a rolling 365 days, with a conformal margin on the p10-p90 band estimated from the last 6 weeks of the window.

**Inputs:** SCADA load up to 09:00 today, observed Kathmandu weather to yesterday, a weather forecast for tomorrow (Kathmandu mean and minimum temperature, Terai apparent temperature), calendar (weekday, Saturday, day of year, days to Dashami and Laxmi Puja, sunset).

## Intended use
A first draft for the Load Dispatch Centre's day-ahead purchase and offer plan. A dispatcher reviews it; it does not place bids.

## Not intended for
- Real-time dispatch or frequency control.
- Provincial or substation load (it only knows the national total).
- Any decision after a structural break it has not seen (a new cross-border line, a blackout, a festival not in the calendar table) without a human check.

## Performance (simulated demand, walk-forward, Jan 2023 - Aug 2025)
See README "Results". In short: 3.3% hourly MAPE against 4.7% for the best naive rule, 79% of outcomes inside the nominal 80% band.

## Limitations, honestly
- **Trained and evaluated on a simulated load series.** It is calibrated to NEA's published FY 2081/82 figures and checked against 40 real days, but the hour-by-hour shape is the twin's. Performance on real SCADA data will differ; how much is unknown until it is run on it.
- The weather forecast input is simulated (observed weather plus noise). Real NWP errors are larger on monsoon-onset days and cold waves.
- The festival table is hand-maintained and covers 2022-2025. A missing year means Dashain is forecast as normal days: up to ~20% over-forecast for a week.
- It under-forecasts slightly on average (about 1% in the backtest), because demand keeps growing and every training window is a little behind.

## Who is affected by errors
Under-forecasts in the dry season mean NEA buys less day-ahead than it needs; the gap is covered at real-time prices or by cutting supply, and supply cuts in Nepal fall first on industrial feeders. Over-forecasts cost NEA the loss on resold power. The purchase basis is set above the median for exactly this asymmetry.
