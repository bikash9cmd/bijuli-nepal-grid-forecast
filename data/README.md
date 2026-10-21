# Data

## What is real and what is simulated

| Input | Status | Source |
|---|---|---|
| Hourly weather, 6 airport stations, Jan 2022 - Aug 2025 | **Real** | NOAA Integrated Surface Database (ISD-Lite), AWS Open Data bucket `noaa-isd-pds`, files pinned by SHA-256 in `bijuli/config.py` |
| NEA FY 2081/82 monthly national energy demand and peak demand | **Real** (published) | NEA Annual Report FY 2024/25, monthly energy and capacity balance tables ([data/nea/monthly_fy2081_82.csv](nea/monthly_fy2081_82.csv)) |
| 40 NEA daily operational reports, 3 Dec 2024 - 11 Jan 2025 | **Real** (published) | NEA "Nepal Daily Operational Report" PDFs ([data/nea/daily_reports_2024-12_2025-01.csv](nea/daily_reports_2024-12_2025-01.csv)), used only to validate the twin |
| Festival dates (Dashain, Tihar) | **Real** | Nepal government calendar |
| Hourly national demand, 2022 - 2025 | **Simulated** | The grid twin, `bijuli/twin.py`: driven by the real weather and calendar, calibrated to the real NEA tables |
| Day-ahead weather forecast | **Simulated** | Observed weather plus autocorrelated error sized to typical day-ahead temperature error (`features.weather_forecast`) |
| EV charging growth, two new industrial feeders, faults in the SCADA feed | **Scenario** | Dated drift and fault events, so detection can be measured |

The NEA tables were transcribed from the PDFs by the open [NepalHydroAndEnergy](https://github.com/abinashkarki/NepalHydroAndEnergy) project (MIT licence); only the columns used here are kept.

## Why the load is simulated

NEA does publish hourly load: Open Data Nepal hosts NEA's hourly load profile for BS 2073 (2016-17), and NEA posts a daily operational report as a PDF. Neither site could be reached from the environment this project was built in, and no mirror of an hourly Nepal load series was found. Rather than borrow another country's load and call it Nepal's, the twin reproduces the numbers NEA publishes and is checked against 40 real days it was never fitted to (README, "The grid twin").

**To use real data:** anything that returns an hourly `pandas.Series` of MW indexed by Nepal time can replace `twin.generate(...)["demand_mw"]`. The contracts, features, models, registry, monitor and replay do not change.

## Weather data quality (real, from `python -m bijuli quality`)

| Station | 2022 | 2023 | 2024 | 2025 (to 24 Aug) |
|---|---:|---:|---:|---:|
| Kathmandu (hourly) | 96.9% | 98.4% | 95.3% | 98.4% |
| Biratnagar (afternoon synop) | 90.1% | 77.0% | 56.0% | 85.5% |
| Simara | 91.8% | 78.4% | 56.6% | 85.1% |
| Bhairahawa | 88.5% | 76.7% | 56.6% | 85.5% |
| Nepalgunj | 87.7% | 74.8% | 54.4% | 85.1% |
| Pokhara | 90.4% | 76.4% | 54.4% | 82.6% |

All five non-Kathmandu stations go silent together from 26 November 2023 to 3 April 2024 (130 days): a feed problem, not five broken thermometers. See docs/CHALLENGES_AND_BUGS.md.

`data/raw/` is not committed; `python -m bijuli download` fetches and verifies it.
