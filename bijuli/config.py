"""Paths, pinned sources and the operating assumptions every module shares."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
NEA_DIR = DATA / "nea"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ARTIFACTS / "reports"
REGISTRY_DIR = ARTIFACTS / "registry"
DOCS_IMG = ROOT / "docs" / "img"

TZ = "Asia/Kathmandu"  # UTC+05:45, no daylight saving

# ---------------------------------------------------------------------------
# Weather: NOAA Integrated Surface Database (ISD-Lite) on the AWS Open Data registry.
# Anonymous reads; every file is pinned by SHA-256 so a silent upstream re-issue is caught.
# ---------------------------------------------------------------------------
ISD_BUCKET = "noaa-isd-pds"
ISD_REGION = "us-east-1"

STATIONS = {
    # usaf: (name, role)
    "444540": ("Kathmandu (Tribhuvan Intl)", "valley"),  # hourly; heating load
    "444780": ("Biratnagar", "terai"),  # 3-hourly daytime synops only
    "444490": ("Simara", "terai"),
    "444380": ("Bhairahawa", "terai"),
    "444180": ("Nepalgunj", "terai"),
    "444340": ("Pokhara", "hill"),
}
YEARS = (2022, 2023, 2024, 2025)

ISD_SHA256 = {
    "444540-2022": "2d3445389eb10ba6b596ea846737fee9a00417ab8c93e059cce673e2ba13c5cb",
    "444540-2023": "ee6a15369fbb83f459811e8c1a548f7b6661436cfbdbfa1a4a2dbe6884565e86",
    "444540-2024": "ee0e4683d153456ef4049e1c020f9fdcbafc2aaff81f730676eea6aae0cfc43e",
    "444540-2025": "a1ae3705418ac5071caad383226397a02a94a976dea162ebe572c997532ab2c7",
    "444780-2022": "b105b84304f0c9242d50cf87ab26aa784e6ffcbf343569175fb70cd525b36dd4",
    "444780-2023": "e794399a624b82d3110906250a9df0dc2132390d6d4ef9f5c5215f085eb4784c",
    "444780-2024": "5bc2ea3d2ed605865f328114c42ca105e93bba948e7c84c3e57564b95396f35b",
    "444780-2025": "fabca8fe3304e9c1b0bbb6a1774518db0cc651852184b54c08c4b52072e8fe20",
    "444490-2022": "4ac56095284bcfba065cbe152c79ad48819072095edbdea65d21fb47b1fc529d",
    "444490-2023": "09bf2e25fb6bf967de57ad7831d2aa31fade307896009f721e30f85484a3499d",
    "444490-2024": "f3dcd26e5d105e282b614f23c43c4f9491cfb535cff1fe263a9373be27e6d7f0",
    "444490-2025": "cf9276239e3587921dd757d44250632a950ac25735922dd47265e5ebad71ea96",
    "444380-2022": "093b4139bd7aeb5aeb2c5301d9337ac0493fb94aa85420187403c6de088d1592",
    "444380-2023": "2ae56e7509fa7971a129ec23512808e61b63e2114a447371ca4955830f8e1eec",
    "444380-2024": "f65751facf6c1e6f99e149bb93f58297f67968b5aad237dc7168db13de537511",
    "444380-2025": "0c56c21a5379de9b0cb9de1f8304eff4d34bbbfe567bc763287b069dbe00b407",
    "444180-2022": "b6c43b1943aec664090490fef481c2612376c893c4e1f6f1967d4b9a88835b3f",
    "444180-2023": "70d1c712c510fa8fd7db6ec203ea23c1d00ca2bd46331ce473b689532c5285cf",
    "444180-2024": "c51bc663c7b2fe755f09ebf97ea025d10aec7b4ffbbb3ac77896c0bd0fda4ed7",
    "444180-2025": "4d2e0c0fef140a6d26abc9b447c26d417ef730532eda9697cdf876fa5d41e9a5",
    "444340-2022": "8db75458b3419d0ee0b38b9384122f168c15dfecf1c8d691558e393578e26a21",
    "444340-2023": "c602de8b9743e969b535df44ea3bb334e6db64ba4167a175a730abd9df49e6ca",
    "444340-2024": "dba5c6c88f8c938ea5bfb136919c7c09eb98a5bcfeef2d21179aa3c4a7907ccc",
    "444340-2025": "7b6e108ad7af0035eb5b73c5e90daa0290a32840139135aaaa55f17060be4d92",
}

# The last hour every pinned weather file covers (Kathmandu, local time).
DATA_START = "2022-01-01"
DATA_END = "2025-08-23"

# ---------------------------------------------------------------------------
# The operating decision this forecast serves
# ---------------------------------------------------------------------------
# NEA bids into India's day-ahead market (IEX DAM), which closes at 12:00 IST (12:15 NPT)
# for the next day. The forecast is issued at 10:00 NPT on day D for all 24 hours of D+1,
# using SCADA load up to 09:00 on D. Nothing observed after 09:00 on D may enter a feature.
ISSUE_HOUR = 10
LAST_KNOWN_HOUR = 9

# Cost of error, used for the bid quantile and the NPR figures in reports. Assumptions, not NEA
# figures: IEX day-ahead clears near INR 5.5/kWh in the dry season; covering a shortfall in the
# real-time market or by cutting supply costs ~1.6x that; buying power you didn't need and
# reselling it loses ~1.0x.
NPR_PER_INR = 1.6
COST_UNDER_NPR_PER_MWH = 5.5 * 1000 * NPR_PER_INR * 1.6
COST_OVER_NPR_PER_MWH = 5.5 * 1000 * NPR_PER_INR * 1.0
BID_QUANTILE = COST_UNDER_NPR_PER_MWH / (COST_UNDER_NPR_PER_MWH + COST_OVER_NPR_PER_MWH)  # ~0.62

QUANTILES = (0.1, 0.5, 0.9)
ABSORB_MW = 50.0  # misses NEA can cover by re-dispatching its own storage plants

# ---------------------------------------------------------------------------
# MLOps policy (README, "What retraining policy?", and docs/ALTERNATIVES.md ADR-004/005 say why)
# ---------------------------------------------------------------------------
TRAIN_WINDOW_DAYS = 365  # rolling window: old demand levels hurt more than they help
MIN_TRAIN_DAYS = 180
SHADOW_DAYS = 7  # a challenger forecasts silently for this long before it can be promoted
PROMOTE_MIN_IMPROVEMENT = 0.03  # challenger must cut live MAPE by >= 3% relative to the champion
ROLLBACK_WINDOW_DAYS = 5
ROLLBACK_TOLERANCE = 1.15  # new champion is rolled back if its live MAPE is >15% worse than the previous one
SCHEDULED_RETRAIN_DAYS = 28
DRIFT_PSI_ALERT = 0.25
CUSUM_K = 0.02  # allowance: daily bias has a ~4% standard deviation, so 2% a day is noise
CUSUM_H = 0.20  # alarm when bias beyond the allowance adds up to 20 percentage points
