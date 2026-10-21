"""The daily operating loop, replayed day by day as if it were live.

Every morning at 10:00 (issue day D):
  1. validate the SCADA feed (contracts) and score the forecasts for D-1 against what happened
  2. serve the champion's forecast for D+1; any shadow challenger forecasts silently alongside it
  3. update the monitors (CUSUM on bias, rolling MAPE, weekly PSI)
  4. decide: retrain? promote the challenger? roll back the last promotion?

The same loop runs with different policies so their cost and accuracy can be compared on the same
days, the same weather and the same injected faults. That comparison is the point of this project:
what should trigger a retrain, and what has to be true before a new model is allowed to buy power.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import contracts, monitor
from .config import (
    PROMOTE_MIN_IMPROVEMENT,
    ROLLBACK_TOLERANCE,
    ROLLBACK_WINDOW_DAYS,
    SCHEDULED_RETRAIN_DAYS,
    SHADOW_DAYS,
    TRAIN_WINDOW_DAYS,
)
from .metrics import error_cost_npr
from .models import QuantileForecaster
from .registry import ModelCard, Registry

GATE_HOLDOUT_DAYS = 14
DRIFT_COOLDOWN_DAYS = 7
BACKSTOP_DAYS = 90
MAPE_TRIGGER_RATIO = 1.5  # 7-day live MAPE above 1.5x the model's own offline MAPE

# Fault injection (identical for every policy)
BAD_RELEASE_FROM = "2024-08-15"  # the first model each policy trains on/after this day has the Fahrenheit bug
UNIT_GLITCH = ("2024-02-05", "2024-02-07")  # SCADA export wrote kW into the MW column
FEED_OUTAGE = ("2024-06-10", "2024-06-12")  # historian down: no load data for these days


@dataclass
class Policy:
    name: str
    scheduled_days: int | None = None  # retrain every N days
    drift_trigger: bool = False
    safe_rollout: bool = False  # offline gate + shadow + automatic rollback
    contracts: bool = True
    retrain: bool = True
    mape_trigger: bool = True  # also retrain when 7-day MAPE > 1.5x the model's held-out MAPE


POLICIES = [
    Policy("Train once, never retrain", retrain=False),
    Policy("Retrain every 4 weeks, auto-deploy", scheduled_days=SCHEDULED_RETRAIN_DAYS),
    Policy("Retrain every 4 weeks, gated rollout", scheduled_days=SCHEDULED_RETRAIN_DAYS, safe_rollout=True),
    Policy("Retrain on drift only, gated rollout", drift_trigger=True, safe_rollout=True),
    Policy("Every 4 weeks or on bias drift, gated (Bijuli)", scheduled_days=SCHEDULED_RETRAIN_DAYS,
           drift_trigger=True, safe_rollout=True, mape_trigger=False),
    Policy("Bijuli without data contracts", scheduled_days=SCHEDULED_RETRAIN_DAYS, drift_trigger=True,
           safe_rollout=True, mape_trigger=False, contracts=False),
]


def inject_faults(truth: pd.Series) -> pd.Series:
    """The feed as SCADA delivers it: the truth plus three real-world faults."""
    feed = truth.copy()
    m = (feed.index >= UNIT_GLITCH[0]) & (feed.index < pd.Timestamp(UNIT_GLITCH[1]) + pd.Timedelta(days=1))
    feed[m] = feed[m] * 1000.0
    m = (feed.index >= FEED_OUTAGE[0]) & (feed.index < pd.Timestamp(FEED_OUTAGE[1]) + pd.Timedelta(days=1))
    feed[m] = np.nan
    return feed


def fahrenheit_release(frame: pd.DataFrame) -> pd.DataFrame:
    """The bug planted in the bad release: the retraining job reads weather from a new provider whose
    API defaults to Fahrenheit, while the daily forecast job still reads the old Celsius feed. Every
    temperature the model learned from is on a different scale from the ones it will be given."""
    f = frame.copy()
    for c in ("fc_ktm_tmean", "fc_ktm_tmin", "obs_ktm_tmean_d1", "fc_apparent_terai"):
        f[c] = f[c] * 1.8 + 32
    f["fc_minus_obs_ktm"] = f["fc_minus_obs_ktm"] * 1.8
    return f


@dataclass
class ReplayResult:
    policy: str
    daily: pd.DataFrame
    events: list
    trainings: int
    summary: dict = field(default_factory=dict)


class Operator:
    def __init__(self, policy: Policy, frame: pd.DataFrame, truth_by_day: pd.DataFrame, registry_root=None):
        self.p = policy
        self.frame = frame  # features (built from the feed) + y from the feed
        self.truth = truth_by_day  # target_day x 24 true MW, used only for scoring
        self.reg = Registry(registry_root or tempfile.mkdtemp(prefix="bijuli-reg-"), persist_models=False)
        self.by_day = {d: g for d, g in frame.groupby("target_day")}
        self.forecasts: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
        self.served: dict[pd.Timestamp, str] = {}  # target day -> version whose forecast bought power
        self.cusum = monitor.Cusum()
        self.shadow: dict | None = None  # {"version", "since"}
        self.watch: dict | None = None  # rollback watch after a promotion
        self.last_train: pd.Timestamp | None = None
        self.trainings = 0
        self.bug_shipped: pd.Timestamp | None = None
        self.log: list[dict] = []

    # -- model lifecycle ----------------------------------------------------------------------
    def _train(self, issue_day: pd.Timestamp, end_day: pd.Timestamp, reason: str, register: bool = True):
        """Fit on the trailing window ending at end_day. The gate copy skips conformal calibration."""
        start = end_day - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
        f = self.frame[(self.frame["target_day"] > start) & (self.frame["target_day"] <= end_day)]
        f = f[f["y"].notna() & f["anchor"].notna() & (f["anchor"] > 0)]
        # One bad release per policy: the first training job after the buggy feature code shipped
        # (the gate copy and the candidate of that run share the bug, as they share the code).
        buggy = issue_day >= pd.Timestamp(BAD_RELEASE_FROM) and self.bug_shipped in (None, issue_day)
        if buggy:
            self.bug_shipped = issue_day
        model = QuantileForecaster().fit(fahrenheit_release(f) if buggy else f, calib_days=42 if register else 0)
        model.meta["has_unit_bug"] = buggy
        self.trainings += 1
        if not register:
            return model, None
        card = ModelCard(
            version=self.reg.next_version(), created=str(issue_day.date()), status="candidate", reason=reason,
            train_start=model.meta["train_start"], train_end=model.meta["train_end"],
            data_sha256=model.meta["data_sha256"], offline={"holdout_mape": model.holdout_mape, "conformal_margin": model.margin},
            reference=monitor.reference_bins(f), params={"unit_bug": buggy},
        )
        return model, self.reg.register(model, card)

    def _offline_gate(self, issue_day: pd.Timestamp) -> tuple[bool, dict]:
        """Train a copy on data up to D-15, forecast D-14..D-1 through the *training* pipeline, and
        compare with what the champion actually served on those days. Catches models that are simply
        worse. It cannot catch training-serving skew: both halves of this test use the training code."""
        cut = issue_day - pd.Timedelta(days=GATE_HOLDOUT_DAYS + 1)
        model, _ = self._train(issue_day, cut, "gate copy", register=False)
        days = pd.date_range(cut + pd.Timedelta(days=1), issue_day - pd.Timedelta(days=1))
        f = pd.concat([self.by_day[d] for d in days if d in self.by_day]).dropna(subset=["anchor"])
        # The training pipeline builds its evaluation features with the same code it trains with,
        # so a buggy release is evaluated on equally buggy features. Truth is looked up by real hour.
        x = fahrenheit_release(f) if model.meta.get("has_unit_bug") else f
        cand = self._mape_rows(f, model.predict(x)["p50"])
        champ = self._live_mape(self.reg.champion(), days)
        return cand <= champ, {"candidate": round(cand, 4), "champion": round(champ, 4)}

    def _mape_rows(self, f, p50) -> float:
        y = self.truth.reindex(f["target_day"].values).to_numpy()[np.arange(len(f)), f["hour"].to_numpy()]
        return float(np.nanmean(np.abs(p50.to_numpy() - y) / y))

    def _live_mape(self, version, days) -> float:
        errs = []
        for d in days:
            fc = self.forecasts.get((version, d))
            if fc is not None:
                y = self.truth.loc[d].to_numpy()
                errs.append(np.abs(fc["p50"].to_numpy() - y) / y)
        return float(np.nanmean(np.concatenate(errs))) if errs else np.nan

    # -- serving ------------------------------------------------------------------------------
    def _serve(self, version: str, target: pd.Timestamp) -> pd.DataFrame | None:
        rows = self.by_day.get(target)
        if rows is None:
            return None
        rows = rows.copy()
        # Fallback chain when yesterday's SCADA is missing: last week's same hour, rescaled; else refuse.
        missing = rows["anchor"].isna()
        rows.loc[missing, "anchor"] = rows.loc[missing, "lag_w_h"]
        if rows["anchor"].isna().any():
            return None
        model = self.reg.load(version)
        out = model.predict(rows)  # always the serving path: correct Nepal-time features
        out["degraded"] = missing.to_numpy()
        return out

    # -- the day ------------------------------------------------------------------------------
    def run_day(self, d: pd.Timestamp) -> None:
        target, yesterday = d + pd.Timedelta(days=1), d - pd.Timedelta(days=1)
        champ = self.reg.champion()
        row = {"issue_day": d, "champion": champ, "served": self.served.get(yesterday), "event": ""}

        # 1. score yesterday: the forecast that was actually served, whichever version made it
        served = self.served.get(yesterday)
        fc = self.forecasts.get((served, yesterday))
        if fc is not None and yesterday in self.truth.index:
            y = self.truth.loc[yesterday].to_numpy()
            e = (fc["p50"].to_numpy() - y) / y
            row.update(mape=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                       cost_npr=error_cost_npr(y, fc["bid"].to_numpy()), degraded=bool(fc["degraded"].any()))
            if self.p.drift_trigger:
                self.cusum.update(row["bias"])
        elif yesterday in self.truth.index and yesterday > self.start:  # a forecast for D-1 was due
            row.update(mape=np.nan, refused=True)

        # 2. serve tomorrow (champion), plus shadow / reverse-shadow forecasts
        for v in {champ, (self.shadow or {}).get("version"), (self.watch or {}).get("previous")} - {None}:
            out = self._serve(v, target)
            if out is not None:
                self.forecasts[(v, target)] = out
                if v == champ:
                    self.served[target] = v

        # 3/4. decisions
        if self.p.retrain:
            self._decide(d, row)
        self.log.append(row)

    def _decide(self, d: pd.Timestamp, row: dict) -> None:
        day = str(d.date())
        ago = lambda n: pd.date_range(d - pd.Timedelta(days=n), d - pd.Timedelta(days=1))  # noqa: E731

        # rollback watch on the latest promotion
        if self.watch and (d - self.watch["since"]).days >= ROLLBACK_WINDOW_DAYS:
            days = pd.date_range(self.watch["since"] + pd.Timedelta(days=1), d - pd.Timedelta(days=1))
            new, old = self._live_mape(self.reg.champion(), days), self._live_mape(self.watch["previous"], days)
            if new > old * ROLLBACK_TOLERANCE:
                self.reg.rollback(day, f"live MAPE {new:.3f} vs previous champion {old:.3f}")
                row["event"] += f"ROLLBACK to {self.reg.champion()} ({new:.1%} vs {old:.1%}); "
                self.cusum.reset()
            self.watch = None

        # shadow evaluation
        if self.shadow and (d - self.shadow["since"]).days > SHADOW_DAYS:
            v = self.shadow["version"]
            days = pd.date_range(self.shadow["since"] + pd.Timedelta(days=2), d - pd.Timedelta(days=1))
            cand, champ = self._live_mape(v, days), self._live_mape(self.reg.champion(), days)
            ok = cand <= champ * (1 - PROMOTE_MIN_IMPROVEMENT) or (self.cusum.alarm and cand <= champ)
            if ok:
                self._promote(v, d, f"shadow {cand:.3f} vs champion {champ:.3f}", row)
            else:
                self.reg.set_status(v, "rejected", day, f"shadow {cand:.3f} vs champion {champ:.3f}")
                row["event"] += f"shadow rejected {v} ({cand:.1%} vs {champ:.1%}); "
            self.shadow = None

        # retrain triggers
        reason = None
        since = (d - self.last_train).days if self.last_train is not None else 10**6
        if self.p.scheduled_days and since >= self.p.scheduled_days:
            reason = "schedule"
        if self.p.drift_trigger and not self.shadow and since >= DRIFT_COOLDOWN_DAYS:
            champ = self.reg.champion()
            live = self._live_mape(champ, ago(7))
            promised = self.reg.card(champ)["offline"]["holdout_mape"]
            if self.cusum.alarm:
                reason = f"bias drift (CUSUM {max(self.cusum.pos, self.cusum.neg):.2f})"
            elif self.p.mape_trigger and np.isfinite(live) and live > MAPE_TRIGGER_RATIO * promised:
                reason = f"7-day MAPE {live:.3f} > {MAPE_TRIGGER_RATIO}x promised {promised:.3f}"
            elif not self.p.scheduled_days and since >= BACKSTOP_DAYS:
                reason = "90-day backstop"
        if reason and not self.shadow:
            self._retrain(d, reason, row)

    def _retrain(self, d: pd.Timestamp, reason: str, row: dict) -> None:
        day = str(d.date())
        self.last_train = d
        if self.p.safe_rollout:
            passed, scores = self._offline_gate(d)
            if not passed:
                row["event"] += f"offline gate rejected ({reason}): {scores}; "
                return
        model, v = self._train(d, d - pd.Timedelta(days=1), reason)
        if self.p.safe_rollout:
            self.reg.set_status(v, "shadow", day, reason)
            self.shadow = {"version": v, "since": d}
            row["event"] += f"trained {v} -> shadow ({reason}); "
        else:
            self._promote(v, d, reason, row)

    def _promote(self, v: str, d: pd.Timestamp, why: str, row: dict) -> None:
        prev = self.reg.champion()
        self.reg.promote(v, str(d.date()), why)
        self.cusum.reset()
        row["event"] += f"PROMOTED {v} ({why}); "
        if self.p.safe_rollout and prev:
            self.watch = {"since": d, "previous": prev}

    def bootstrap(self, first_issue_day: pd.Timestamp) -> None:
        self.start = first_issue_day
        _, v = self._train(first_issue_day, first_issue_day - pd.Timedelta(days=1), "initial model")
        self.reg.promote(v, str(first_issue_day.date()), "initial model")
        self.last_train = first_issue_day


def replay(policy: Policy, frame_clean: pd.DataFrame, frame_raw: pd.DataFrame, truth_by_day: pd.DataFrame,
           start: str, end: str) -> ReplayResult:
    frame = frame_clean if policy.contracts else frame_raw
    op = Operator(policy, frame, truth_by_day)
    days = pd.date_range(start, end)
    op.bootstrap(days[0])
    for d in days:
        op.run_day(d)
    daily = pd.DataFrame(op.log).set_index("issue_day")
    scored = daily.dropna(subset=["mape"]) if "mape" in daily else daily
    summary = {
        "policy": policy.name,
        "mape": float(scored["mape"].mean()),
        "worst_week_mape": float(scored["mape"].rolling(7).mean().max()),
        "cost_npr_m_per_day": float(scored["cost_npr"].mean() / 1e6),
        "trainings": op.trainings,
        "promotions": sum(1 for e in op.reg.events() if e["event"] == "promoted") - 1,
        "rollbacks": sum(1 for e in op.reg.events() if e["event"] == "rolled_back"),
        "days_refused": int(daily.get("refused", pd.Series(dtype=bool)).fillna(False).sum()),
        "days_on_buggy_model": int(sum(op.reg.card(c)["params"].get("unit_bug", False) for c in daily["served"].dropna())),
    }
    return ReplayResult(policy.name, daily, op.reg.events(), op.trainings, summary)


def prepare(truth: pd.Series, daily_weather, forecast, build):
    """Feature frames for the replay: from the faulty feed after contracts (clean) and without (raw)."""
    feed = inject_faults(truth)
    clean, findings = contracts.check_load(feed, reference_median=float(truth.loc[: UNIT_GLITCH[0]].median()))
    frame_clean = build(clean, daily_weather, forecast)
    frame_raw = build(feed, daily_weather, forecast)
    truth_by_day = truth.to_frame("y").assign(d=truth.index.normalize(), h=truth.index.hour).pivot_table(
        index="d", columns="h", values="y")
    return frame_clean, frame_raw, truth_by_day, findings
