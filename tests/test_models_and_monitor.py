import numpy as np
import pandas as pd
import pytest

from bijuli import monitor
from bijuli.models import QuantileForecaster, bid_from_quantiles, frame_hash
from bijuli.registry import ModelCard, Registry


def test_quantiles_are_ordered_and_the_bid_sits_between_p50_and_p90(small_frame):
    f, _, _ = small_frame
    m = QuantileForecaster(params={"n_estimators": 40}).fit(f, calib_days=14)
    p = m.predict(f.tail(48))
    assert (p["p10"] <= p["p50"]).all() and (p["p50"] <= p["p90"]).all()
    assert ((p["bid"] >= p["p50"]) & (p["bid"] <= p["p90"])).all()
    assert np.isfinite(m.margin) and np.isfinite(m.holdout_mape)  # the margin may be negative: CQR can also narrow


def test_bid_quantile_follows_the_cost_asymmetry():
    q = pd.DataFrame({"p50": [1000.0], "p90": [1100.0]})
    assert abs(bid_from_quantiles(q, target=0.5)[0] - 1000.0) < 1e-9
    assert abs(bid_from_quantiles(q, target=0.9)[0] - 1100.0) < 1e-9


def test_training_data_hash_changes_when_the_data_does(small_frame):
    f, _, _ = small_frame
    g = f.copy()
    g.iloc[0, g.columns.get_loc("y")] += 1
    assert frame_hash(f) == frame_hash(f.copy()) != frame_hash(g)


def test_cusum_ignores_noise_but_catches_a_sustained_bias():
    rng = np.random.default_rng(3)
    c = monitor.Cusum()
    assert not any(c.update(b) for b in rng.normal(0, 0.03, 60) * 0.5)
    c = monitor.Cusum()
    hits = [c.update(b) for b in rng.normal(-0.04, 0.02, 30)]  # under-forecasting 4% a day
    assert any(hits) and hits.index(True) < 15


def test_psi_is_small_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    bins = monitor.reference_bins(ref, ["x"])
    assert monitor.psi(bins, pd.DataFrame({"x": rng.normal(0, 1, 2000)}))["x"] < 0.05
    assert monitor.psi(bins, pd.DataFrame({"x": rng.normal(1.0, 1, 2000)}))["x"] > 0.25


def _card(v, status="candidate"):
    return ModelCard(version=v, created="2025-01-01", status=status, reason="test", train_start="2024-01-01",
                     train_end="2024-12-31", data_sha256="abc")


def test_registry_promotion_and_rollback_round_trip(tmp_path):
    r = Registry(tmp_path)
    r.register({"m": 1}, _card("v001"))
    r.promote("v001", "2025-01-01", "initial")
    r.register({"m": 2}, _card("v002"))
    r.promote("v002", "2025-02-01", "shadow won")
    assert r.champion() == "v002" and r.card("v001")["status"] == "archived"
    assert r.rollback("2025-02-06", "live MAPE worse") == "v001"
    assert r.card("v002")["status"] == "rolled_back" and r.champion() == "v001"
    again = Registry(tmp_path)  # survives a restart
    assert again.load("v001") == {"m": 1}
    assert [e["event"] for e in again.events()][-1] in {"rolled_back", "promoted"}


def test_promote_refuses_unknown_versions_and_the_current_champion(tmp_path):
    r = Registry(tmp_path, persist_models=False)
    r.register({"m": 1}, _card("v001"))
    r.promote("v001", "2025-01-01", "initial")
    r.register({"m": 2}, _card("v002"))
    r.promote("v002", "2025-02-01", "shadow won")
    with pytest.raises(KeyError, match="v009"):
        r.promote("v009", "2025-02-02", "typo")
    with pytest.raises(ValueError, match="already the champion"):
        r.promote("v002", "2025-02-02", "re-run")
    assert r.champion() == "v002" and r.card("v002")["status"] == "champion"  # nothing half-applied
    assert r.rollback("2025-02-06", "live MAPE worse") == "v001"  # rollback target survived
