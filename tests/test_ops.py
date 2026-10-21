import pandas as pd

from bijuli import features, ops

from .conftest import synthetic_load, synthetic_weather


def test_the_unit_bug_converts_temperatures_and_nothing_else():
    f = pd.DataFrame({"fc_ktm_tmean": [0.0, 20.0], "fc_ktm_tmin": [-5.0, 10.0], "obs_ktm_tmean_d1": [0.0, 20.0],
                      "fc_apparent_terai": [30.0, 40.0], "fc_minus_obs_ktm": [0.0, 1.0], "hour": [0, 1]})
    g = ops.fahrenheit_release(f)
    assert list(g["fc_ktm_tmean"]) == [32.0, 68.0] and list(g["fc_minus_obs_ktm"]) == [0.0, 1.8]
    assert list(g["hour"]) == [0, 1] and list(f["fc_ktm_tmean"]) == [0.0, 20.0]  # original untouched


def test_injected_faults_are_where_the_scenario_says():
    idx = pd.date_range("2024-02-01", "2024-06-30 23:00", freq="h")
    truth = pd.Series(1500.0, index=idx)
    feed = ops.inject_faults(truth)
    assert (feed["2024-02-06"] == 1.5e6).all()
    assert feed["2024-06-11"].isna().all()
    assert (feed["2024-03-01"] == 1500.0).all()


def test_a_short_replay_runs_and_scores_every_served_day():
    load = synthetic_load(start="2023-01-01", days=400)
    w = synthetic_weather(start="2022-12-01", days=440)
    frame = features.build(load, w, w)
    truth = load.to_frame("y").assign(d=load.index.normalize(), h=load.index.hour).pivot_table(index="d", columns="h", values="y")
    pol = ops.Policy("test", drift_trigger=True, safe_rollout=True)
    res = ops.replay(pol, frame, frame, truth, "2023-12-01", "2024-01-20")
    assert res.summary["mape"] < 0.06
    assert res.daily["mape"].notna().sum() >= 49
    assert res.summary["days_refused"] == 0
