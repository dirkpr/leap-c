"""Randomization and train/test-split behaviour of the i4b environment.

Covers the four stochasticity levers wired into ``I4bEnv``: random per-episode
start window, the non-overlapping train/test split (reused from ``HvacDataset``),
observation-only measurement noise, and process noise on the true state.

Assertions are structural (distinctness, reproducibility, non-overlap) rather than
tied to specific indices, so they hold for both the real local data assets and the
synthetic hermetic stubs used in CI (see ``conftest.py``).
"""

import numpy as np
import pytest

pytest.importorskip("i4b")

from leap_c.examples.hvac.forecast import ForecastConfig  # noqa: E402
from leap_c.examples.i4b.env import (  # noqa: E402
    BUILDING_NAMES2CLASS,
    Heatpump_AW,
    I4bEnv,
    I4bEnvConfig,
)


def _cfg(**kw) -> I4bEnvConfig:
    base = dict(
        building_params=BUILDING_NAMES2CLASS["i4c"],
        hp_model=Heatpump_AW(mdot_HP=0.25),
        seed=0,  # pin the COFACTOR gains draw so envs are comparable across configs
    )
    base.update(kw)
    return I4bEnvConfig(**base)


def _short_rollout(env: I4bEnv, seed: int, n: int = 15) -> np.ndarray:
    env.reset(seed=seed, options={"mode": "train"})
    a = np.zeros(1, dtype=np.float32)
    return np.array([env.step(a)[0]["state"].copy() for _ in range(n)])


# ── Start-window selection / split ────────────────────────────────────────────


def test_train_split_decorrelates_start_window():
    """Different seeds must land on different train windows (the main lever)."""
    env = I4bEnv(cfg=_cfg())
    starts = []
    for s in range(6):
        env.reset(seed=s, options={"mode": "train"})
        starts.append(int(env._idx))
    assert len(set(starts)) >= 5


def test_test_split_is_reproducible_and_in_season():
    """Validation windows are fixed, replayable, and restricted to heating months."""
    env = I4bEnv(cfg=_cfg())

    def test_sequence():
        env.reset(seed=123)  # no "mode" -> test split, resets the test cursor
        seq = [int(env._idx)]
        for _ in range(4):
            env.reset()  # cycles the fixed test windows
            seq.append(int(env._idx))
        return seq

    seq1 = test_sequence()
    seq2 = test_sequence()
    assert seq1 == seq2
    months = {env.dataset.index[i].month for i in seq1}
    assert months <= {1, 2, 12}


def test_train_windows_never_overlap_test_windows():
    """No leakage: train start windows stay clear of every reserved test window."""
    cfg = _cfg()
    env = I4bEnv(cfg=cfg)
    env.reset(seed=0)  # materialise the stratified test windows
    test_indices = list(env.dataset._test_indices)
    assert test_indices  # the split must be active
    episode_len = cfg.N_forecast + cfg.days * 24 * 4
    for s in range(40):
        env.reset(seed=s, options={"mode": "train"})
        ti = int(env._idx)
        assert all(abs(ti - tj) >= episode_len for tj in test_indices)


def test_split_disabled_when_no_test_episodes():
    """total_test_episodes=0 disables the split (eval samples freely)."""
    env = I4bEnv(cfg=_cfg(total_test_episodes=0))
    env.reset(seed=0)
    assert env.dataset._test_indices == []


def test_start_date_pins_window_deterministically():
    """A fixed start_date is the deterministic escape hatch: seed-independent."""
    env = I4bEnv(cfg=_cfg(start_date="2021-12-15", total_test_episodes=0))
    env.reset(seed=1, options={"mode": "train"})
    i1 = int(env._idx)
    env.reset(seed=2)
    i2 = int(env._idx)
    assert i1 == i2
    assert env.dataset.index[i1].strftime("%Y-%m-%d") == "2021-12-15"


# ── Noise ─────────────────────────────────────────────────────────────────────


def test_clean_env_is_deterministic():
    env = I4bEnv(cfg=_cfg())
    assert np.allclose(_short_rollout(env, 7), _short_rollout(env, 7))


def test_measurement_noise_perturbs_observation():
    clean = _short_rollout(I4bEnv(cfg=_cfg()), 7)
    noisy = _short_rollout(I4bEnv(cfg=_cfg(noise_level=0.2)), 7)
    assert not np.allclose(clean, noisy)


def test_process_noise_perturbs_state():
    clean = _short_rollout(I4bEnv(cfg=_cfg()), 7)
    noisy = _short_rollout(I4bEnv(cfg=_cfg(process_noise_std=0.05)), 7)
    assert not np.allclose(clean, noisy)


def test_noise_is_reproducible_with_seed():
    a = _short_rollout(I4bEnv(cfg=_cfg(noise_level=0.1, process_noise_std=0.02)), 7)
    b = _short_rollout(I4bEnv(cfg=_cfg(noise_level=0.1, process_noise_std=0.02)), 7)
    assert np.allclose(a, b)


# ── Forecast noise (optional hvac-style AR(1) model) ──────────────────────────


def _forecast_at_reset(seed: int = 7, **cfg_kw):
    """Reset on a fixed train window and return ``(env, obs)``."""
    env = I4bEnv(cfg=_cfg(**cfg_kw))
    obs, _ = env.reset(seed=seed, options={"mode": "train"})
    return env, obs


def test_forecast_noise_is_on_by_default():
    """The default config applies AR(1) noise (forecast deviates from the dataset slice)."""
    env, obs = _forecast_at_reset()
    nf = env.cfg.N_forecast
    plain = env.dataset.get_column_view("temperature_2m", env._idx, nf)
    assert not np.allclose(obs["forecast"]["T_amb"], plain)


def test_perfect_foresight_when_disabled():
    """With forecast_noise=None the forecast is the exact dataset slice."""
    env, obs = _forecast_at_reset(forecast_noise=None)
    nf = env.cfg.N_forecast
    plain = env.dataset.get_column_view("temperature_2m", env._idx, nf)
    assert np.allclose(obs["forecast"]["T_amb"], plain)


def test_forecast_noise_perturbs_t_amb_and_keeps_solar_nonnegative():
    noise = ForecastConfig(temp_uncertainty="medium", solar_uncertainty="medium")
    env, obs = _forecast_at_reset(forecast_noise=noise)
    nf = env.cfg.N_forecast
    plain_T = env.dataset.get_column_view("temperature_2m", env._idx, nf)
    assert not np.allclose(obs["forecast"]["T_amb"], plain_T)
    for k in ("dhi", "ghi", "dni"):
        assert np.all(obs["forecast"][k] >= 0.0)


def test_forecast_noise_leaves_true_dynamics_untouched():
    """Only the forecast is noisy; the realized state and true gains are unchanged."""
    noise = ForecastConfig(temp_uncertainty="high", solar_uncertainty="high")
    env_clean = I4bEnv(cfg=_cfg(forecast_noise=None))
    env_noisy = I4bEnv(cfg=_cfg(forecast_noise=noise))
    env_clean.reset(seed=7, options={"mode": "train"})
    env_noisy.reset(seed=7, options={"mode": "train"})
    assert env_clean._idx == env_noisy._idx  # same start window

    a = np.zeros(1, dtype=np.float32)
    for _ in range(15):
        oc, _, _, _, ic = env_clean.step(a)
        on, _, _, _, in_ = env_noisy.step(a)
        assert np.allclose(oc["state"], on["state"])
        assert ic["Qdot_gains"] == in_["Qdot_gains"]


def test_forecast_noise_is_reproducible_with_seed():
    noise = ForecastConfig(temp_uncertainty="medium", solar_uncertainty="medium")
    _, obs1 = _forecast_at_reset(seed=11, forecast_noise=noise)
    _, obs2 = _forecast_at_reset(seed=11, forecast_noise=noise)
    for k in ("T_amb", "dhi", "ghi", "dni"):
        assert np.allclose(obs1["forecast"][k], obs2["forecast"][k])
