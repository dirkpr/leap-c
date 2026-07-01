"""Offline tests for the minimal BOPTEST heat-pump example.

The planner / OCP is exercised on synthetic observations so no BOPTEST server is
needed. A single end-to-end env test is included but skipped unless a server URL
is provided via the ``BOPTEST_URL`` environment variable.
"""

import os

import numpy as np
import pytest
import torch

from leap_c.examples.boptest.acados_ocp import FORECAST_PARAM_NAMES
from leap_c.examples.boptest.planner import BoptestPlanner, BoptestPlannerConfig

# Small horizon keeps the one-off acados solver build fast.
N = 8


@pytest.fixture(scope="module")
def planner() -> BoptestPlanner:
    return BoptestPlanner(cfg=BoptestPlannerConfig(N_horizon=N))


def _make_obs(
    batch_size: int = 1,
    t_zone: float = 290.0,
    t_amb: float = 278.15,
    solar: float = 0.0,
    price: float = 0.2,
    t_lower: float = 294.15,
    t_upper: float = 297.15,
) -> dict:
    n_fc = N + 1
    forecast_values = {
        "T_amb": t_amb,
        "solar": solar,
        "price": price,
        "T_lower": t_lower,
        "T_upper": t_upper,
    }
    forecast = {
        name: torch.full((batch_size, n_fc), value, dtype=torch.float64)
        for name, value in forecast_values.items()
    }
    return {
        "state": torch.full((batch_size, 1), t_zone, dtype=torch.float64),
        "forecast": forecast,
    }


def test_forecast_names_consistent():
    assert FORECAST_PARAM_NAMES == ["T_amb", "solar", "price", "T_lower", "T_upper"]


@pytest.mark.parametrize("batch_size", [1, 4])
def test_forward_shapes_and_bounds(planner: BoptestPlanner, batch_size: int):
    obs = _make_obs(batch_size=batch_size)
    ctx, u0, x, u, value = planner.forward(obs)

    assert u0.shape == (batch_size, 1)
    assert torch.isfinite(u0).all()
    # Modulation must respect the hard box [0, 1].
    assert (u0 >= -1e-6).all() and (u0 <= 1.0 + 1e-6).all()
    assert x.shape[0] == batch_size and torch.isfinite(x).all()
    assert torch.isfinite(value).all()


def test_heats_when_cold(planner: BoptestPlanner):
    # Zone well below the comfort band with cold ambient: expect strong heating.
    obs = _make_obs(t_zone=289.0, t_amb=273.15)
    _, u0, _, _, _ = planner.forward(obs)
    assert u0.item() > 0.5


def test_idle_when_warm(planner: BoptestPlanner):
    # Zone comfortably inside the band (heating-only): expect little/no heating,
    # since running the heat pump only adds electricity cost.
    obs = _make_obs(t_zone=296.0, t_amb=283.15)
    _, u0, _, _, _ = planner.forward(obs)
    assert u0.item() < 0.5


def test_default_param_in_param_space(planner: BoptestPlanner):
    param = planner.default_param(None)
    assert planner.param_space.contains(
        {k: np.asarray(v, dtype=np.float64) for k, v in param.items()}
    )


@pytest.mark.skipif(
    not os.environ.get("BOPTEST_URL"),
    reason="needs a running BOPTEST server (set BOPTEST_URL)",
)
def test_env_reset_step():
    from leap_c.examples.boptest.env import BoptestEnv, BoptestEnvConfig

    cfg = BoptestEnvConfig(url=os.environ["BOPTEST_URL"], N_forecast=N, max_episode_length=3600)
    env = BoptestEnv(cfg=cfg)

    obs, _ = env.reset()
    assert obs["state"].shape == (1,)
    for name in FORECAST_PARAM_NAMES:
        assert obs["forecast"][name].shape == (N + 1,)

    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
