"""Tests for the configurable, OCP-aligned i4b reward (leap_c/examples/i4b/reward.py)."""

import numpy as np
import pytest

from leap_c.examples.i4b.reward import (
    RewardConfig,
    compute_reward,
    derive_ws,
    grid_signal_from_reward,
    make_scenarios,
    phi,
)


def test_r0_default_is_negative_energy():
    """Default RewardConfig must reproduce the historical r = -E_k exactly."""
    r, terms = compute_reward(RewardConfig(), E_k=2.0, price_k=0.3, d_k=1.5, o_k=0.7)
    assert r == pytest.approx(-2.0)
    assert terms["grid_signal"] == pytest.approx(1.0)
    assert terms["r_comfort"] == 0.0
    assert terms["r_overheat"] == 0.0
    assert terms["r_shaping"] == 0.0
    assert terms["cost_eur"] == pytest.approx(0.6)


def test_r1_cost_aware():
    """R1: r = -pi * E."""
    r, _ = compute_reward(RewardConfig(lam=1.0), E_k=2.0, price_k=0.3, d_k=1.5)
    assert r == pytest.approx(-0.6)


def test_r2_comfort_floor_quadratic():
    """R2: r = -E - w_c * d**2."""
    r, terms = compute_reward(RewardConfig(w_c=10.0), E_k=2.0, price_k=0.3, d_k=1.5)
    assert r == pytest.approx(-2.0 - 10.0 * 1.5**2)
    assert terms["r_comfort"] == pytest.approx(-10.0 * 1.5**2)


def test_r3_combined_normalized():
    """R3 normalized combined cost + comfort."""
    cfg = RewardConfig(lam=0.5, pi_ref=0.5, E_ref=2.0, dT_ref=1.0, w_c=3.0)
    r, terms = compute_reward(cfg, E_k=2.0, price_k=0.5, d_k=1.0)
    # grid = 0.5*(0.5/0.5) + 0.5 = 1.0; r_energy = -1.0*(2/2) = -1.0; r_comfort = -3*1 = -3.
    assert terms["grid_signal"] == pytest.approx(1.0)
    assert terms["r_energy"] == pytest.approx(-1.0)
    assert terms["r_comfort"] == pytest.approx(-3.0)
    assert r == pytest.approx(-4.0)


def test_overheating_penalty():
    cfg = RewardConfig(w_c=2.0, penalize_overheating=True)
    r, terms = compute_reward(cfg, E_k=0.0, price_k=0.0, d_k=0.0, o_k=2.0)
    assert terms["r_overheat"] == pytest.approx(-2.0 * 4.0)
    assert r == pytest.approx(-8.0)


@pytest.mark.parametrize(
    "shape,expected",
    [("linear", 2.0), ("quadratic", 4.0), ("mixed", 6.0)],
)
def test_penalty_shapes(shape, expected):
    assert phi(2.0, shape) == pytest.approx(expected)


def test_shaping_term():
    cfg = RewardConfig(shaping=True, Phi_scale=1.0, gamma=0.9)
    r, terms = compute_reward(
        cfg,
        E_k=0.0,
        price_k=0.0,
        d_k=0.0,
        T_room=22.0,
        T_set_lower=20.0,
        T_room_prev=21.0,
        T_set_lower_prev=20.0,
    )
    # gamma*Phi(s') - Phi(s) = 0.9*2 - 1 = 0.8
    assert terms["r_shaping"] == pytest.approx(0.8)
    assert r == pytest.approx(0.8)


def test_reward_is_sum_of_terms():
    cfg = RewardConfig(lam=0.5, pi_ref=0.4, E_ref=1.5, w_c=3.0, penalize_overheating=True)
    r, t = compute_reward(cfg, E_k=1.2, price_k=0.6, d_k=0.8, o_k=0.3)
    assert r == pytest.approx(t["r_energy"] + t["r_comfort"] + t["r_overheat"] + t["r_shaping"])


def test_derive_ws_floor_for_no_comfort():
    """w_c == 0 keeps the legacy regularizing slack (0.1) for QP conditioning."""
    assert derive_ws(RewardConfig(), 900.0) == pytest.approx(0.1)
    assert derive_ws(RewardConfig(lam=1.0), 900.0) == pytest.approx(0.1)


def test_derive_ws_25x_mapping():
    """At delta_t=900 (gap=25) and unit refs, ws = w_c / 25 (the note's heuristic)."""
    assert derive_ws(RewardConfig(w_c=10.0), 900.0) == pytest.approx(0.4)
    assert derive_ws(RewardConfig(w_c=25.0), 900.0) == pytest.approx(1.0)
    # E_ref and dT_ref scale the mapping.
    assert derive_ws(RewardConfig(w_c=3.0, E_ref=2.0, dT_ref=2.0), 900.0) == pytest.approx(
        3.0 * 2.0 / (25.0 * 4.0)
    )


def test_derive_ws_delta_t_scaling():
    """The energy-unit gap scales with delta_t (gap = delta_t/3600*100)."""
    assert derive_ws(RewardConfig(w_c=25.0), 1800.0) == pytest.approx(0.5)


def test_grid_signal_from_reward():
    # lam=0 -> identically 1.0 (energy-only, historical pinned value).
    assert grid_signal_from_reward(0.37, RewardConfig()) == pytest.approx(1.0)
    gs = grid_signal_from_reward(np.array([0.1, 0.5, 0.9]), RewardConfig())
    assert np.allclose(gs, 1.0)
    # lam=1, pi_ref=1 -> grid_signal == price.
    gs = grid_signal_from_reward(np.array([0.1, 0.5]), RewardConfig(lam=1.0))
    assert np.allclose(gs, [0.1, 0.5])
    # lam=0.5 normalized.
    gs = grid_signal_from_reward(np.array([0.4]), RewardConfig(lam=0.5, pi_ref=0.4))
    assert gs[0] == pytest.approx(1.0)


def test_make_scenarios_keys():
    sc = make_scenarios(pi_ref=0.9, E_ref=1.5)
    assert list(sc.keys()) == ["R0", "R1", "R2", "R3"]
    assert sc["R3"].pi_ref == pytest.approx(0.9)
    assert sc["R3"].E_ref == pytest.approx(1.5)


def test_env_r0_regression():
    """Integration: with the default (R0) reward, the env reward equals -E_el_kWh."""
    from i4b.gym_interface import BUILDING_NAMES2CLASS
    from i4b.models.model_hvac import Heatpump_AW

    from leap_c.examples.i4b.env import I4bEnv, I4bEnvConfig

    env = I4bEnv(
        cfg=I4bEnvConfig(
            building_params=BUILDING_NAMES2CLASS["i4c"],
            hp_model=Heatpump_AW(mdot_HP=0.25),
            days=1,
            seed=0,
        )
    )
    obs, _ = env.reset(seed=0)
    for _ in range(24):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        assert "reward_terms" in info
        assert r == pytest.approx(-info["E_el_kWh"])
        if terminated or truncated:
            break
