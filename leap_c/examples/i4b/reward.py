"""Configurable, OCP-aligned reward for the i4b environment.

This module is the single source of truth for the i4b reward.  The same
``RewardConfig`` object is embedded in both ``I4bEnvConfig`` and
``I4bPlannerConfig`` so the model-free actor and the differentiable-MPC planner
optimize the *same* objective (the env docstring's "matches the OCP exactly"
promise).  See ``~/Documents/llm_vault/wiki/leap-c/i4b-reward-design.md``.

General parametric reward (per step k), env side:

    r_k = -[ lam*(pi_k/pi_ref) + (1-lam) ] * (E_k/E_ref)
          - w_c * phi(d_k/dT_ref)
          [ - w_c * phi(o_k/dT_ref)        if penalize_overheating ]
          [ + gamma*Phi(s_{k+1}) - Phi(s_k) if shaping ]   (R4, off by default)

with
    E_k     electrical energy this step [kWh]
    pi_k    electricity price [EUR/kWh]
    d_k     under-heating max(0, T_set_lower - T_room) [K]
    o_k     over-heating  max(0, T_room - T_set_upper) [K]
    lam     interpolates energy-optimal (0) <-> cost-optimal (1)
    Phi(s)  ~ (T_room - T_set_lower)_+  potential-based pre-heat shaping

Named scenarios (R0..R3) are just presets of ``RewardConfig`` -- see
``REWARD_SCENARIOS`` / ``make_scenarios``.

OCP co-design (so env and OCP read identical numbers):
    * price weighting -> per-stage ``grid_signal_k = lam*(pi_k/pi_ref) + (1-lam)``
      (``grid_signal_from_reward``); at lam=0 this is identically 1.0.
    * comfort weighting -> the OCP quadratic slack ``ws`` derived from the env
      comfort weight via ``derive_ws`` (the constant ``delta_t/3600*100`` is the
      env/OCP energy-unit gap, = 25 at delta_t=900 s).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

PenaltyShape = Literal["linear", "quadratic", "mixed"]

# Hard thermal-power cap used by the OCP (acados_ocp.py: uh = [..., 26.0]) [kW].
_P_TH_CAP_KW = 26.0


@dataclass(kw_only=True)
class RewardConfig:
    """Parameters that define the i4b reward.

    The defaults reproduce R0 (energy-only, ``r = -E_k``), which is the historical
    i4b reward when ``grid_signal == 1``.  This keeps every existing caller
    unchanged.

    Attributes:
        lam: Energy/cost interpolation in [0, 1]; 0 = energy-optimal, 1 = cost-optimal.
        pi_ref: Price normalizer [EUR/kWh].
        E_ref: Energy normalizer [kWh].
        dT_ref: Comfort-deviation normalizer [K].
        w_c: Comfort penalty weight (under-heating, and over-heating if enabled).
        penalty_shape: Shape phi of the comfort penalty: "linear" (x), "quadratic"
            (x**2, mirrors the OCP slack), or "mixed" (x**2 + x, hvac-style).
        penalize_overheating: Also penalize over-heating o_k symmetrically.
        shaping: Enable potential-based pre-heat shaping (R4). Off by default.
        gamma: Discount used only by the shaping term.
        Phi_scale: Coefficient of the potential Phi(s) ~ (T_room - T_set_lower)_+.
    """

    lam: float = 0.0
    pi_ref: float = 1.0
    E_ref: float = 1.0
    dT_ref: float = 1.0
    w_c: float = 0.0
    penalty_shape: PenaltyShape = "quadratic"
    penalize_overheating: bool = False
    # R4 potential-based pre-heat shaping (deferred; off by default).
    shaping: bool = False
    gamma: float = 1.0
    Phi_scale: float = 0.0


def phi(x: float, shape: PenaltyShape) -> float:
    """Comfort penalty shape applied to a non-negative deviation ``x``."""
    if shape == "linear":
        return x
    if shape == "quadratic":
        return x * x
    if shape == "mixed":
        return x * x + x
    raise ValueError(f"Unknown penalty_shape: {shape!r}")


def Phi(T_room: float, T_set_lower: float, scale: float) -> float:
    """Potential rewarding stored thermal energy above the lower comfort bound."""
    return scale * max(0.0, T_room - T_set_lower)


def grid_signal_from_reward(price: "float | np.ndarray", cfg: RewardConfig) -> "float | np.ndarray":
    """OCP per-stage ``grid_signal`` aligned with the env energy weighting.

    ``grid_signal = lam*(price/pi_ref) + (1-lam)``.  Accepts a scalar or array of
    prices (the planner passes the staged price forecast).  At ``lam == 0`` this is
    identically ``1.0`` -- i.e. the historical pinned value.
    """
    return cfg.lam * (price / cfg.pi_ref) + (1.0 - cfg.lam)


def derive_ws(cfg: RewardConfig, delta_t: float) -> float:
    """OCP quadratic slack weight ``ws`` aligned with the env comfort weight.

    Matching the comfort/energy ratio between env and OCP gives
    ``ws = w_c * E_ref / ((delta_t/3600*100) * dT_ref**2)``.  The factor
    ``delta_t/3600*100`` is the env/OCP energy-unit gap (= 25 at delta_t=900 s),
    recovering the note's ``w_c ~= 25*ws`` heuristic when E_ref = dT_ref = 1.

    For ``w_c == 0`` (no env comfort term, e.g. R0/R1) the OCP keeps a small
    regularizing slack (the legacy 0.1) so comfort stays MPC-protected and the QP
    is well conditioned -- this matches the historical behavior exactly.
    """
    if cfg.w_c == 0.0:
        return 0.1
    energy_unit_gap = (delta_t / 3600.0) * 100.0
    return cfg.w_c * cfg.E_ref / (energy_unit_gap * cfg.dT_ref**2)


def compute_refs(dataset, hp_model, delta_t: float) -> tuple[float, float]:
    """hvac-style normalization references from the dataset and HP model.

    Returns ``(pi_ref, E_ref)`` where ``pi_ref`` is the dataset price maximum
    [EUR/kWh] and ``E_ref`` is the maximum plausible step electrical energy [kWh],
    taken as the OCP thermal-power cap divided by a nominal COP (A7/W35 rating
    point) over one step.
    """
    pi_ref = float(dataset.max["price"])
    cop_nom = float(hp_model.COP(35.0, 7.0))
    E_ref = (_P_TH_CAP_KW / cop_nom) * (delta_t / 3600.0)
    return pi_ref, E_ref


def compute_reward(
    cfg: RewardConfig,
    *,
    E_k: float,
    price_k: float,
    d_k: float,
    o_k: float = 0.0,
    T_room: float | None = None,
    T_set_lower: float | None = None,
    T_room_prev: float | None = None,
    T_set_lower_prev: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute the reward and its decomposition for one step.

    Args:
        cfg: The reward configuration (single source of truth, shared with the OCP).
        E_k: Electrical energy this step [kWh].
        price_k: Electricity price for this interval [EUR/kWh].
        d_k: Under-heating deviation max(0, T_set_lower - T_room) [K].
        o_k: Over-heating deviation max(0, T_room - T_set_upper) [K].
        T_room: End-of-step room temperature, used by the shaping potential Phi(s_{k+1}).
        T_set_lower: End-of-step lower comfort bound, used by Phi(s_{k+1}).
        T_room_prev: Start-of-step room temperature, used by Phi(s_k).
        T_set_lower_prev: Start-of-step lower comfort bound, used by Phi(s_k).

    Returns:
        ``(reward, terms)`` where ``terms`` publishes every additive component plus
        the inputs needed for the summary table.  ``terms["grid_signal"]`` is the
        same number the OCP uses at this stage.
    """
    grid_signal = float(grid_signal_from_reward(price_k, cfg))
    r_energy = -grid_signal * (E_k / cfg.E_ref)

    r_comfort = -cfg.w_c * phi(d_k / cfg.dT_ref, cfg.penalty_shape)

    r_overheat = 0.0
    if cfg.penalize_overheating:
        r_overheat = -cfg.w_c * phi(o_k / cfg.dT_ref, cfg.penalty_shape)

    r_shaping = 0.0
    if cfg.shaping:
        Phi_next = Phi(T_room, T_set_lower, cfg.Phi_scale)
        Phi_prev = Phi(T_room_prev, T_set_lower_prev, cfg.Phi_scale)
        r_shaping = cfg.gamma * Phi_next - Phi_prev

    reward = r_energy + r_comfort + r_overheat + r_shaping

    terms = {
        "r_energy": r_energy,
        "r_comfort": r_comfort,
        "r_overheat": r_overheat,
        "r_shaping": r_shaping,
        "reward": reward,
        "E_kWh": E_k,
        "price": price_k,
        "cost_eur": price_k * E_k,
        "d_under": d_k,
        "o_over": o_k,
        "grid_signal": grid_signal,
    }
    return reward, terms


def make_scenarios(pi_ref: float = 1.0, E_ref: float = 1.0) -> dict[str, RewardConfig]:
    """Named reward presets (R0..R3).

    R0 energy-only, R1 cost-aware (r = -pi*E), R2 comfort floor (r = -E - w_c*d**2),
    R3 combined cost+comfort (normalized; the recommended default).  Pass the dataset
    references (see ``compute_refs``) to normalize R3's energy/cost terms.

    R4 (potential-based pre-heat shaping) is intentionally omitted from the default
    set; obtain it by setting ``shaping=True`` (with ``Phi_scale``/``gamma``) on R3.
    """
    return {
        "R0": RewardConfig(),
        "R1": RewardConfig(lam=1.0),
        "R2": RewardConfig(w_c=10.0),
        "R3": RewardConfig(lam=0.5, pi_ref=pi_ref, E_ref=E_ref, dT_ref=1.0, w_c=3.0),
    }


REWARD_SCENARIOS: dict[str, RewardConfig] = make_scenarios()
