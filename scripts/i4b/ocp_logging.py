"""Log the OCP-solver configuration needed to rebuild an i4b policy.

The per-seed ``config.yaml`` written by ``Trainer.__init__`` only captures the
RL/trainer hyperparameters (``asdict(trainer_cfg)``). It does *not* record the
MPC/OCP controller config (horizon, comfort weight ``ws``, thermal model, the
reward scenario that drives the cost) nor the env recipe -- so a single seed
directory is not enough to rebuild the exact ``HierachicalMPCActor`` used at
training time.

``dump_solver_config`` closes that gap by writing, per run:

* ``ocp_solver_config.yaml`` -- the authoritative rebuild recipe: the resolved
  ``I4bPlannerConfig`` plus the env recipe, the reward scenario name, and the
  algorithm (so the reconstruction picks the right trainer/config class).
* ``acados_ocp.json`` -- the acados-native dump of the assembled OCP (cost
  weights, prediction horizon, model parameters) for human inspection.

See ``scripts/i4b/verify_reconstruction.py`` for the inverse (rebuild a policy
from these artifacts and reproduce its deterministic evaluation).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
from warnings import warn

from yaml import safe_dump

from leap_c.examples.i4b.env import I4bEnvConfig

SCHEMA_VERSION = 1


def _planner_cfg_to_dict(planner_cfg: Any) -> dict:
    """``asdict`` the I4bPlannerConfig, coercing the non-YAML-safe ``dtype``."""
    d = asdict(planner_cfg)
    dtype = d.get("dtype")
    if dtype is not None:
        d["dtype"] = str(dtype)
    return d


def _env_recipe(env_cfg: I4bEnvConfig, building_name: str) -> dict:
    """Capture the scalar fields needed to rebuild an equivalent ``I4bEnvConfig``.

    ``building_params`` is captured indirectly via ``building_name`` (a key into
    ``BUILDING_NAMES2CLASS``) and ``hp_model`` via its class name + mass flow, so
    the record stays small and YAML-safe.
    """
    return {
        "building_name": building_name,
        "hp_model": type(env_cfg.hp_model).__name__,
        "mdot_hp": float(env_cfg.mdot_hp),
        "method": env_cfg.method,
        "delta_t": int(env_cfg.delta_t),
        "days": env_cfg.days,
        "random_init": bool(env_cfg.random_init),
        "noise_level": float(env_cfg.noise_level),
        "T_set_lower": float(env_cfg.T_set_lower),
        "T_set_upper": float(env_cfg.T_set_upper),
        "N_forecast": int(env_cfg.N_forecast),
        "grid_signal": float(env_cfg.grid_signal),
        "apply_heating_logic": bool(env_cfg.apply_heating_logic),
        "start_date": env_cfg.start_date,
        "seed": env_cfg.seed,
        "reward": asdict(env_cfg.reward),
    }


def dump_solver_config(
    output_path: str | Path,
    controller: Any,
    *,
    algo: str,
    reward_name: str,
    env_cfg: I4bEnvConfig,
    building_name: str = "i4c",
) -> Path:
    """Write the OCP-solver reconstruction record (and acados_ocp.json) for a run.

    Args:
        output_path: The run's output directory (where ``config.yaml`` lives).
        controller: The controller used for training. For ``ControllerFromPlanner``
            the planner config and assembled OCP are read from ``controller.planner``.
            ``None`` (e.g. baseline random policy) writes the env/algo record only.
        algo: Algorithm tag -- ``"sac_zop"``, ``"sac_fop"`` or ``"baseline"``.
        reward_name: Reward scenario (R0..R3); ``resolve_reward`` rebuilds the cost.
        env_cfg: The environment config used for training.
        building_name: Key into ``BUILDING_NAMES2CLASS`` for ``env_cfg.building_params``.

    Returns:
        The path to the written ``ocp_solver_config.yaml``.
    """
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    planner = getattr(controller, "planner", None)

    record = {
        "schema_version": SCHEMA_VERSION,
        "algo": algo,
        "reward_name": reward_name,
        "controller": (
            {"name": "i4b", "planner": _planner_cfg_to_dict(planner.cfg)}
            if planner is not None
            else None
        ),
        "env": _env_recipe(env_cfg, building_name),
    }

    yaml_path = out / "ocp_solver_config.yaml"
    with open(yaml_path, "w") as f:
        safe_dump(record, f, sort_keys=False)

    # acados-native dump (best effort -- never fail a training run over a log).
    # ``dump_to_json`` writes to ``ocp.code_gen_opts.json_file``; point it at the
    # run directory. The OCP was made consistent when its batch solver was built.
    ocp = getattr(planner, "ocp", None) if planner is not None else None
    if ocp is not None:
        try:
            ocp.code_gen_opts.json_file = str(out / "acados_ocp.json")
            ocp.dump_to_json()
        except Exception as exc:  # noqa: BLE001 - logging must not break training
            warn(f"Could not dump acados_ocp.json: {exc}", RuntimeWarning, stacklevel=2)

    return yaml_path
