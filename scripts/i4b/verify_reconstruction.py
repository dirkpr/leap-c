"""Rebuild an i4b policy from a run's logged outputs and reproduce its eval.

This is the inverse of ``scripts/i4b/ocp_logging.py``: given a single seed
output directory containing ``config.yaml`` (trainer hyperparameters),
``ocp_solver_config.yaml`` (OCP/env rebuild recipe) and ``ckpts/last_pi.ckpt``
(trained actor weights), it reconstructs the exact ``HierachicalMPCActor`` that
was used at training time and runs a deterministic validation rollout.

Used both as a library (``rebuild_trainer_from_output`` /
``reconstructed_score``) by the reconstruction test and as a CLI:

    python scripts/i4b/verify_reconstruction.py <run_dir>/sac_zop/R0/seed_0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from tempfile import mkdtemp

import torch
from reward_setup import resolve_reward
from run_baseline import BaselineTrainer, BaselineTrainerConfig
from trainer import I4bSacFopTrainer, I4bSacZopTrainer
from yaml import safe_load

from leap_c.examples import create_controller
from leap_c.examples.hvac.forecast import (
    ForecastConfig,
    SolarUncertaintyConfig,
    TemperatureUncertaintyConfig,
)
from leap_c.examples.i4b.env import (
    BUILDING_NAMES2CLASS,
    Heatpump_AW,
    Heatpump_Vitocal,
    I4bEnv,
    I4bEnvConfig,
)
from leap_c.torch.rl.sac_fop import SacFopTrainerConfig
from leap_c.torch.rl.sac_zop import SacZopTrainerConfig
from leap_c.utils.cfg import update_dataclass_from_dict

# algo -> (trainer class, trainer-config class, default eval dtype).
TRAINER_BY_ALGO = {
    "sac_zop": (I4bSacZopTrainer, SacZopTrainerConfig, torch.float32),
    "sac_fop": (I4bSacFopTrainer, SacFopTrainerConfig, torch.float32),
    "baseline": (BaselineTrainer, BaselineTrainerConfig, torch.float64),
}

_HP_MODELS = {"Heatpump_AW": Heatpump_AW, "Heatpump_Vitocal": Heatpump_Vitocal}

# Sentinel distinguishing "forecast_noise not in the recipe" (pre-2026-06-09 logs)
# from "forecast_noise recorded as null" (perfect foresight).
_MISSING = object()


def _forecast_noise_from_recipe(value) -> ForecastConfig | None:
    """Rebuild ``forecast_noise`` from its recipe entry.

    ``value`` is the recorded ``asdict(ForecastConfig)`` dict, ``None`` (forecast noise
    explicitly disabled), or ``_MISSING`` for logs predating its recording (fall back to
    the ``I4bEnvConfig`` default — the negative_bias preset that was active then). The
    recorded per-channel entries are already-resolved AR(1) parameter dicts (or ``None``),
    which ``ForecastConfig`` passes through untouched.
    """
    if value is _MISSING:
        return ForecastConfig()
    if value is None:
        return None
    temp = value.get("temp_uncertainty")
    solar = value.get("solar_uncertainty")
    return ForecastConfig(
        horizon_hours=value.get("horizon_hours", 24),
        temp_uncertainty=TemperatureUncertaintyConfig(**temp) if temp is not None else None,
        solar_uncertainty=SolarUncertaintyConfig(**solar) if solar is not None else None,
    )


def _env_cfg_from_recipe(env: dict, reward_cfg) -> I4bEnvConfig:
    """Rebuild an ``I4bEnvConfig`` from the ``env`` block of ocp_solver_config.yaml."""
    hp_cls = _HP_MODELS[env["hp_model"]]
    return I4bEnvConfig(
        building_params=BUILDING_NAMES2CLASS[env["building_name"]],
        hp_model=hp_cls(mdot_HP=env["mdot_hp"]),
        method=env["method"],
        mdot_hp=env["mdot_hp"],
        delta_t=env["delta_t"],
        days=env["days"],
        random_init=env["random_init"],
        noise_level=env["noise_level"],
        # New fields default to the I4bEnvConfig defaults so pre-existing run logs
        # (written before these were added) still reconstruct.
        process_noise_std=env.get("process_noise_std", 0.0),
        data_mode=env.get("data_mode", "random"),
        valid_months=env.get("valid_months", [1, 2, 12]),
        total_test_episodes=env.get("total_test_episodes", 16),
        split_seed=env.get("split_seed", 42),
        T_set_lower=env["T_set_lower"],
        T_set_upper=env["T_set_upper"],
        N_forecast=env["N_forecast"],
        forecast_noise=_forecast_noise_from_recipe(env.get("forecast_noise", _MISSING)),
        grid_signal=env["grid_signal"],
        apply_heating_logic=env["apply_heating_logic"],
        start_date=env["start_date"],
        seed=env["seed"],
        reward=reward_cfg,
    )


def rebuild_trainer_from_output(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    reuse_code_dir: Path | None = None,
):
    """Rebuild the trainer (and its ``HierachicalMPCActor``) from a run directory.

    Reads ``ocp_solver_config.yaml`` (algo + OCP/env recipe) and ``config.yaml``
    (trainer hyperparameters), then reconstructs the controller, the environment
    and the trainer exactly as they were built for training. The trainer's
    loggers are disabled and a throwaway ``output_path`` is used so reconstruction
    does not touch the original run's files.
    """
    run_dir = Path(run_dir)
    rec = safe_load((run_dir / "ocp_solver_config.yaml").read_text())
    cfg_dict = safe_load((run_dir / "config.yaml").read_text())

    algo = rec["algo"]
    reward_name = rec["reward_name"]
    trainer_cls, cfg_cls, default_dtype = TRAINER_BY_ALGO[algo]
    dtype = default_dtype if dtype is None else dtype

    trainer_cfg = update_dataclass_from_dict(cfg_cls(), cfg_dict)
    # Reconstruction only needs the deterministic eval score, not new logs.
    trainer_cfg.log.wandb_logger = False
    trainer_cfg.log.csv_logger = False
    trainer_cfg.log.tensorboard_logger = False

    reward_cfg = resolve_reward(reward_name)
    env_cfg = _env_cfg_from_recipe(rec["env"], reward_cfg)
    controller = create_controller("i4b", reuse_code_dir, reward=reward_cfg)

    if output_path is None:
        output_path = mkdtemp(prefix="i4b_recon_")

    val_env = I4bEnv(cfg=env_cfg)
    if algo == "baseline":
        trainer = trainer_cls(
            cfg=trainer_cfg,
            val_env=val_env,
            output_path=output_path,
            device=device,
            dtype=dtype,
            policy_type="controller",
            controller=controller,
            train_env=None,
        )
    else:
        trainer = trainer_cls(
            val_env=val_env,
            train_env=I4bEnv(cfg=env_cfg),
            controller=controller,
            output_path=output_path,
            device=device,
            dtype=dtype,
            cfg=trainer_cfg,
        )
    return trainer


def reconstructed_score(
    run_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    reuse_code_dir: Path | None = None,
) -> float:
    """Rebuild the policy from ``run_dir`` and return its deterministic eval score.

    For SAC runs the trained actor weights are loaded from ``ckpts/last_pi.ckpt``.
    Baseline runs carry no learned weights -- the policy is the OCP controller
    with default parameters, fully determined by the logged OCP config.
    """
    trainer = rebuild_trainer_from_output(
        run_dir, device=device, dtype=dtype, reuse_code_dir=reuse_code_dir
    )

    ckpt_dir = Path(run_dir) / "ckpts"
    pi_ckpt = ckpt_dir / "last_pi.ckpt"
    if pi_ckpt.exists() and hasattr(trainer, "pi"):
        trainer.pi.load_state_dict(torch.load(pi_ckpt, weights_only=False))
        state_ckpt = ckpt_dir / "last_trainer_state.ckpt"
        if state_ckpt.exists():
            trainer.state = torch.load(state_ckpt, weights_only=False)

    trainer.eval()
    with torch.inference_mode():
        return trainer.validate()


def _logged_final_score(run_dir: Path) -> float | None:
    """Return the score of the last row of ``val_log.csv`` if present."""
    val_log = run_dir / "val_log.csv"
    if not val_log.exists():
        return None
    with open(val_log) as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]["score"]) if rows else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="A single seed output directory.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--reuse-code-dir", type=Path, default=None)
    args = parser.parse_args()

    score = reconstructed_score(
        args.run_dir, device=args.device, reuse_code_dir=args.reuse_code_dir
    )
    print(f"Reconstructed deterministic eval score: {score:.6f}")

    logged = _logged_final_score(args.run_dir)
    if logged is not None:
        print(f"Logged final val_log.csv score:        {logged:.6f}")
        print(f"Difference:                            {abs(score - logged):.3e}")


if __name__ == "__main__":
    main()
