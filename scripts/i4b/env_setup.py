"""Configure and build the stochastic i4b environment from the CLI.

Centralizes the noise/uncertainty knobs added in commit 45e4adf (measurement noise,
process noise, randomised initial state, AR(1) forecast uncertainty on temperature + the
three solar channels, and a deterministic start-date override) so ``run_sac_fop`` /
``run_sac_zop`` / ``run_baseline`` and the ``run_ablation`` orchestrator expose and build
the environment identically -- mirroring the ``reward_setup.resolve_reward`` pattern.

The forecast levels are stored as strings (e.g. ``"negative_bias"``) on
``EnvStochasticityConfig`` so they serialize readably into the per-run config /
W&B logging; ``build_env_cfg`` resolves them to the concrete ``ForecastConfig`` AR(1)
objects via the existing ``ForecastConfig.__post_init__``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from i4b.gym_interface.env import BUILDING_NAMES2CLASS, Heatpump_AW, I4bEnvConfig
from i4b.gym_interface.forecast import ForecastConfig
from i4b.gym_interface.reward import RewardConfig

# Forecast-uncertainty preset levels. "none" disables the channel; the rest map to the
# hvac ForecastConfig presets resolved in ForecastConfig.__post_init__.
FORECAST_LEVELS = ("none", "low", "medium", "high", "negative_bias")


@dataclass
class EnvStochasticityConfig:
    """Noise/uncertainty knobs applied to the i4b environment.

    Attributes:
        noise_level: Std dev of observation-only measurement noise [degC].
        process_noise_std: Std dev of process noise on the true state per step [degC].
        random_init: Randomise the initial building state on reset.
        forecast_temp: Forecast-uncertainty level for the T_amb channel (FORECAST_LEVELS).
        forecast_solar: Forecast-uncertainty level for the solar channels dhi/ghi/dni.
        start_date: Deterministic single-window start override (None lets the split pick).
    """

    noise_level: float = 0.1
    process_noise_std: float = 0.02
    random_init: bool = True
    forecast_temp: str = "negative_bias"
    forecast_solar: str = "negative_bias"
    start_date: str | None = None


def add_env_noise_args(parser: argparse.ArgumentParser) -> None:
    """Register the noise/uncertainty CLI flags (defaults match EnvStochasticityConfig)."""
    d = EnvStochasticityConfig()
    group = parser.add_argument_group("Environment noise and uncertainty")
    group.add_argument(
        "--noise-level",
        type=float,
        default=d.noise_level,
        help="Std dev of observation-only measurement noise [degC] (0 disables).",
    )
    group.add_argument(
        "--process-noise-std",
        type=float,
        default=d.process_noise_std,
        help="Std dev of process noise on the true state per step [degC] (0 disables).",
    )
    group.add_argument(
        "--random-init",
        action=argparse.BooleanOptionalAction,
        default=d.random_init,
        help="Randomise the initial building state on reset.",
    )
    group.add_argument(
        "--forecast-temp-uncertainty",
        type=str,
        default=d.forecast_temp,
        choices=FORECAST_LEVELS,
        help="AR(1) forecast-uncertainty level for the T_amb forecast ('none' disables).",
    )
    group.add_argument(
        "--forecast-solar-uncertainty",
        type=str,
        default=d.forecast_solar,
        choices=FORECAST_LEVELS,
        help="AR(1) forecast-uncertainty level for the solar forecasts dhi/ghi/dni "
        "('none' disables).",
    )
    group.add_argument(
        "--start-date",
        type=str,
        default=d.start_date,
        help="Deterministic episode start window (e.g. '2023-01-15'); default lets the "
        "train/test split choose.",
    )


def stochasticity_from_args(args: argparse.Namespace) -> EnvStochasticityConfig:
    """Build an EnvStochasticityConfig from a namespace populated by add_env_noise_args."""
    return EnvStochasticityConfig(
        noise_level=args.noise_level,
        process_noise_std=args.process_noise_std,
        random_init=args.random_init,
        forecast_temp=args.forecast_temp_uncertainty,
        forecast_solar=args.forecast_solar_uncertainty,
        start_date=args.start_date,
    )


def resolve_forecast_noise(temp: str, solar: str) -> ForecastConfig | None:
    """Map per-channel level strings to a ForecastConfig (None = perfect foresight).

    "none" disables a channel; when both are off the whole forecast model is disabled
    (returns None). Otherwise the preset strings are passed straight to ForecastConfig,
    whose __post_init__ resolves them to the AR(1) parameter objects.
    """
    t = None if temp == "none" else temp
    s = None if solar == "none" else solar
    if t is None and s is None:
        return None
    return ForecastConfig(temp_uncertainty=t, solar_uncertainty=s)


def build_env_cfg(
    stoch: EnvStochasticityConfig,
    reward_cfg: RewardConfig,
    *,
    seed: int,
    days: int = 3,
) -> I4bEnvConfig:
    """Assemble the i4b env config with the chosen reward, seed and stochasticity knobs.

    Single source of truth replacing the duplicated I4bEnvConfig(...) blocks in the run
    scripts. ``seed`` pins the COFACTOR internal-gains draw so the disturbance is
    reproducible and varies per run.
    """
    return I4bEnvConfig(
        building_params=BUILDING_NAMES2CLASS["i4c"],
        hp_model=Heatpump_AW(mdot_HP=0.25),
        days=days,
        reward=reward_cfg,
        seed=seed,
        noise_level=stoch.noise_level,
        process_noise_std=stoch.process_noise_std,
        random_init=stoch.random_init,
        forecast_noise=resolve_forecast_noise(stoch.forecast_temp, stoch.forecast_solar),
        start_date=stoch.start_date,
    )


def env_noise_to_cli(stoch: EnvStochasticityConfig) -> list[str]:
    """Emit the CLI flags for a subprocess run script (inverse of add_env_noise_args)."""
    flags = [
        "--noise-level",
        str(stoch.noise_level),
        "--process-noise-std",
        str(stoch.process_noise_std),
        "--random-init" if stoch.random_init else "--no-random-init",
        "--forecast-temp-uncertainty",
        stoch.forecast_temp,
        "--forecast-solar-uncertainty",
        stoch.forecast_solar,
    ]
    if stoch.start_date is not None:
        flags += ["--start-date", stoch.start_date]
    return flags
