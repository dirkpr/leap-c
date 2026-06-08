"""Gymnasium environment for i4b building heat-pump control.

Wraps the i4b simulator (external/i4b/src/simulator.py) and exposes the same
building/HP model objects used in acados_ocp.py, ensuring the environment
dynamics and cost match the OCP exactly.

OCP parameter alignment (p per stage):
    p[0] = T_set_lower   [degC]
    p[1] = T_set_upper   [degC]
    p[2] = grid_signal   [-]
    p[3] = T_amb         [degC]
    p[4] = Qdot_gains    [W]

Available building models (building_params dicts from i4b_data/buildings/):
    sfh_1919_1948 … sfh_2016_now  (0_soc, 1_enev, 2_kfw variants)
    i4c

Available building methods: "2R2C", "4R3C", "5R4C"
Available HP models: Heatpump_AW, Heatpump_Vitocal

Observation space (spaces.Dict):
    "state":          Box(nx,)   – building thermal states [degC]
    "disturbances":   Dict
        "T_amb":      Box(1,)    – current ambient temperature [degC]
    "setpoints":      Dict
        "T_set_lower": Box(1,)   – lower comfort bound [degC]
        "T_set_upper": Box(1,)   – upper comfort bound [degC]
    "forecast":       Dict       – only if weather_forecast_steps is non-empty
        "T_amb":      Box(nf,)   – ambient temperature forecast [degC]

The true total heat gains ``Qdot_gains`` are deliberately *not* part of the
observation: they are the per-stage quantity the policy must learn to predict, so
exposing them would let the feature extractor leak the answer into the parameters.
The true value is published in the ``step()`` ``info`` dict instead (out-of-band,
never seen by the actor) for diagnostics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from i4b.disturbances import get_solar_gains
from i4b.gym_interface import BUILDING_NAMES2CLASS
from i4b.gym_interface.constant import OBSERVATION_SPACE_LIMIT
from i4b.models.model_buildings import Building
from i4b.models.model_hvac import Heatpump, Heatpump_AW, Heatpump_Vitocal  # noqa: F401
from i4b.simulator import Model_simulator

from leap_c.examples.hvac.dataset import DataConfig, HvacDataset, load_and_prepare_data
from leap_c.examples.hvac.forecast import ForecastConfig, predict_ar1_error
from leap_c.examples.i4b.reward import RewardConfig, compute_reward

# Action bounds for T_HP supply temperature [degC], shared with the planner.
_T_HP_ACT_LOW: float = OBSERVATION_SPACE_LIMIT["T_hp_sup"][0]
_T_HP_ACT_HIGH: float = OBSERVATION_SPACE_LIMIT["T_hp_sup"][1]

# Re-export for convenience
__all__ = [
    "I4bEnvConfig",
    "I4bEnv",
    "BUILDING_NAMES2CLASS",
    "Building",
    "Heatpump_AW",
    "Heatpump_Vitocal",
]

# COFACTOR per-apartment electricity dataset (hourly, area-normalized W/m^2)
_ELIMP_PARQUET = Path(__file__).parent / "assets" / "elimp_per_apartment.parquet"

# Scaling factor to convert the COFACTOR electricity proxy to internal heat gains,
# set so the peak is in the magnitude of the DIN EN 16798-1 ResidentialFlat profile
# for the i4c archetype
# Source: external/i4b/i4b_data/profiles/InternalGains/ResidentialFlat.csv
_DIN_INT_GAIN_PEAK_W_PER_M2 = 10.0

# Allowed +/-10% random deviation applied to the scaled internal-gain peak.
_INT_GAIN_PEAK_JITTER = 0.10


@dataclass(kw_only=True)
class I4bEnvConfig:
    """Configuration for I4bEnv.

    Attributes:
        building_params: Building parameter dict (e.g. sfh_2016_now_0_soc from
            BUILDING_NAMES2CLASS or imported directly from i4b_data/buildings/).
        hp_model: Instantiated heat pump model. Same object can be passed to
            export_parametric_ocp to guarantee dynamics/cost consistency.
        method: Building thermal model type. One of "2R2C", "4R3C", "5R4C".
        mdot_hp: Mass flow rate of the HP system [kg/s].
        delta_t: Simulation timestep [s].
        days: Episode length in days. None uses full weather dataset.
        random_init: Randomise the initial building state on reset (sampled from the
            observation-space limits). The episode *start time* is chosen by the
            dataset split (see ``data_mode``/``valid_months``), independent of this flag.
        noise_level: Std dev of Gaussian measurement noise added to the building state
            observation in step(). 0 disables it.
        process_noise_std: Std dev of Gaussian process noise [degC] added to each
            building state after the simulator integration in step(). 0 disables it.
        data_mode: Dataset sampling mode passed to the HvacDataset split machinery.
            "random" samples a fixed-length window per episode (train excludes the
            stratified test windows); "continual" walks the dataset sequentially.
        valid_months: Months (1-12) the start time may fall in (heating season by
            default). None allows all months.
        total_test_episodes: Number of fixed, stratified (year, month) test windows
            reserved for evaluation. Should be >= the validation rollout count so eval
            windows don't repeat within a pass. 0 disables the split (eval samples freely).
        split_seed: Seed for the reproducible train/test split.
        T_set_lower: Default lower comfort temperature setpoint [degC]. Overridden
            by time-varying values from get_temperature_limits if using dynamic setpoints.
        T_set_upper: Default upper comfort temperature [degC].
        N_forecast: Number of future steps with available weather/setpoint forecasts
            in the "forecast" observation dict. Should match the MPC horizon for best results,
            but can be set to 0 to disable the "forecast" part of the observation.
        forecast_noise: hvac-style AR(1) forecast-error model. Defaults to a
            ``ForecastConfig()`` (the hvac ``negative_bias`` preset), so the ``T_amb``
            forecast gets Normal AR(1) noise and the solar forecasts (``dhi``/``ghi``/``dni``)
            get Laplace AR(1) noise, drawn fresh each step from the env RNG; the true
            disturbance stepping the dynamics stays unperturbed. Set to None for perfect
            foresight (exact future dataset values). ``ForecastConfig.horizon_hours`` is
            ignored (the horizon is governed by ``N_forecast``); only ``temp_uncertainty``
            and ``solar_uncertainty`` are used.
        grid_signal: Deprecated for the reward. Price weighting now flows through
            ``reward`` (RewardConfig); this field is kept for backward compatibility
            but no longer multiplies the reward.
        reward: Reward configuration (single source of truth, shared with the
            planner). Defaults to R0 (energy-only, ``r = -E_k``) -- the historical
            i4b reward when ``grid_signal == 1``.
        seed: If set, seeds the random COFACTOR apartment draw in
            _augment_dataset so the internal-gain disturbance is reproducible.
            None draws from entropy.
    """

    building_params: dict
    hp_model: Heatpump
    method: str = "4R3C"
    mdot_hp: float = 0.25
    delta_t: int = 900
    days: int = 3
    random_init: bool = False
    noise_level: float = 0.0
    process_noise_std: float = 0.0
    data_mode: Literal["random", "continual"] = "random"
    valid_months: list[int] | None = field(default_factory=lambda: [1, 2, 12])
    total_test_episodes: int = 16
    split_seed: int = 42
    T_set_lower: float = 20.0
    T_set_upper: float = 26.0
    N_forecast: int = 24 * 4
    forecast_noise: ForecastConfig | None = field(default_factory=ForecastConfig)
    grid_signal: float = 1.0
    reward: RewardConfig = field(default_factory=RewardConfig)
    seed: int | None = None
    apply_heating_logic: bool = False
    start_date: str | None = None
    """If True, apply the legacy RoomHeatEnv heating logic in step(): add T_offset when
    T_amb < T_amb_lim, fall back to T_hp_ret when warm, and clip via check_hp().
    If False (default), the denormalised MPC action is passed directly to the simulator,
    matching the OCP model."""


class I4bEnv(gym.Env):
    """Gymnasium environment for building heat-pump control using the i4b simulator.

    The observation is a ``spaces.Dict`` — see module docstring for layout.

    The action is a normalised supply temperature in [-1, 1], mapped linearly
    to T_HP in [0, 65] degC (matching the OCP control bounds).

    The reward is configurable via ``cfg.reward`` (a ``RewardConfig``); see
    ``leap_c/examples/i4b/reward.py``. It defaults to R0 (energy-only,
    ``r = -E_el_kWh``) and is co-designed with the OCP stage cost so the env and the
    planner optimize the same objective. The per-term decomposition is published in
    ``info["reward_terms"]``.

    The method `get_ocp_parameters(t)` returns the OCP parameter vector
    p = [T_amb, Qdot_gains, T_set_lower, T_set_upper, grid_signal] at timestep t,
    matching the parameter layout expected by export_parametric_ocp.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        render_mode: str | None = None,
        cfg: I4bEnvConfig | None = None,
        dataset: HvacDataset | None = None,
    ):
        """Initialise the environment.

        Args:
            render_mode: Unused; kept for Gymnasium compatibility.
            cfg: Environment configuration. Uses default i4c /
                Heatpump_AW / 4R3C if None.
            dataset: Pre-built HvacDataset with weather and price data. When
                None a default dataset is loaded from local CSV assets.
        """
        super().__init__()

        if cfg is None:
            cfg = I4bEnvConfig(
                building_params=BUILDING_NAMES2CLASS["i4c"],
                hp_model=Heatpump_AW(mdot_HP=0.25),
            )

        self.cfg = cfg

        # ── Build models ──────────────────────────────────────────────────────
        self.bldg_model = Building(
            params=cfg.building_params,
            mdot_hp=cfg.mdot_hp,
            method=cfg.method,
            T_room_set_lower=cfg.T_set_lower,
            T_room_set_upper=cfg.T_set_upper,
        )
        self.hp_model = cfg.hp_model
        self.simulator = Model_simulator(
            hp_model=self.hp_model,
            bldg_model=self.bldg_model,
            timestep=cfg.delta_t,
        )

        # ── Dataset ───────────────────────────────────────────────────────────
        self.dataset = dataset if dataset is not None else self._create_default_dataset()
        self._augment_dataset()

        # ── Spaces ────────────────────────────────────────────────────────────
        self.state_keys = self.bldg_model.state_keys  # e.g. ("T_room","T_wall","T_hp_ret")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = self._make_obs_space()

        # ── Episode state ─────────────────────────────────────────────────────
        self._idx = 0  # current index into dataset
        self.step_counter = 0
        self.max_steps = 0
        self.state: dict | None = None

        self.reset()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _create_default_dataset(self) -> HvacDataset:
        """Load a default HvacDataset from local CSV assets.

        The dataset is configured with the train/test split machinery so that
        ``reset()`` can sample non-overlapping train and (fixed, stratified) test
        windows. ``start_date`` is folded into ``DataConfig.start_time`` (snapped to
        the nearest grid point) as the deterministic single-window escape hatch.
        """
        cfg = self.cfg
        data = load_and_prepare_data(
            price_zone="DE-LU",
            price_data_path=Path(__file__).parent / "assets" / "price.csv",
            weather_data_path=Path(__file__).parent / "assets" / "weather.csv",
        )
        if cfg.days is not None:
            max_hours = int(cfg.days * 24)
        else:
            # Full-dataset episode: a single long window from the first valid index.
            max_hours = int((len(data) - cfg.N_forecast - 1) / 4)

        start_time = None
        if cfg.start_date is not None:
            pos = int(data.index.searchsorted(pd.Timestamp(cfg.start_date, tz="UTC")))
            start_time = data.index[min(pos, len(data) - 1)]

        data_cfg = DataConfig(
            mode=cfg.data_mode,
            max_hours=max_hours,
            valid_months=cfg.valid_months,
            total_test_episodes=cfg.total_test_episodes,
            split_seed=cfg.split_seed,
            start_time=start_time,
        )
        return HvacDataset(data=data, cfg=data_cfg)

    def _augment_dataset(self) -> None:
        """Compute building-specific disturbances and write them to the dataset.

        Adds ``Qdot_gains``, ``T_set_lower``, and ``T_set_upper`` columns to
        ``self.dataset`` using the full dataset index so that any episode start
        can be served from the pre-computed arrays.
        """
        cfg = self.cfg
        idx = self.dataset.data.index

        weather = self.dataset.data[
            [
                "temperature_2m",
                "diffuse_radiation",
                "shortwave_radiation",
                "direct_normal_irradiance",
            ]
        ].rename(
            columns={
                "temperature_2m": "T_amb",
                "diffuse_radiation": "dhi",
                "shortwave_radiation": "ghi",
                "direct_normal_irradiance": "dni",
            }
        )

        T_set_lower, T_set_upper = get_temperature_limits(idx)
        elimp = pd.read_parquet(_ELIMP_PARQUET)
        rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else self.np_random
        int_gains = get_int_gains(idx, elimp, bldg_area=cfg.building_params["area_floor"], rng=rng)
        Qdot_sol: pd.Series = get_solar_gains(weather=weather, bldg_params=cfg.building_params)

        self.dataset.add_columns(
            {
                "Qdot_gains": (Qdot_sol + int_gains).to_numpy(dtype=np.float32),
                "Qdot_int_tot": int_gains.to_numpy(dtype=np.float32),
                "Qdot_sol": Qdot_sol.to_numpy(dtype=np.float32),
                "T_set_lower": T_set_lower.astype(np.float32),
                "T_set_upper": T_set_upper.astype(np.float32),
            }
        )

    def _make_obs_space(self) -> spaces.Dict:
        state_lows = np.array(
            [OBSERVATION_SPACE_LIMIT[k][0] for k in self.state_keys], dtype=np.float32
        )
        state_highs = np.array(
            [OBSERVATION_SPACE_LIMIT[k][1] for k in self.state_keys], dtype=np.float32
        )
        T_amb_lo = np.float32(OBSERVATION_SPACE_LIMIT["T_amb"][0])
        T_amb_hi = np.float32(OBSERVATION_SPACE_LIMIT["T_amb"][1])

        obs_spaces: dict = {
            "state": spaces.Box(low=state_lows, high=state_highs, dtype=np.float32),
            "disturbances": spaces.Dict(
                {
                    "T_amb": spaces.Box(low=T_amb_lo, high=T_amb_hi, shape=(1,), dtype=np.float32),
                }
            ),
            "setpoints": spaces.Dict(
                {
                    # low 12.0 = night setback lower bound from get_temperature_limits.
                    "T_set_lower": spaces.Box(
                        low=np.float32(12.0), high=np.float32(30.0), shape=(1,), dtype=np.float32
                    ),
                    "T_set_upper": spaces.Box(
                        low=np.float32(20.0), high=np.float32(35.0), shape=(1,), dtype=np.float32
                    ),
                }
            ),
        }
        nf = self.cfg.N_forecast
        obs_spaces["forecast"] = spaces.Dict(
            {
                "T_amb": spaces.Box(low=T_amb_lo, high=T_amb_hi, shape=(nf,), dtype=np.float32),
                "T_set_lower": spaces.Box(
                    low=np.float32(12.0), high=np.float32(30.0), shape=(nf,), dtype=np.float32
                ),
                "T_set_upper": spaces.Box(
                    low=np.float32(20.0), high=np.float32(35.0), shape=(nf,), dtype=np.float32
                ),
                "quarter_hour": spaces.Box(low=0, high=int(24 * 4 - 1), shape=(nf,), dtype=int),
                "day_of_year": spaces.Box(low=0, high=365, shape=(nf,), dtype=int),
                "day_of_week": spaces.Box(low=0, high=6, shape=(nf,), dtype=int),
                "dhi": spaces.Box(low=0, high=np.float32(2000.0), shape=(nf,), dtype=np.float32),
                "ghi": spaces.Box(low=0, high=np.float32(2000.0), shape=(nf,), dtype=np.float32),
                "dni": spaces.Box(low=0, high=np.float32(2000.0), shape=(nf,), dtype=np.float32),
                # price in €/kWh
                "price": spaces.Box(low=0, high=np.float32(10.0), shape=(nf,), dtype=np.float32),
            }
        )
        return spaces.Dict(obs_spaces)

    def _build_obs(self, state_dict: dict) -> dict:
        # All arrays in self.dataset._arrays are pre-cast to float32 at build
        # time so no dtype conversions are needed here.
        nf = self.cfg.N_forecast
        gcv = self.dataset.get_column_view  # zero-copy view, raises on out-of-bounds
        obs: dict = {
            "state": np.array([state_dict[k] for k in self.state_keys], dtype=np.float32),
            "disturbances": {
                "T_amb": gcv("temperature_2m", self._idx),
            },
            "setpoints": {
                "T_set_lower": gcv("T_set_lower", self._idx),
                "T_set_upper": gcv("T_set_upper", self._idx),
            },
        }
        if nf:
            obs["forecast"] = {
                "T_amb": gcv("temperature_2m", self._idx, nf),
                "T_set_lower": gcv("T_set_lower", self._idx, nf),
                "T_set_upper": gcv("T_set_upper", self._idx, nf),
                "quarter_hour": gcv("quarter_hour", self._idx, nf),
                "day_of_year": gcv("day_of_year", self._idx, nf),
                "day_of_week": gcv("day_of_week", self._idx, nf),
                "dhi": gcv("diffuse_radiation", self._idx, nf),
                "ghi": gcv("shortwave_radiation", self._idx, nf),
                "dni": gcv("direct_normal_irradiance", self._idx, nf),
                "price": gcv("price", self._idx, nf),
            }
            if self.cfg.forecast_noise is not None:
                self._add_forecast_noise(obs["forecast"])
        return obs

    def _add_forecast_noise(self, fc: dict) -> None:
        """Perturb the forecast dict in place with hvac-style AR(1) forecast error.

        ``T_amb`` gets Normal AR(1) noise and each solar component
        (``dhi``/``ghi``/``dni``) gets an independent Laplace AR(1) draw, clamped to
        ``>= 0``. New arrays are created so the zero-copy dataset views are not mutated.
        The true disturbance stepping the dynamics is left untouched -- only the forecast
        the controller plans against is noisy.
        """
        nf = self.cfg.N_forecast
        cfg = self.cfg.forecast_noise

        tu = cfg.temp_uncertainty
        if tu is not None:
            err = predict_ar1_error(
                hp=nf,
                initial_mean=tu.F0,
                initial_scale=tu.K0,
                ar_factor=tu.F,
                ar_mean=tu.mu,
                ar_scale=tu.K,
                np_random=self.np_random,
                distribution="normal",
            )
            fc["T_amb"] = (fc["T_amb"] + err).astype(np.float32)

        su = cfg.solar_uncertainty
        if su is not None:
            for key in ("dhi", "ghi", "dni"):
                err = predict_ar1_error(
                    hp=nf,
                    initial_mean=su.ag0,
                    initial_scale=su.bg0,
                    ar_factor=su.phi,
                    ar_mean=su.ag,
                    ar_scale=su.bg,
                    np_random=self.np_random,
                    distribution="laplace",
                )
                fc[key] = np.maximum(0.0, fc[key] + err).astype(np.float32)

    def _copy_obs(self, obs: dict) -> dict:
        """Return a shallow-copied dict obs (arrays copied, nested dicts recursed)."""
        return {k: self._copy_obs(v) if isinstance(v, dict) else v.copy() for k, v in obs.items()}

    def _denorm_action(self, a: np.ndarray) -> float:
        """Map normalised action in [-1, 1] to T_HP in [degC]."""
        mid = (_T_HP_ACT_HIGH + _T_HP_ACT_LOW) / 2
        half = (_T_HP_ACT_HIGH - _T_HP_ACT_LOW) / 2
        return float(np.clip(a, -1.0, 1.0).flat[0] * half + mid)

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        """Advance one timestep.

        Args:
            action: Normalised supply temperature in [-1, 1].

        Returns:
            obs, reward, terminated, truncated, info
        """
        state_dict = dict(zip(self.state_keys, self.state["state"]))
        pk_dict = {
            "T_amb": float(self.dataset.get_column("temperature_2m", self._idx)),
            "Qdot_gains": float(self.dataset.get_column("Qdot_gains", self._idx)),
        }

        # Reward signals at the current index k, captured *before* the _idx increment
        # below: the interval price pi_k pairs with the energy E_k, and the
        # start-of-step room/bound feed the shaping potential Phi(s_k).
        price_k = float(self.dataset.get_column("price", self._idx))
        T_room_prev = float(state_dict["T_room"])
        T_set_lower_prev = float(self.dataset.get_column("T_set_lower", self._idx))

        T_hp_sup = self._denorm_action(action)

        if self.cfg.apply_heating_logic:
            # Legacy RoomHeatEnv behaviour: add T_offset when cold outside, fall back to
            # T_hp_ret when ambient is warm, then clip via check_hp.
            if pk_dict["T_amb"] < self.bldg_model.params["T_amb_lim"]:
                T_hp_sup = max(
                    T_hp_sup + self.bldg_model.params["T_offset"],
                    state_dict["T_hp_ret"],
                )
            else:
                T_hp_sup = state_dict["T_hp_ret"]
            T_hp_sup = self.hp_model.check_hp(T_hp_sup, state_dict["T_hp_ret"])

        res = self.simulator.get_next_state(state_dict, T_hp_sup, pk_dict)
        next_state = res["state"]
        # Process noise: perturb each integrated building state [degC] so the realized
        # trajectory (and thus the comfort the occupant experiences) is stochastic.
        # Applied to the true state, separate from the observation-only measurement noise.
        if self.cfg.process_noise_std > 0:
            for k in self.state_keys:
                next_state[k] += float(self.np_random.normal(0, self.cfg.process_noise_std))
        costs = res["cost"]
        E_el_kWh = float(costs["E_el"]) / 1000.0

        self._idx += 1
        self.step_counter += 1
        self.state = self._build_obs(next_state)

        # Comfort deviations at the arrival index k+1 (self._idx now == k+1) against
        # the realized room temperature. Computed directly from the dynamic dataset
        # comfort band -- NOT costs["dev_neg_*"], which use the static 20 degC bound.
        T_room_next = float(next_state["T_room"])
        T_set_lower_kp1 = float(self.dataset.get_column("T_set_lower", self._idx))
        T_set_upper_kp1 = float(self.dataset.get_column("T_set_upper", self._idx))
        d_k = max(0.0, T_set_lower_kp1 - T_room_next)
        o_k = max(0.0, T_room_next - T_set_upper_kp1)

        reward, reward_terms = compute_reward(
            self.cfg.reward,
            E_k=E_el_kWh,
            price_k=price_k,
            d_k=d_k,
            o_k=o_k,
            T_room=T_room_next,
            T_set_lower=T_set_lower_kp1,
            T_room_prev=T_room_prev,
            T_set_lower_prev=T_set_lower_prev,
        )
        truncated = (
            self._idx + self.cfg.N_forecast >= len(self.dataset)
            or self.step_counter >= self.max_steps
        )

        info = {
            "cost": float(costs["dev_neg_max"]),
            "E_el_kWh": E_el_kWh,
            "dev_sum": float(costs["dev_neg_sum"]),
            "dev_max": float(costs["dev_neg_max"]),
            "T_room": float(next_state["T_room"]),
            "T_hp_sup": T_hp_sup,
            # True heat gains, published out-of-band (not in the observation, so the
            # actor can't see them) for the predicted-vs-true Qdot_gains diagnostic.
            "Qdot_gains": pk_dict["Qdot_gains"],
            "t": self._idx,
            # Per-term reward decomposition (energy/comfort/overheat/shaping/total
            # plus E, price, cost, deviations, grid_signal) for diagnostics/plots.
            "reward_terms": reward_terms,
        }

        obs = self._copy_obs(self.state)
        if self.cfg.noise_level > 0:
            obs["state"] += self.np_random.normal(
                0, self.cfg.noise_level, obs["state"].shape
            ).astype(np.float32)

        return obs, float(reward), False, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)

        # Select the episode start window via the dataset split. ``options["mode"]
        # == "train"`` (set by the SAC loops) samples from the train split, excluding
        # the fixed stratified test windows; everything else (e.g. validation rollouts,
        # which pass no options) draws the reproducible test windows in order.
        if options is not None and options.get("mode") == "train":
            split = "train"
        else:
            split = "test"
            # Reset the test cursor so each seeded validation pass replays the same
            # stratified test windows in the same order.
            if self.dataset.cfg.mode == "random" and seed is not None:
                self.dataset.reset_test_counter()

        self._idx, self.max_steps = self.dataset.sample_start_index(
            rng=self.np_random,
            horizon=self.cfg.N_forecast,
            split=split,
        )

        self.step_counter = 0

        if self.cfg.random_init:
            state_dict = {
                k: float(
                    self.np_random.uniform(
                        OBSERVATION_SPACE_LIMIT[k][0],
                        OBSERVATION_SPACE_LIMIT[k][1],
                    )
                )
                for k in self.state_keys
            }
        else:
            state_dict = {k: self.cfg.T_set_lower for k in self.state_keys}

        self.state = self._build_obs(state_dict)
        return self._copy_obs(self.state), {}

    def plot_dataset(self, days: int = 7, start: int = 0):
        """Return a plotly Figure of ``days`` days of every signal in ``self.dataset``.

        Stacked, x-shared subplots (one per numeric column) over the window
        ``[start, start + days * steps_per_day)``, for quick visual inspection. The
        caller decides what to do with it, e.g. ``fig.show()`` or
        ``fig.write_html(path)``.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        steps = days * 24 * int(3600 / self.cfg.delta_t)
        df = self.dataset.data.iloc[start : start + steps]
        skip = {"time", "date"}
        cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
        fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, subplot_titles=cols)
        for row, c in enumerate(cols, start=1):
            fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c, mode="lines"), row=row, col=1)
        fig.update_layout(
            height=160 * len(cols),
            showlegend=False,
            title=f"i4b dataset — {days} day(s) from index {start}",
        )
        return fig


def get_temperature_limits(
    time: pd.DatetimeIndex,
    night_start_hour: int = 22,
    night_end_hour: int = 8,
    lb_night: float = 12.0,
    lb_day: float = 17.0,
    ub_night: float = 21.0,
    ub_day: float = 21.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Get temperature limits based on the time of day."""
    hours = time.hour
    night_idx = (hours >= night_start_hour) | (hours < night_end_hour)
    lb = np.where(night_idx, lb_night, lb_day)
    ub = np.where(night_idx, ub_night, ub_day)
    return lb, ub


def get_int_gains(
    time: pd.DatetimeIndex,
    elimp: pd.DataFrame,
    bldg_area: float,
    peak_w_per_m2: float = _DIN_INT_GAIN_PEAK_W_PER_M2,
    jitter: float = _INT_GAIN_PEAK_JITTER,
    rng=None,
) -> pd.Series:
    """Draw a random COFACTOR apartment-electricity channel as indoor heat gain [W].

    ``elimp`` is the hourly, area-normalized apartment electricity-import frame
    [W/m^2] (one column per apartment, tz-aware). A random column is picked and
    clock-re-based onto ``time``: the channel is shifted by a whole number of days
    so its wall-clock hour-of-day is preserved while the absolute year/season is
    ignored (COFACTOR guidance: condition the gain on local clock hour only, since
    leap-c uses different weather). The hourly series is then linearly interpolated
    onto ``time``.

    The raw COFACTOR proxy peaks far above realistic internal gains, so its shape is
    kept but its peak magnitude is max-scaled to the DIN EN 16798-1 peak,
    ``peak_w_per_m2 * bldg_area`` [W], with a +/-``jitter`` fractional random
    deviation drawn from ``rng``.

    Returns a ``pd.Series`` of ``Qdot_int`` [W] indexed by ``time``.
    """
    rng = np.random.default_rng(rng)
    col = str(rng.choice(elimp.columns.to_numpy()))
    hourly = elimp[col].copy()

    # Work in naive wall-clock; a whole-day shift keeps hour-of-day, drops year/season.
    hourly.index = hourly.index.tz_localize(None)
    target = time.tz_localize(None) if time.tz is not None else time
    shift_days = (target[0].normalize() - hourly.index[0].normalize()).days
    hourly.index = hourly.index + pd.Timedelta(days=shift_days)

    signal = (
        hourly.reindex(hourly.index.union(target))
        .interpolate("time", limit_direction="both")
        .reindex(target)
    )
    signal.index = time

    # Max-scale the COFACTOR proxy so its peak matches the DIN EN 16798-1 peak
    # (peak_w_per_m2 * bldg_area), with a +/-`jitter` random deviation from rng.
    peak_factor = 1.0 + rng.uniform(-jitter, jitter)
    target_peak = peak_w_per_m2 * float(bldg_area) * peak_factor
    denom = float(signal.max())
    scale = target_peak / denom if denom > 0 else 0.0
    return (signal * scale).rename("Qdot_int")
