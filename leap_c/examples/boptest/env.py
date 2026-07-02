"""Minimal leap-c environment wrapper around boptest-gym.

Reuses :class:`BoptestGymEnv` from the vendored ``external/project1-boptest-gym``
submodule and only adapts it to the leap-c convention: a structured ``Dict``
observation ``{"state", "forecast": {...}}`` that the planner can consume
directly, and a plain ``Box`` action that maps straight onto the heat-pump
modulation input ``oveHeaPumY_u`` of the ``bestest_hydronic_heat_pump`` case.

boptest-gym is a REST client that talks to a running BOPTEST server, and its
``BoptestGymEnv.__init__`` already contacts that server. To keep this wrapper
importable and constructible without a server (needed for offline unit tests and
for registry discovery), the underlying env is created lazily on the first
:meth:`reset`; the gym spaces are built here from static metadata.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from leap_c.examples.boptest.acados_ocp import FORECAST_PARAM_NAMES

# Repo-root-relative path to the flat boptest-gym module (imported lazily).
_BOPTEST_GYM_ROOT = Path(__file__).resolve().parents[3] / "external" / "project1-boptest-gym"

# BOPTEST measurement used as the MPC state (zone operative temperature [K]).
_STATE_POINT = "reaTZon_y"

# BOPTEST forecast points mapped, in order, onto the leap-c forecast names.
# The order here MUST match ``FORECAST_PARAM_NAMES`` because the flat boptest-gym
# observation is decoded positionally.
_FORECAST_POINTS = {
    "TDryBul": "T_amb",
    "HGloHor": "solar",
    "PriceElectricPowerDynamic": "price",
    "LowerSetp[1]": "T_lower",
    "UpperSetp[1]": "T_upper",
}
assert list(_FORECAST_POINTS.values()) == FORECAST_PARAM_NAMES, (
    "Forecast decode order must match the OCP stagewise parameter order."
)

# Static (min, max) bounds for every requested signal. Used both for the gym
# spaces here and for the ``observations`` dict handed to boptest-gym.
_SIGNAL_BOUNDS = {
    "reaTZon_y": (280.0, 310.0),
    "TDryBul": (250.0, 320.0),
    "HGloHor": (0.0, 1200.0),
    "PriceElectricPowerDynamic": (0.0, 1.0),
    "LowerSetp[1]": (280.0, 310.0),
    "UpperSetp[1]": (280.0, 310.0),
}

_ACTION_POINT = "oveHeaPumY_u"


@dataclass(kw_only=True)
class BoptestEnvConfig:
    """Configuration for :class:`BoptestEnv`.

    Attributes:
        url: URL of the BOPTEST server. The default matches a local
            ``docker compose up web worker provision`` deployment of BOPTEST
            v0.8.0 (web service on port 80).
        testcase: BOPTEST test case identifier.
        step_period: Sampling / control step in seconds.
        N_forecast: Number of forecast steps ahead exposed in the observation.
            Must equal the planner horizon ``N_horizon``; each forecast array
            then has ``N_forecast + 1`` values (current step + horizon).
        scenario: BOPTEST scenario (pricing).
        start_time: Fixed episode start time in seconds from the beginning of
            the year (a winter heating day by default).
        warmup_period: Simulation warmup before the episode, in seconds.
        max_episode_length: Episode length in seconds.
    """

    url: str = "http://127.0.0.1"
    testcase: str = "bestest_hydronic_heat_pump"
    step_period: int = 900
    N_forecast: int = 12
    scenario: dict = field(default_factory=lambda: {"electricity_price": "dynamic"})
    start_time: int = 16 * 24 * 3600
    warmup_period: int = 3 * 24 * 3600
    max_episode_length: int = 24 * 3600


class BoptestEnv(gym.Env):
    """leap-c-aligned wrapper around ``BoptestGymEnv`` for the heat-pump case."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: BoptestEnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = BoptestEnvConfig() if cfg is None else cfg
        self.render_mode = render_mode
        self._core = None  # lazily created BoptestGymEnv (needs a server)

        n_fc = self.cfg.N_forecast + 1
        forecast_space = spaces.Dict(
            {
                name: spaces.Box(
                    low=_SIGNAL_BOUNDS[point][0],
                    high=_SIGNAL_BOUNDS[point][1],
                    shape=(n_fc,),
                    dtype=np.float32,
                )
                for point, name in _FORECAST_POINTS.items()
            }
        )
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=_SIGNAL_BOUNDS[_STATE_POINT][0],
                    high=_SIGNAL_BOUNDS[_STATE_POINT][1],
                    shape=(1,),
                    dtype=np.float32,
                ),
                "forecast": forecast_space,
            }
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def _ensure_core(self):
        """Create the underlying ``BoptestGymEnv`` (contacts the server)."""
        if self._core is not None:
            return
        if str(_BOPTEST_GYM_ROOT) not in sys.path:
            # boptest-gym is a flat repo; its root must be importable for both
            # ``boptestGymEnv`` and its ``examples`` package.
            sys.path.insert(0, str(_BOPTEST_GYM_ROOT))
        from boptestGymEnv import BoptestGymEnv  # noqa: PLC0415 (lazy, heavy import)

        observations = {_STATE_POINT: _SIGNAL_BOUNDS[_STATE_POINT]}
        for point in _FORECAST_POINTS:
            observations[point] = _SIGNAL_BOUNDS[point]

        self._core = BoptestGymEnv(
            url=self.cfg.url,
            testcase=self.cfg.testcase,
            actions=[_ACTION_POINT],
            observations=observations,
            predictive_period=self.cfg.N_forecast * self.cfg.step_period,
            scenario=self.cfg.scenario,
            step_period=self.cfg.step_period,
            start_time=self.cfg.start_time,
            warmup_period=self.cfg.warmup_period,
            max_episode_length=self.cfg.max_episode_length,
            random_start_time=False,
        )

    def _decode(self, flat: np.ndarray) -> dict:
        """Decode a flat boptest-gym observation into the leap-c ``Dict``.

        Layout (no time / no regressive terms):
        ``[state] + [T_amb x (N+1)] + [solar x (N+1)] + ...`` following the
        insertion order of ``observations`` given to boptest-gym.
        """
        flat = np.asarray(flat, dtype=np.float32)
        n_fc = self.cfg.N_forecast + 1
        forecast = {}
        offset = 1
        for name in _FORECAST_POINTS.values():
            forecast[name] = flat[offset : offset + n_fc].copy()
            offset += n_fc
        return {"state": flat[0:1].copy(), "forecast": forecast}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_core()
        obs, info = self._core.reset(seed=seed, options=options)
        return self._decode(obs), info

    def step(self, action):
        self._ensure_core()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = self._core.step(action)
        return self._decode(obs), float(reward), bool(terminated), bool(truncated), info

    def get_kpis(self) -> dict:
        """Return the BOPTEST core KPIs for the current test (needs the server)."""
        self._ensure_core()
        return self._core.get_kpis()

    def close(self):
        if self._core is not None:
            self._core.close()
