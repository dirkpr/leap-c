"""Differentiable MPC planner for the minimal BOPTEST heat-pump example."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from leap_c.examples.boptest.acados_ocp import FORECAST_PARAM_NAMES, export_parametric_ocp
from leap_c.ocp.acados.diff_mpc import collate_acados_diff_mpc_ctx
from leap_c.ocp.acados.torch import AcadosDiffMpcCtx, AcadosDiffMpcTorch
from leap_c.planner import ParameterizedPlanner
from leap_c.utils.parameters import broadcast_default_param


@dataclass(kw_only=True)
class BoptestPlannerConfig:
    """Configuration for the :class:`BoptestPlanner`.

    Attributes:
        N_horizon: Number of shooting intervals in the MPC horizon. Must match
            the environment's ``N_forecast``.
        step_period: Sampling time in seconds (matches the BOPTEST step period).
        discount_factor: Optional discount factor along the MPC horizon.
        n_batch_init: Initially supported batch size of the batch OCP solver.
        num_threads_batch_solver: Number of parallel threads for the batch solver.
        dtype: Type the planner output tensors are cast to.
    """

    N_horizon: int = 12
    step_period: float = 900.0
    discount_factor: float | None = None
    n_batch_init: int | None = None
    num_threads_batch_solver: int | None = None
    dtype: torch.dtype | None = None


class BoptestPlanner(ParameterizedPlanner[AcadosDiffMpcCtx]):
    """Acados-based planner for the ``bestest_hydronic_heat_pump`` BOPTEST case.

    The state is the zone temperature and the single control is the heat-pump
    modulation ``oveHeaPumY_u``. Forecast/exogenous quantities (ambient
    temperature, solar irradiance, price and the comfort bounds) are read from
    the structured observation and injected as non-differentiable stagewise
    parameters; the R1C1 gains and cost weights are differentiable parameters.
    """

    cfg: BoptestPlannerConfig
    collate_fn_map = {AcadosDiffMpcCtx: collate_acados_diff_mpc_ctx}

    def __init__(
        self,
        cfg: BoptestPlannerConfig | None = None,
        export_directory: Path | None = None,
    ) -> None:
        self.cfg = BoptestPlannerConfig() if cfg is None else cfg
        super().__init__()

        ocp, param_manager, param_space, default_param = export_parametric_ocp(
            name="boptest",
            N_horizon=self.cfg.N_horizon,
            step_period=self.cfg.step_period,
        )

        self.diff_mpc = AcadosDiffMpcTorch(
            ocp,
            param_manager,
            discount_factor=self.cfg.discount_factor,
            export_directory=export_directory,
            n_batch_init=self.cfg.n_batch_init,
            num_threads_batch_solver=self.cfg.num_threads_batch_solver,
            dtype=self.cfg.dtype,
        )
        self.param_manager = param_manager
        self._param_space = param_space
        self._default_param = default_param

    def forward(
        self,
        obs: dict,
        action: torch.Tensor | None = None,
        params: dict[str, torch.Tensor | np.ndarray] | None = None,
        ctx: AcadosDiffMpcCtx | None = None,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = obs["state"]
        if not isinstance(x0, torch.Tensor):
            x0 = torch.as_tensor(x0)
        batch_size = x0.shape[0]
        n_stages = self.cfg.N_horizon + 1

        stagewise: dict[str, np.ndarray] = {}
        for name in FORECAST_PARAM_NAMES:
            fc = obs["forecast"][name]
            if isinstance(fc, torch.Tensor):
                fc = fc.detach().cpu().numpy()
            fc = np.asarray(fc, dtype=np.float64)
            if fc.shape[1] < n_stages:
                pad = np.repeat(fc[:, -1:], n_stages - fc.shape[1], axis=1)
                fc = np.concatenate([fc, pad], axis=1)
            stagewise[name] = fc[:, :n_stages].reshape(batch_size, n_stages, 1)

        merged: dict[str, torch.Tensor | np.ndarray] = dict(params) if params else {}
        merged.update(stagewise)
        return self.diff_mpc(x0=x0, u0=action, params=merged, ctx=ctx)

    def default_param(self, obs: Any = None) -> dict[str, np.ndarray]:
        ref = obs["state"] if isinstance(obs, dict) else obs
        return broadcast_default_param(self._default_param, ref)
