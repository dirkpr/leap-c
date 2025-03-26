from copy import copy
from typing import Any, Optional

import gymnasium as gym
import torch
import numpy as np
from gymnasium import spaces

from leap_c.examples.quadrotor.env import QuadrotorStop
from leap_c.examples.quadrotor.mpc import QuadrotorMpc
from leap_c.nn.modules import MpcSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task
from .utils import read_from_yaml
from functools import cached_property

from ...mpc import MpcInput, MpcParameter


@register_task("quadrotor_terminal_weights")
class QuadrotorStopTask(Task):

    def __init__(self):
        mpc = QuadrotorMpc(N_horizon=4, params_learnable=["q_diag_e"])
        mpc_layer = MpcSolutionModule(mpc)

        self.q_diag_e = mpc.given_default_param_dict["q_diag_e"]

        self.param_low = mpc.ocp_sensitivity.p_global_values * 1e-5
        self.param_high = mpc.ocp_sensitivity.p_global_values * 5

        super().__init__(mpc_layer)

    @property
    def param_space(self) -> spaces.Box:
        low = self.param_low
        high = self.param_high
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def prepare_mpc_input(
            self,
            obs: Any,
            param_nn: Optional[torch.Tensor] = None,
            action: Optional[torch.Tensor] = None,
    ) -> MpcInput:
        if param_nn is None:
            raise ValueError("Parameter tensor is required for MPC task.")

        # prepare y_ref_e
        param_q_e_batch = param_nn.detach().cpu().numpy()
        b, n = param_q_e_batch.shape
        param_W_e_batch = param_q_e_batch[:, :, None] * np.eye(n)

        mpc_param = MpcParameter(
            p_global=param_nn,
            p_W_e=param_W_e_batch,
        )

        return MpcInput(x0=obs, parameters=mpc_param)

    def create_env(self, train: bool) -> gym.Env:
        return QuadrotorStop()

@register_task("quadrotor_refe")
class QuadrotorStopTask(Task):

    def __init__(self):
        mpc = QuadrotorMpc(N_horizon=4, params_learnable=["xref_e"])
        mpc_layer = MpcSolutionModule(mpc)

        self.param_low = copy(mpc.ocp_sensitivity.p_global_values)
        self.param_low[0:3] = -2.0
        self.param_high = copy(mpc.ocp_sensitivity.p_global_values)
        self.param_high[0:3] = 2.0

        super().__init__(mpc_layer)

    @property
    def param_space(self) -> spaces.Box:
        low = self.param_low
        high = self.param_high
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def prepare_mpc_input(
            self,
            obs: Any,
            param_nn: Optional[torch.Tensor] = None,
            action: Optional[torch.Tensor] = None,
    ) -> MpcInput:
        if param_nn is None:
            raise ValueError("Parameter tensor is required for MPC task.")

        # prepare y_ref_e
        param_q_e_batch = param_nn.detach().cpu().numpy()
        #b, n = param_q_e_batch.shape
        #param_W_e_batch = param_q_e_batch[:, :, None] * np.eye(n)

        mpc_param = MpcParameter(
            p_global=param_nn,
            p_yref_e=param_q_e_batch,
        )

        return MpcInput(x0=obs, parameters=mpc_param)

    def create_env(self, train: bool) -> gym.Env:
        return QuadrotorStop()
