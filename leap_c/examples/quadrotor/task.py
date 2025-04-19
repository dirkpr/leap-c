from copy import copy
from typing import Any, Optional

import gymnasium as gym
import torch
import numpy as np
from gymnasium import spaces
from collections import OrderedDict
from leap_c.examples.quadrotor.env import QuadrotorStop
from leap_c.examples.quadrotor.mpc import QuadrotorMpc
from leap_c.nn.modules import MpcSolutionModule
from leap_c.registry import register_task
from leap_c.task import Task
from ...mpc import MpcInput, MpcParameter

# Set the default parameters
PARAMS = OrderedDict([
    ("xref1", np.array([0.])),  # x
    ("xref2", np.array([0.])),  # y
    ("xref3", np.array([0.])),  # z

    ("xref4", np.array([0.])),  # roll angles not quaternions!
    ("xref5", np.array([0.])),  # pitch
    ("xref6", np.array([0.])),  # yaw

    ("xref7", np.array([0.])),
    ("xref8", np.array([0.])),
    ("xref9", np.array([0.])),

    ("xref10", np.array([0.])),
    ("xref11", np.array([0.])),
    ("xref12", np.array([0.])),

    ("uref", np.array([970.437] * 4)),

    ("L11", np.array([np.sqrt(45.0)])),
    ("L22", np.array([np.sqrt(45.0)])),
    ("L33", np.array([np.sqrt(45.0)])),

    ("L44", np.array([np.sqrt(20.)])),
    ("L55", np.array([np.sqrt(20.)])),
    ("L66", np.array([np.sqrt(0.01)])),

    ("L77", np.array([np.sqrt(0.01)])),
    ("L88", np.array([np.sqrt(0.01)])),
    ("L99", np.array([np.sqrt(0.01)])),

    ("L1010", np.array([np.sqrt(0.1)])),
    ("L1111", np.array([np.sqrt(0.1)])),
    ("L1212", np.array([np.sqrt(1)])),

    ("L1313", np.array([np.sqrt(3e-5)])),
    ("L1414", np.array([np.sqrt(3e-5)])),
    ("L1515", np.array([np.sqrt(3e-5)])),
    ("L1616", np.array([np.sqrt(3e-5)])),
    ("Lloweroffdiag", np.array([0.0] * (15 + 14 + 13 + 12 + 11 + 10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1))),
])


@register_task("quadrotor_ref")
class QuadrotorStopTask(Task):
    def __init__(self):
        params = PARAMS
        mpc = QuadrotorMpc(N_horizon=10, params_learnable=["xref1", "xref2", "xref3"], params=params)
        mpc_layer = MpcSolutionModule(mpc)

        nx, nu, Nhor = mpc.ocp.dims.nx, mpc.ocp.dims.nu, mpc.ocp.dims.N
        self.nx, self.nu, self.Nhor = nx, nu, Nhor
        self.param_low = copy(mpc.ocp_sensitivity.p_global_values) - 3  # -3
        self.param_high = copy(mpc.ocp_sensitivity.p_global_values) + 3  # +3

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

        mpc_param = MpcParameter(
            p_global=param_nn,
        )

        return MpcInput(x0=obs, parameters=mpc_param)

    def create_env(self, train: bool) -> gym.Env:
        return QuadrotorStop(difficulty="hard")


@register_task("quadrotor_look_down")
class QuadrotorStopTask(Task):
    def __init__(self):
        params = PARAMS
        params["xref4"] = np.array([-np.pi / 2])
        params["L44"] = np.array([200])
        mpc = QuadrotorMpc(N_horizon=10, params_learnable=["xref1", "xref2", "xref3"], params=params)
        mpc_layer = MpcSolutionModule(mpc)

        nx, nu, Nhor = mpc.ocp.dims.nx, mpc.ocp.dims.nu, mpc.ocp.dims.N
        self.nx, self.nu, self.Nhor = nx, nu, Nhor
        self.param_low = copy(mpc.ocp_sensitivity.p_global_values) - 3  # -3
        self.param_high = copy(mpc.ocp_sensitivity.p_global_values) + 3  # +3

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

        mpc_param = MpcParameter(
            p_global=param_nn,
        )

        return MpcInput(x0=obs, parameters=mpc_param)

    def create_env(self, train: bool) -> gym.Env:
        return QuadrotorStop(difficulty="hard", look_down_reward=True)


@register_task("quadrotor_look_down_2")
class QuadrotorStopTask(Task):
    def __init__(self):
        params = PARAMS
        params["xref4"] = np.array([-np.pi / 2])
        params["L44"] = np.array([200])
        mpc = QuadrotorMpc(N_horizon=10,
                           params_learnable=["xref1", "xref2", "xref3", "xref4", "L11", "L22", "L33", "L44"],
                           params=params)
        mpc_layer = MpcSolutionModule(mpc)

        nx, nu, Nhor = mpc.ocp.dims.nx, mpc.ocp.dims.nu, mpc.ocp.dims.N
        self.nx, self.nu, self.Nhor = nx, nu, Nhor
        self.param_low = copy(mpc.ocp_sensitivity.p_global_values)
        self.param_high = copy(mpc.ocp_sensitivity.p_global_values)
        self.param_low[0:3] = -3.0
        self.param_low[3] = -np.pi / 2
        self.param_low[4:8] = 1e-5
        self.param_low[8] = -np.pi / 2
        self.param_low[0:2] = 3.0
        self.param_low[2] = 0.0
        self.param_low[3] = 0.0
        self.param_low[4:8] = 60
        self.param_low[8] = 200

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

        mpc_param = MpcParameter(
            p_global=param_nn,
        )

        return MpcInput(x0=obs, parameters=mpc_param)

    def create_env(self, train: bool) -> gym.Env:
        return QuadrotorStop(difficulty="hard", look_down_reward=True)
