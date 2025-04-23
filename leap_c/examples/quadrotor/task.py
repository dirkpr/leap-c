from collections import OrderedDict
from copy import copy
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

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


@register_task("quadrotor_ref_easy")
class QuadrotorRefEasy(Task):
    def __init__(self):
        params = PARAMS
        mpc = QuadrotorMpc(N_horizon=10, params_learnable=["xref1", "xref2", "xref3"], params=params)
        mpc_layer = MpcSolutionModule(mpc)

        nx, nu, Nhor = mpc.ocp.dims.nx, mpc.ocp.dims.nu, mpc.ocp.dims.N
        self.nx, self.nu, self.Nhor = nx, nu, Nhor
        self.param_low = copy(mpc.ocp_sensitivity.p_global_values)-1#-3
        self.param_high = copy(mpc.ocp_sensitivity.p_global_values)+1#+3

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
        obs = obs[:, :-self.nu]

        if param_nn is None:
            raise ValueError("Parameter tensor is required for MPC task.")

        mpc_param = MpcParameter(
            p_global=param_nn,  # type:     
        )

        return MpcInput(x0=obs, parameters=mpc_param)

    def create_env(self, train: bool) -> gym.Env:
        return QuadrotorStop(difficulty="hard")


@register_task("quadrotor_ref_medium")
class QuadrotorRefMedium(QuadrotorRefEasy):
    def create_env(self, train: bool) -> gym.Env:
         return QuadrotorStop(difficulty="medium")


@register_task("quadrotor_ref_hard")
class QuadrotorRefHard(QuadrotorRefEasy):
    def create_env(self, train: bool) -> gym.Env:
         return QuadrotorStop(difficulty="hard")


#
# @register_task("quadrotor_diag_costs")
# class QuadrotorStopTask(Task):
#
#     def __init__(self):
#         mpc = QuadrotorMpc(N_horizon=8, params_learnable=["xref", "xref_e", "uref", "q_diag", "q_diag_e", "r_diag"])
#         mpc_layer = MpcSolutionModule(mpc)
#
#         nx, nu, Nhor = mpc.ocp.dims.nx, mpc.ocp.dims.nu, mpc.ocp.dims.N
#         self.nx, self.nu, self.Nhor = nx, nu, Nhor
#         self.param_low = copy(mpc.ocp_sensitivity.p_global_values)
#         self.param_high = copy(mpc.ocp_sensitivity.p_global_values)
#
#         self.param_low[0:3] = -5.0
#         #self.param_low[nx:nx + 3] = -2.0
#         self.param_low[7:nx] = -10.
#         #self.param_low[nx + 7:nx + nx] = -10.
#         #self.param_low[2 * nx:2 * nx + nu] = 0.
#         #self.param_low[2 * nx + nu:] = self.param_low[2 * nx + nu:] * 1
#
#         self.param_high[0:3] = 5.0
#         #self.param_high[nx:nx + 3] = 2.0
#         self.param_high[7:nx] = 10.
#         #self.param_high[nx + 7:nx + nx] = 10.
#         #self.param_high[2 * nx:2 * nx + nu] = 5000.
#         #self.param_high[2 * nx + nu:] = self.param_low[2 * nx + nu:] * 1
#
#         super().__init__(mpc_layer)
#
#     @property
#     def param_space(self) -> spaces.Box:
#         low = self.param_low
#         high = self.param_high
#         return spaces.Box(low=low, high=high, dtype=np.float32)
#
#     def prepare_mpc_input(
#             self,
#             obs: Any,
#             param_nn: Optional[torch.Tensor] = None,
#             action: Optional[torch.Tensor] = None,
#     ) -> MpcInput:
#         if param_nn is None:
#             raise ValueError("Parameter tensor is required for MPC task.")
#
#         # prepare parameters
#         p_xref_batch = param_nn.detach().cpu().numpy()[:, :self.nx, ...]
#         p_xrefe_batch = param_nn.detach().cpu().numpy()[:, self.nx:2 * self.nx, ...]
#         p_uref_batch = param_nn.detach().cpu().numpy()[:, 2 * self.nx:2 * self.nx + self.nu, ...]
#         p_q_batch = param_nn.detach().cpu().numpy()[:, 2 * self.nx + self.nu: 3 * self.nx + self.nu, ...]
#         p_qe_batch = param_nn.detach().cpu().numpy()[:, 3 * self.nx + self.nu: 4 * self.nx + self.nu, ...]
#         p_r_batch = param_nn.detach().cpu().numpy()[:, 4 * self.nx + self.nu:, ...]
#
#         b, n = p_qe_batch.shape
#         p_W_e_batch = p_qe_batch[:, :, None] * np.eye(n)
#
#         p_qr_batch = np.concatenate((p_q_batch, p_r_batch), axis=1)
#         p_W_batch = p_qr_batch[:, :, None] * np.eye(self.nx + self.nu) # make diagonal matrix out of vector by expanding dimension
#         p_W_batch = np.expand_dims(p_W_batch, axis=1)
#         p_W_batch = np.repeat(p_W_batch, self.Nhor, axis=1)
#
#         p_yref_batch = np.concatenate((p_xref_batch, p_uref_batch), axis=1)
#         p_yref_batch = np.expand_dims(p_yref_batch, axis=1)
#         p_yref_batch = np.repeat(p_yref_batch, self.Nhor, axis=1)
#
#         p_yrefe_batch = p_xrefe_batch
#
#         mpc_param = MpcParameter(
#             p_global=param_nn,
#             p_W_e=p_W_e_batch,
#             p_W=p_W_batch,
#             p_yref_e=p_yrefe_batch,
#             p_yref=p_yref_batch,
#         )
#
#         return MpcInput(x0=obs, parameters=mpc_param)
#
#     def create_env(self, train: bool) -> gym.Env:
#         return QuadrotorStop()
#
#
#
# @register_task("quadrotor_ref")
# class QuadrotorStopTask(Task):
#
#     def __init__(self):
#         mpc = QuadrotorMpc(N_horizon=9, params_learnable=["xref"])
#         self.uref = np.expand_dims(mpc.params["uref"],0)
#         mpc_layer = MpcSolutionModule(mpc)
#
#         nx, nu, Nhor = mpc.ocp.dims.nx, mpc.ocp.dims.nu, 9
#         self.nx, self.nu, self.Nhor = nx, nu, Nhor
#         self.param_low = copy(mpc.ocp_sensitivity.p_global_values)
#         self.param_high = copy(mpc.ocp_sensitivity.p_global_values)
#
#         self.param_low[0:3] = -10.
#         #self.param_low[3:6] = -1
#         self.param_low[7:10] = -10.
#         self.param_low[10:] = -10.
#
#         self.param_high[0:3] = 10.
#         #self.param_high[3:6] = 1
#         self.param_high[7:10] = 10.
#         self.param_high[10:] = 10.
#
#         super().__init__(mpc_layer)
#
#     @property
#     def param_space(self) -> spaces.Box:
#         low = self.param_low
#         high = self.param_high
#         return spaces.Box(low=low, high=high, dtype=np.float32)
#
#     def prepare_mpc_input(
#             self,
#             obs: Any,
#             param_nn: Optional[torch.Tensor] = None,
#             action: Optional[torch.Tensor] = None,
#     ) -> MpcInput:
#         if param_nn is None:
#             raise ValueError("Parameter tensor is required for MPC task.")
#
#         # prepare parameters
#         p_xref_batch = param_nn.detach().cpu().numpy()[:, :self.nx, ...]
#         nb, nx = p_xref_batch.shape
#
#         uref = np.repeat(self.uref, nb, axis=0)
#
#         p_yref_batch = np.concatenate((p_xref_batch, uref), axis=1)
#         p_yref_batch = np.expand_dims(p_yref_batch, axis=1)
#         p_yref_batch = np.repeat(p_yref_batch, self.Nhor, axis=1)
#
#         mpc_param = MpcParameter(
#             p_global=param_nn,
#             p_yref=p_yref_batch,
#         )
#
#         return MpcInput(x0=obs, parameters=mpc_param)
#
#     def create_env(self, train: bool) -> gym.Env:
#         return QuadrotorStop()
#
#
# @register_task("quadrotor_terminal_weights")
# class QuadrotorStopTask(Task):
#
#     def __init__(self):
#         mpc = QuadrotorMpc(N_horizon=4, params_learnable=["q_diag_e"])
#         mpc_layer = MpcSolutionModule(mpc)
#
#         self.q_diag_e = mpc.given_default_param_dict["q_diag_e"]
#         self.param_low = mpc.ocp_sensitivity.p_global_values * 1e-5
#         self.param_high = mpc.ocp_sensitivity.p_global_values * 5
#
#         super().__init__(mpc_layer)
#
#     @property
#     def param_space(self) -> spaces.Box:
#         low = self.param_low
#         high = self.param_high
#         return spaces.Box(low=low, high=high, dtype=np.float32)
#
#     def prepare_mpc_input(
#             self,
#             obs: Any,
#             param_nn: Optional[torch.Tensor] = None,
#             action: Optional[torch.Tensor] = None,
#     ) -> MpcInput:
#         if param_nn is None:
#             raise ValueError("Parameter tensor is required for MPC task.")
#
#         # prepare y_ref_e
#         param_q_e_batch = param_nn.detach().cpu().numpy()
#         b, n = param_q_e_batch.shape
#         param_W_e_batch = param_q_e_batch[:, :, None] * np.eye(n)
#
#         mpc_param = MpcParameter(
#             p_global=param_nn,
#             p_W_e=param_W_e_batch,
#         )
#
#         return MpcInput(x0=obs, parameters=mpc_param)
#
#     def create_env(self, train: bool) -> gym.Env:
#         return QuadrotorStop()
#
#
# @register_task("quadrotor_refe")
# class QuadrotorStopTask(Task):
#
#     def __init__(self):
#         mpc = QuadrotorMpc(N_horizon=4, params_learnable=["xref_e"])
#         mpc_layer = MpcSolutionModule(mpc)
#
#         self.param_low = copy(mpc.ocp_sensitivity.p_global_values)
#         self.param_low[0:3] = -2.0
#         self.param_high = copy(mpc.ocp_sensitivity.p_global_values)
#         self.param_high[0:3] = 2.0
#
#         super().__init__(mpc_layer)
#
#     @property
#     def param_space(self) -> spaces.Box:
#         low = self.param_low
#         high = self.param_high
#         return spaces.Box(low=low, high=high, dtype=np.float32)
#
#     def prepare_mpc_input(
#             self,
#             obs: Any,
#             param_nn: Optional[torch.Tensor] = None,
#             action: Optional[torch.Tensor] = None,
#     ) -> MpcInput:
#         if param_nn is None:
#             raise ValueError("Parameter tensor is required for MPC task.")
#
#         # prepare y_ref_e
#         param_q_e_batch = param_nn.detach().cpu().numpy()
#         # b, n = param_q_e_batch.shape
#         # param_W_e_batch = param_q_e_batch[:, :, None] * np.eye(n)
#
#         mpc_param = MpcParameter(
#             p_global=param_nn,
#             p_yref_e=param_q_e_batch,
#         )
#
#         return MpcInput(x0=obs, parameters=mpc_param)
#
#     def create_env(self, train: bool) -> gym.Env:
#         return QuadrotorStop()

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
