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


@register_task("quadrotor_ref")
class QuadrotorStopTask(Task):
    def __init__(self):
        mpc = QuadrotorMpc(N_horizon=9, params_learnable=["xref1", "xref2", "xref3"])
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
        if param_nn is None:
            raise ValueError("Parameter tensor is required for MPC task.")

        mpc_param = MpcParameter(
            p_global=param_nn,
        )

        return MpcInput(x0=obs, parameters=mpc_param)

    def create_env(self, train: bool) -> gym.Env:
         return QuadrotorStop(scale_disturbances=0.000)

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
