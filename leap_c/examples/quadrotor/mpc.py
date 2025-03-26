import json
from collections import OrderedDict
from copy import copy
from typing import Dict, List

import casadi as ca
import numpy as np
import scipy
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosOcpIterate, AcadosOcpFlattenedIterate
from casadi.tools import struct_symSX

from leap_c.examples.quadrotor.casadi_models import get_rhs_quadrotor
from leap_c.examples.quadrotor.utils import read_from_yaml
from leap_c.examples.util import translate_learnable_param_to_p_global, find_param_in_p_or_p_global
from leap_c.mpc import Mpc, MpcInput
from leap_c.utils import set_standard_sensitivity_options
from os.path import dirname, abspath


class QuadrotorMpc(Mpc):

    def __init__(
            self,
            params: dict[str, np.ndarray] | None = None,
            discount_factor: float = 0.99,
            n_batch: int = 64,
            N_horizon: int = 3,
            params_learnable: list[str] | None = None,
    ):
        """
        Args:
            params: A dict with the parameters of the ocp, together with their default values.
                For a description of the parameters, see the docstring of the class.
            learnable_params: A list of the parameters that should be learnable
                (necessary for calculating their gradients).
            N_horizon: The number of steps in the MPC horizon.
                The MPC will have N+1 nodes (the nodes 0...N-1 and the terminal node N).
            T_horizon: The length (meaning time) of the MPC horizon.
                One step in the horizon will equal T_horizon/N_horizon simulation time.
            discount_factor: The discount factor for the cost.
            n_batch: The batch size the MPC should be able to process
                (currently this is static).
            least_squares_cost: If True, the cost will be the LLS cost, if False it will
                be the general quadratic cost(see above).
            exact_hess_dyn: If False, the contributions of the dynamics will be left out of the Hessian.
        """
        not_implemented_params = ["q_diag", "r_diag", "xref", "uref", "m"]
        if params_learnable is not None:
            for param in not_implemented_params:
                if param in params_learnable:
                    raise ValueError(f"{param} cannot be learnable in this example.")

        params = (
            {
                #"m": 0.6,
                "q_diag": np.array([1e4, 1e4, 1e4,
                                   1e0, 1e4, 1e4, 1e0,
                                   1e1, 1e1, 1e3,
                                   1e1, 1e1, 1e1]),
                "r_diag": np.array([0.06, 0.06, 0.06, 0.06]),
                "q_diag_e": 10 * np.array([1e4, 1e4, 1e4,
                                            1e0, 1e4, 1e4, 1e0,
                                            1e1, 1e1, 1e3,
                                            1e1, 1e1, 1e1]),
                "xref": np.array([0, 0, 0,
                                  1, 0, 0, 0,
                                  0, 0, 0,
                                  0, 0, 0]),
                "uref": np.array([970.437] * 4),
                "xref_e": np.array([0., 0., 0.,
                                    1., 0., 0., 0.,
                                    0., 0., 0.,
                                    0., 0., 0.]),
            }
            if params is None else params
        )
        params_learnable = params_learnable if params_learnable is not None else []
        print("learnable_params: ", params_learnable)


        ocp = export_parametric_ocp(
            name="quadrotor_lls",
            N_horizon=N_horizon,
            sensitivity_ocp=False,
            nominal_param=params,
            params_learnable=params_learnable,
        )

        ocp_sens = export_parametric_ocp(
            name="quadrotor_lls_exact",
            N_horizon=N_horizon,
            sensitivity_ocp=True,
            nominal_param=params,
            params_learnable=params_learnable,
        )

        self.given_default_param_dict = params

        # with open(dirname(abspath(__file__)) + "/init_iterateN5.json", "r") as file:
        #     init_iterate = json.load(file)  # Parse JSON into a Python dictionary
        #     init_iterate = parse_ocp_iterate(init_iterate, N=N_horizon)
        #
        # def initialize_default(mpc_input: MpcInput):
        #     init_iterate.x_traj = [mpc_input.x0] * (ocp.solver_options.N_horizon + 1)
        #     return init_iterate

        # init_state_fn = initialize_default
        # Convert dictionary to a namedtuple

        super().__init__(
            ocp=ocp,
            ocp_sensitivity=ocp_sens,
            discount_factor=discount_factor,
            n_batch=n_batch,
            init_state_fn=None,
        )


def export_parametric_ocp(
        nominal_param: dict[str, np.ndarray],
        name: str = "quadrotor",
        N_horizon: int = 5,
        sensitivity_ocp: bool = False,
        params_learnable: list[str] | None = None,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ######## Dimensions ########
    dt = 0.04  # 0.005

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = N_horizon * dt

    ######## Process parameters ########

    ocp = translate_learnable_param_to_p_global(nominal_param, params_learnable, ocp)

    ######## Model ########
    # Quadrotor parameters
    model_params = read_from_yaml(dirname(abspath(__file__)) + "/model_params.yaml")

    # For now, no mass parameter
    x, u, p, rhs, rhs_func = get_rhs_quadrotor(model_params, model_fidelity="low", sym_params=False)
    ocp.model.disc_dyn_expr = disc_dyn_expr(rhs, x, u, p, dt)

    ocp.model.name = name
    ocp.model.x = x
    ocp.model.u = u

    xdot = ca.SX.sym('xdot', x.shape)
    ocp.model.xdot = xdot
    ocp.model.f_impl_expr = xdot - rhs

    ocp.dims.nx = x.size()[0]
    ocp.dims.nu = u.size()[0]
    nx, nu, ny, ny_e = ocp.dims.nx, ocp.dims.nu, ocp.dims.nx + ocp.dims.nu, ocp.dims.nx

    ######## Cost ########
    # stage cost
    Q = np.diag([1e4, 1e4, 1e4,
                 1e0, 1e4, 1e4, 1e0,
                 1e1, 1e1, 1e3,
                 1e1, 1e1, 1e1])

    R = np.diag([0.06, 0.06, 0.06, 0.06])

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx: nx + nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref[3] = 1
    ocp.cost.yref[nx:nx + nu] = 970.437

    if sensitivity_ocp:
        # Sensitivity solver
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_e = _cost_expr_ext_cost_e(ocp=ocp)

        # Solver options
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.qp_solver_ric_alg = 1
        ocp.solver_options.qp_solver_cond_N = ocp.solver_options.N_horizon
        ocp.solver_options.exact_hess_dyn = True
        ocp.solver_options.exact_hess_cost = True
        ocp.solver_options.exact_hess_constr = True
        ocp.solver_options.with_solution_sens_wrt_params = True
        ocp.solver_options.with_value_sens_wrt_params = True
        ocp.model.name += "_sensitivity"  # type:ignore

    else:
        # Forward OCP with SQP and GN Hessian
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

        ocp.cost.W_e = np.diag(nominal_param["q_diag_e"])
        Vx_e = np.zeros((ny_e, nx))
        Vx_e[:nx, :nx] = np.eye(nx)
        ocp.cost.Vx_e = Vx_e

        ocp.cost.yref_e = np.zeros((ny_e,))
        ocp.cost.yref_e[3] = 1

    # constraints
    ocp.constraints.idxbx = np.array([2])
    ocp.constraints.lbx = np.array([-model_params["bound_z"] * 10])
    ocp.constraints.ubx = np.array([model_params["bound_z"]])

    ocp.constraints.idxbx_e = np.array([2])
    ocp.constraints.lbx_e = np.array([-model_params["bound_z"] * 10])
    ocp.constraints.ubx_e = np.array([model_params["bound_z"]])

    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.zu = ocp.cost.zl = np.array([0])
    ocp.cost.Zu = ocp.cost.Zl = np.array([1e10])

    ######## Constraints ########
    ocp.constraints.x0 = np.array([0] * 13)
    ocp.constraints.lbu = np.array([0] * 4)
    ocp.constraints.ubu = np.array([model_params["motor_omega_max"]] * 4)
    ocp.constraints.idxbu = np.array(range(4))

    ######## Solver configuration ########
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 30

    ocp.solver_options.sim_method_num_stages = 2
    ocp.solver_options.sim_method_num_steps = 1



    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp

def disc_dyn_expr(rhs, x, u, p, dt: float) -> ca.SX:
    if p is not None:
        ode = ca.Function("ode", [x, u, p], [rhs])
        k1 = ode(x, u, *p)
        k2 = ode(x + dt / 2 * k1, u, *p)  # type:ignore
        k3 = ode(x + dt / 2 * k2, u, *p)  # type:ignore
        k4 = ode(x + dt * k3, u, *p)  # type:ignore
    else:
        ode = ca.Function("ode", [x, u], [rhs])
        k1 = ode(x, u)
        k2 = ode(x + dt / 2 * k1, u)  # type:ignore
        k3 = ode(x + dt / 2 * k2, u)  # type:ignore
        k4 = ode(x + dt * k3, u)  # type:ignore


    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def _create_diag_matrix(
    _q_sqrt: np.ndarray | ca.SX,
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [_q_sqrt]):
        return ca.diag(_q_sqrt)
    else:
        return np.diag(_q_sqrt)

def _cost_expr_ext_cost(ocp: AcadosOcp) -> ca.SX:
    x = ocp.model.x
    u = ocp.model.u

    Q_sqrt = _create_diag_matrix(
        find_param_in_p_or_p_global(["q_diag"], ocp.model)["q_diag"]
    )
    R_sqrt = _create_diag_matrix(
        find_param_in_p_or_p_global(["r_diag"], ocp.model)["r_diag"]
    )

    xref = find_param_in_p_or_p_global(["xref"], ocp.model)["xref"]
    uref = find_param_in_p_or_p_global(["uref"], ocp.model)["uref"]

    td = ocp.solver_options.tf / ocp.solver_options.N_horizon

    return td * 0.5 * (
        ca.mtimes([ca.transpose(x - xref), Q_sqrt.T, Q_sqrt, x - xref])
        + ca.mtimes([ca.transpose(u - uref), R_sqrt.T, R_sqrt, u - uref])
    )


def _cost_expr_ext_cost_e(ocp: AcadosOcp) -> ca.SX:
    x = ocp.model.x

    Q_e = _create_diag_matrix(
        find_param_in_p_or_p_global(["q_diag_e"], ocp.model)["q_diag_e"]
    )

    xref_e = find_param_in_p_or_p_global(["xref_e"], ocp.model)["xref_e"]

    return 0.5 * ca.mtimes([ca.transpose(x - xref_e), Q_e, x - xref_e])