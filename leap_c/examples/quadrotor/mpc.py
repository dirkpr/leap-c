import json
from collections import OrderedDict
from copy import copy
from typing import Dict, List

import casadi as ca
import numpy as np
import scipy
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosOcpIterate, AcadosOcpFlattenedIterate, AcadosModel
from casadi.tools import struct_symSX

from leap_c.examples.quadrotor.casadi_models import get_rhs_quadrotor
from leap_c.examples.quadrotor.utils import read_from_yaml
from leap_c.examples.util import translate_learnable_param_to_p_global, find_param_in_p_or_p_global, \
    assign_lower_triangular
from leap_c.mpc import Mpc, MpcInput
from leap_c.utils import set_standard_sensitivity_options
from os.path import dirname, abspath


class QuadrotorMpc(Mpc):
    def __init__(
            self,
            params: dict[str, np.ndarray] | None = None,
            discount_factor: float = 0.99,
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
            least_squares_cost: If True, the cost will be the LLS cost, if False it will
                be the general quadratic cost(see above).
            exact_hess_dyn: If False, the contributions of the dynamics will be left out of the Hessian.
        """
        not_implemented_params = ["m"]
        if params_learnable is not None:
            for param in not_implemented_params:
                if param in params_learnable:
                    raise ValueError(f"{param} cannot be learnable in this example.")

        params = OrderedDict([
            ("xref1", np.array([0.])),#x
            ("xref2", np.array([0.])),#y
            ("xref3", np.array([0.])),#z
            ("xref4", np.array([1.])),
            ("xref5", np.array([0.])),
            ("xref6", np.array([0.])),
            ("xref7", np.array([0.])),
            ("xref8", np.array([0.])),
            ("xref9", np.array([0.])),
            ("xref10", np.array([0.])),
            ("xref11", np.array([0.])),
            ("xref12", np.array([0.])),
            ("xref13", np.array([0.])),
            ("uref", np.array([970.437] * 4)),
            ("L11", np.array([np.sqrt(1.0)])),
            ("L22", np.array([np.sqrt(1.0)])),
            ("L33", np.array([np.sqrt(1.0)])),
            ("L44", np.array([np.sqrt(0.0001)])),
            ("L55", np.array([np.sqrt(1.0)])),
            ("L66", np.array([np.sqrt(1.0)])),
            ("L77", np.array([np.sqrt(0.0001)])),
            ("L88", np.array([np.sqrt(0.01)])),
            ("L99", np.array([np.sqrt(0.01)])),
            ("L1010", np.array([np.sqrt(0.01)])),
            ("L1111", np.array([np.sqrt(0.001)])),
            ("L1212", np.array([np.sqrt(0.001)])),
            ("L1313", np.array([np.sqrt(0.001)])),
            ("L1414", np.array([np.sqrt(3e-7)])),
            ("L1515", np.array([np.sqrt(3e-7)])),
            ("L1616", np.array([np.sqrt(3e-7)])),
            ("L1717", np.array([np.sqrt(3e-7)])),
            ("Lloweroffdiag", np.array([0.0] * (16 + 15 + 14 + 13 + 12 + 11 + 10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1))),
        ]) if params is None else params

        self.params = params
        params_learnable = params_learnable if params_learnable is not None else []
        print("learnable_params: ", params_learnable)

        ocp = export_parametric_ocp(
            name="quadrotor_lls",
            N_horizon=N_horizon,
            sensitivity_ocp=False,
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
            discount_factor=discount_factor,
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

    # Forward OCP with SQP and GN Hessian
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    ocp.cost.W = cost_matrix_casadi(ocp.model)
    ocp.cost.yref = yref_casadi(ocp.model)
    ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)

    ocp.cost.W_e = ocp.cost.W[:13, :13]
    ocp.cost.yref_e = ocp.cost.yref[:13]
    ocp.model.cost_y_expr_e = ocp.model.x

    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

    # constraints
    ocp.constraints.idxbx = np.array([2])
    ocp.constraints.lbx = np.array([model_params["lower_bound_z"]])
    ocp.constraints.ubx = np.array([model_params["upper_bound_z"]])

    ocp.constraints.idxbx_e = np.array([2,7,8,9])
    ocp.constraints.lbx_e = np.array([model_params["lower_bound_z"],0.,0.,0.])
    ocp.constraints.ubx_e = np.array([model_params["upper_bound_z"],0.,0.,0.])

    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.zu = ocp.cost.zl = np.array([0])
    ocp.cost.Zu = ocp.cost.Zl = np.array([1e3])

    ocp.constraints.idxsbx_e = np.array([0, 1, 2, 3])
    ocp.cost.zu_e = ocp.cost.zl_e = np.array([0,0,0,0])
    ocp.cost.Zu_e = ocp.cost.Zl_e = np.array([1e4,1e4,1e4,1e4])

    ######## Constraints ########
    ocp.constraints.x0 = np.array([0] * 13)
    ocp.constraints.lbu = np.array([0] * 4)
    ocp.constraints.ubu = np.array([model_params["motor_omega_max"]] * 4)
    ocp.constraints.idxbu = np.array(range(4))

    ######## Solver configuration ########
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 100# dont set to 1!! does not update controls
    ocp.solver_options.num_threads_in_batch_solve = 1

    ocp.solver_options.sim_method_num_stages = 2
    ocp.solver_options.sim_method_num_steps = 2
    ocp.solver_options.tol = 1e-3  # Is default
    ocp.solver_options.qp_tol = 1e-4

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


def yref_casadi(model: AcadosModel) -> ca.SX:
    return ca.vertcat(
        *find_param_in_p_or_p_global([f"xref{i}" for i in range(1, 14)] + ["uref"], model).values())  # type:ignore


def cost_matrix_casadi(model: AcadosModel) -> ca.SX:
    L = ca.diag(ca.vertcat(*find_param_in_p_or_p_global([f"L{i}{i}" for i in range(1, 18)], model).values()))
    L_offdiag = find_param_in_p_or_p_global(["Lloweroffdiag"], model)["Lloweroffdiag"]

    assign_lower_triangular(L, L_offdiag)

    return L @ L.T
