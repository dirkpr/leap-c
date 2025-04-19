from collections import OrderedDict
import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosModel
from casadi.tools import struct_symSX

from leap_c.examples.quadrotor.casadi_models import get_rhs_quadrotor
from leap_c.examples.quadrotor.utils import read_from_yaml, quat_error
from leap_c.examples.util import translate_learnable_param_to_p_global, find_param_in_p_or_p_global, \
    assign_lower_triangular
from leap_c.mpc import Mpc
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
        # Set the default parameters
        params = OrderedDict([
            ("xref1", np.array([0.])),  # x
            ("xref2", np.array([0.])),  # y
            ("xref3", np.array([0.])),  # z

            ("xref4", np.array([0.])),  # angles not quaternions!
            ("xref5", np.array([0.])),
            ("xref6", np.array([0.])),

            ("xref7", np.array([0.])),
            ("xref8", np.array([0.])),
            ("xref9", np.array([0.])),

            ("xref10", np.array([0.])),
            ("xref11", np.array([0.])),
            ("xref12", np.array([0.])),

            ("uref", np.array([970.437] * 4)),

            ("L11", np.array([np.sqrt(25.0)])),
            ("L22", np.array([np.sqrt(25.0)])),
            ("L33", np.array([np.sqrt(25.0)])),

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
        ]) if params is None else params

        self.params = params
        params_learnable = params_learnable if params_learnable is not None else []
        print("learnable_params: ", params_learnable)

        ocp = export_parametric_ocp(
            name="quadrotor_lls",
            N_horizon=N_horizon,
            nominal_param=params,
            params_learnable=params_learnable,
        )

        self.given_default_param_dict = params

        super().__init__(
            ocp=ocp,
            discount_factor=discount_factor,
            n_batch=n_batch,
        )


def export_parametric_ocp(
        nominal_param: dict[str, np.ndarray],
        name: str = "quadrotor",
        N_horizon: int = 5,
        params_learnable: list[str] | None = None,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ######## Dimensions ########
    dt = 0.04

    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = N_horizon * dt

    ######## Process parameters ########

    ocp = translate_learnable_param_to_p_global(nominal_param, params_learnable, ocp)

    ######## Model ########
    # Quadrotor parameters
    model_params = read_from_yaml(dirname(abspath(__file__)) + "/model_params.yaml")

    # For now, no mass parameter
    x, u, rhs, rhs_func = get_rhs_quadrotor(model_params, model_fidelity="low")
    ocp.model.disc_dyn_expr = disc_dyn_expr(rhs, x, u, None, dt)

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
    ocp.model.cost_y_expr = ca.vertcat(ocp.model.x[:3],
                                       quat_error(ocp.model.x[3:7], np.array([1, 0, 0, 0])),
                                       ocp.model.x[7:],
                                       ocp.model.u)

    ocp.cost.W_e = ocp.cost.W[:12, :12]
    ocp.cost.yref_e = ocp.cost.yref[:12]
    ocp.model.cost_y_expr_e = ca.vertcat(ocp.model.x[:3],
                                         quat_error(ocp.model.x[3:7], np.array([1, 0, 0, 0])),
                                         ocp.model.x[7:])

    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

    # constraints
    ocp.constraints.idxbx = np.array([2])
    ocp.constraints.lbx = np.array([model_params["lower_bound_z"]])
    ocp.constraints.ubx = np.array([model_params["upper_bound_z"]])

    ocp.constraints.idxbx_e = np.array([2, 7, 8, 9])
    ocp.constraints.lbx_e = np.array([model_params["lower_bound_z"], 0., 0., 0.])
    ocp.constraints.ubx_e = np.array([model_params["upper_bound_z"], 0., 0., 0.])

    ocp.constraints.idxsbx = np.array([0])
    ocp.cost.zu = ocp.cost.zl = np.array([0])
    ocp.cost.Zu = ocp.cost.Zl = np.array([1e3])

    ocp.constraints.idxsbx_e = np.array([0, 1, 2, 3])
    ocp.cost.zu_e = ocp.cost.zl_e = np.array([0, 0, 0, 0])
    ocp.cost.Zu_e = ocp.cost.Zl_e = np.array([1e4, 1e4, 1e4, 1e4])

    ######## Constraints ########
    ocp.constraints.x0 = np.array([0] * 13)
    ocp.constraints.lbu = np.array([0] * 4)
    ocp.constraints.ubu = np.array([model_params["motor_omega_max"]] * 4)
    ocp.constraints.idxbu = np.array(range(4))

    ######## Solver configuration ########
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 100  # dont set to 1!! does not update controls
    ocp.solver_options.num_threads_in_batch_solve = 1

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


def yref_casadi(model: AcadosModel) -> ca.SX:
    return ca.vertcat(
        *find_param_in_p_or_p_global([f"xref{i}" for i in range(1, 13)] + ["uref"], model).values())  # type:ignore


def cost_matrix_casadi(model: AcadosModel) -> ca.SX:
    L = ca.diag(ca.vertcat(*find_param_in_p_or_p_global([f"L{i}{i}" for i in range(1, 17)], model).values()))
    L_offdiag = find_param_in_p_or_p_global(["Lloweroffdiag"], model)["Lloweroffdiag"]

    assign_lower_triangular(L, L_offdiag)

    return L @ L.T
