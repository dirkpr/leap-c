from collections import OrderedDict
import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosModel
from casadi.tools import struct_symSX

from leap_c.examples.quadrotor.casadi_models import get_rhs_quadrotor
from leap_c.examples.quadrotor.utils import read_from_yaml, quat_error
from leap_c.examples.util import (
    translate_learnable_param_to_p_global,
    find_param_in_p_or_p_global,
    assign_lower_triangular,
)
from leap_c.mpc import Mpc
from os.path import dirname, abspath

from leap_c.examples.quadrotor.utils import quaternion_multiply_casadi, quaternion_rotate_vector_casadi, read_from_yaml


class QuadrotorMpcc(Mpc):
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
        # Set the default parameters
        params = (
            OrderedDict(
                [
                    ("xref1", np.array([0.0])),  # x
                    ("xref2", np.array([0.0])),  # y
                    ("xref3", np.array([0.0])),  # z
                    ("xref4", np.array([-np.pi / 2])),  # roll angles not quaternions!
                    ("xref5", np.array([0.0])),  # pitch
                    ("xref6", np.array([0.0])),  # yaw
                    ("xref7", np.array([0.0])),
                    ("xref8", np.array([0.0])),
                    ("xref9", np.array([0.0])),
                    ("xref10", np.array([0.0])),
                    ("xref11", np.array([0.0])),
                    ("xref12", np.array([0.0])),
                    ("uref", np.array([970.437] * 4)),
                    ("L11", np.array([np.sqrt(45.0)])),
                    ("L22", np.array([np.sqrt(45.0)])),
                    ("L33", np.array([np.sqrt(45.0)])),
                    ("L44", np.array([np.sqrt(200.0)])),
                    ("L55", np.array([np.sqrt(20.0)])),
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
                    (
                        "Lloweroffdiag",
                        np.array(
                            [0.0]
                            * (
                                15
                                + 14
                                + 13
                                + 12
                                + 11
                                + 10
                                + 9
                                + 8
                                + 7
                                + 6
                                + 5
                                + 4
                                + 3
                                + 2
                                + 1
                            )
                        ),
                    ),
                ]
            )
            if params is None
            else params
        )

        self.params = params
        params_learnable = params_learnable if params_learnable is not None else []
        print("learnable_params: ", params_learnable)

        ocp = export_parametric_ocp(
            N_horizon=N_horizon,
            nominal_param=params,
            params_learnable=params_learnable,
        )

        self.given_default_param_dict = params

        super().__init__(
            ocp=ocp,
            discount_factor=discount_factor,
            init_state_fn=None,
        )

def get_cost_expr_ext_cost(ocp: AcadosOcp) -> ca.SX:


    # TODO: Extract position from x
    pos = ocp.model.x[0:3]
    # TODO: Compute ec
    # TODO: Add cost term on ec
    # TODO: Compute el
    # TODO: Add cost term on el
    # TODO: Extract omega from x
    # TODO: Add damping term on omega
    # TODO: Extract path acceleration from u
    # TODO: Add cost term on path acceleration
    # TODO: Extract path velocity from x
    # TODO: Add negative cost term on path velocity to incentivize path progression
    pass

def get_cost_expr_ext_cost_e(ocp: AcadosOcp) -> ca.SX:

    # TODO: Extract position from x
    # TODO: Compute ec
    # TODO: Add cost term on ec
    # TODO: Compute el
    # TODO: Add cost term on el
    # TODO: Extract omega from x
    # TODO: Add damping term on omega
    # TODO: Extract path acceleration from u
    # TODO: Add cost term on path acceleration
    # TODO: Extract path velocity from x
    # TODO: Add negative cost term on path velocity to incentivize path progression
    pass


def get_f_expl_expr(ocp: AcadosOcp) -> ca.SX:
    pass


def get_rhs(params: dict, model_fidelity: str = "low"):
    """
    Returns the right-hand side of the quadrotor dynamics.
    We model 4 rotors which are controlled by the motor speeds.
        params: Dict containing the parameters of the quadrotor model.
    """

    # states
    x_pos = ca.SX.sym("x_pos", 3)
    x_quat = ca.SX.sym("x_quat", 4)
    x_vel = ca.SX.sym("x_vel", 3)
    x_rot = ca.SX.sym("x_rot", 3)
    x_motor_speeds = ca.SX.sym("x_motor_speeds", 4)

    p_mass = params["mass"]

    x = ca.vertcat(x_pos, x_quat, x_vel, x_rot, x_motor_speeds)

    # controls
    u_motor_acceleration = ca.SX.sym("u_motor_acceleration", 4)

    # controls-2-thrust
    fz_props = x_motor_speeds ** 2 * params["motor_cl"]
    f_prop = ca.vertcat(0, 0, ca.sum1(fz_props))

    # controls-2-torques
    tau_prop = ca.vertcat(0, 0, 0)
    for i in range(4):
        tau = ca.vertcat(0, 0, params["motor_kappa"][i] * params["motor_cl"] * x_motor_speeds[i] ** 2)
        fz_prop = ca.vertcat(0, 0, x_motor_speeds[i] ** 2 * params["motor_cl"])
        tau_prop += tau + ca.cross(params["rotor_loc"][i], fz_prop)

    # residuals
    if model_fidelity == "low":
        f_res = ca.vertcat(0, 0, 0)
    elif model_fidelity == "high":
        mean_sqr_motor_speed = ca.sum1(ca.vertcat(*[x_motor_speeds[i] ** 2 for i in range(4)])) / 4
        v_xy = ca.sqrt(x_vel[0] ** 2 + x_vel[1] ** 2)
        v_x, v_y, v_z = x_vel[0], x_vel[1], x_vel[2]
        fres_x = p_mass * (params["cx_vx"] * v_x +
                           params["cx_vx_sqr"] * v_x * ca.fabs(v_x) +
                           params["cx_vx_mmean"] * v_x * mean_sqr_motor_speed)
        fres_y = p_mass * (params["cy_vy"] * v_y +
                           params["cy_vy_sqr"] * v_y * ca.fabs(v_y) +
                           params["cy_vy_mmean"] * v_y * mean_sqr_motor_speed)
        fres_z = p_mass * (params["cz_vz"] * v_z +
                           params["cz_mmean"] * mean_sqr_motor_speed +
                           params["cz_vhor"] * v_xy +
                           params["cz_vhor_sqr"] * v_xy ** 2 +
                           params["cz_vhor_mmean"] * v_xy * mean_sqr_motor_speed +
                           params["cz_vhor_vz_mmean"] * v_xy * mean_sqr_motor_speed * v_z +
                           params["cz_vz_cub"] * v_z ** 3)
        f_res = ca.vertcat(fres_x, fres_y, fres_z)
    else:
        raise NotImplementedError(f"Model fidelity {model_fidelity} not implemented")

    # derivatives
    dx_pos = x_vel
    dx_quat = quaternion_multiply_casadi(0.5 * x_quat, ca.vertcat(0, x_rot))

    acc_gravity = ca.vertcat(0, 0, -params["gravity"])
    acc_thrust = 1 / p_mass * quaternion_rotate_vector_casadi(x_quat, f_prop + f_res)
    dx_vel = acc_gravity + acc_thrust

    intertia = np.array(params["inertia"])
    drot = np.linalg.inv(intertia) @ (tau_prop - ca.cross(x_rot, intertia @ x_rot))

    rhs = ca.vertcat(dx_pos, dx_quat, dx_vel, drot, u_motor_acceleration)

    u = u_motor_acceleration

    rhs_func = ca.Function("f_ode", [x, u], [rhs])

    return x, u, rhs, rhs_func

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
    x, u, rhs, rhs_func = get_rhs(model_params, model_fidelity="low")

    # Additional state and input variables
    theta = ca.SX.sym("theta", 1)  # path variable
    theta_dot = ca.SX.sym("theta_dot", 1)  # path velocity
    theta_ddot = ca.SX.sym("theta_ddot", 1)  # path acceleration

    # Augment x with virtual states for path variable and path velocity
    x = ca.vertcat(x, theta, theta_dot)

    # Augment u with virtual path acceleration to u
    u = ca.vertcat(u, theta_ddot)

    # Add double integrator chain from path acceleration to path variable to rhs
    rhs = ca.vertcat(rhs, theta_dot, theta_ddot)

    # Redefine rhs_func to use the augmented x and u, and rhs
    rhs_func = ca.Function("f_ode", [x, u], [rhs])

    ocp.model.disc_dyn_expr = get_disc_dyn_expr(rhs, x, u, None, dt)

    ocp.model.name = name
    ocp.model.x = x
    ocp.model.u = u

    xdot = ca.SX.sym("xdot", x.shape)
    ocp.model.xdot = xdot
    ocp.model.f_impl_expr = xdot - rhs

    ocp.dims.nx = x.size()[0]
    ocp.dims.nu = u.size()[0]

    # Forward OCP with SQP and GN Hessian
    # TODO: Define external cost based on el, ec, omega, control inputs and negative path velocity

    # TODO: Define external cost function
    ocp.model.cost_expr_ext_cost_0 = get_cost_expr_ext_cost(ocp=ocp)
    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = get_cost_expr_ext_cost(ocp=ocp)
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = get_cost_expr_ext_cost_e(ocp=ocp)
    ocp.cost.cost_type_e = "EXTERNAL"







    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"

    # constraints
    ocp.constraints.idxbx = np.array([2])
    ocp.constraints.lbx = np.array([model_params["lower_bound_z"]])
    ocp.constraints.ubx = np.array([model_params["upper_bound_z"]])

    ocp.constraints.idxbx_e = np.array([2, 7, 8, 9])
    ocp.constraints.lbx_e = np.array([model_params["lower_bound_z"], 0.0, 0.0, 0.0])
    ocp.constraints.ubx_e = np.array([model_params["upper_bound_z"], 0.0, 0.0, 0.0])

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
    ocp.solver_options.nlp_solver_max_iter = (
        100  # dont set to 1!! does not update controls
    )
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


def get_disc_dyn_expr(rhs, x, u, p, dt: float) -> ca.SX:
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
        *find_param_in_p_or_p_global(
            [f"xref{i}" for i in range(1, 13)] + ["uref"], model
        ).values()
    )  # type:ignore


def cost_matrix_casadi(model: AcadosModel) -> ca.SX:
    L = ca.diag(
        ca.vertcat(
            *find_param_in_p_or_p_global(
                [f"L{i}{i}" for i in range(1, 17)], model
            ).values()
        )
    )
    L_offdiag = find_param_in_p_or_p_global(["Lloweroffdiag"], model)["Lloweroffdiag"]

    assign_lower_triangular(L, L_offdiag)

    return L @ L.T
