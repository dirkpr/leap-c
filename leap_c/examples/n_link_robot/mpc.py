from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi.tools import struct_symSX
from leap_c.examples.util import (
    find_param_in_p_or_p_global,
    translate_learnable_param_to_p_global,
)
from leap_c.mpc import Mpc
from typing import Iterable


class NLinkRobotMpc(Mpc):
    """docstring for NLinkRobotMpc."""

    def __init__(
        self,
        params: dict[str, np.ndarray] | None = None,
        learnable_params: list[str] | None = None,
        N_horizon: int = 100,
        T_horizon: float = 10.0,
        discount_factor: float = 0.99,
        n_batch: int = 64,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
        throw_error_if_u0_is_outside_ocp_bounds: bool = True,
        n_link: int = 2,
    ):
        params = (
            {
                "m": 1.0,
                "l": 1.0,
                "xy_e_ref": np.array([1.0, 1.0]),
                "q_sqrt_diag": np.array([10.0, 10.0]),
                "r_sqrt_diag": np.array([0.05, 0.05]),
            }
            if params is None
            else params
        )

        learnable_params = learnable_params if learnable_params is not None else []

        print("learnable_params: ", learnable_params)

        ocp = export_parametric_ocp(
            nominal_param=params,
            learnable_params=learnable_params,
            N_horizon=N_horizon,
            tf=T_horizon,
            n_link=n_link,
        )

        ocp_solver = AcadosOcpSolver(ocp)

        theta_ref = np.array([np.deg2rad(45.0)] * n_link)
        # theta_ref = np.array([np.deg2rad(-10.0), np.deg2rad(45.0)])
        xy_ee_ref = forward_kinematics(theta_ref, params["l"])[-1, :]

        yref = np.concatenate([xy_ee_ref, np.zeros(ocp.dims.nu)])
        yref_e = xy_ee_ref

        print("theta_ref", theta_ref)

        for stage in range(ocp_solver.acados_ocp.solver_options.N_horizon):
            ocp_solver.cost_set(stage, "yref", yref)

        ocp_solver.cost_set(
            ocp_solver.acados_ocp.solver_options.N_horizon, "yref", yref_e
        )

        x0 = np.zeros((ocp.dims.nx,))
        ocp_solver.solve_for_x0(x0, fail_on_nonzero_status=True)

        print("status", ocp_solver.get_status())

        x = np.array(
            [
                ocp_solver.get(stage, "x")
                for stage in range(ocp_solver.acados_ocp.solver_options.N_horizon)
            ]
        )

        print("x", x)

        exit(0)

        configure_ocp_solver(ocp=ocp, exact_hess_dyn=True)

        self.given_default_param_dict = params
        super().__init__(
            ocp=ocp,
            n_batch_max=n_batch,
            export_directory=export_directory,
            export_directory_sensitivity=export_directory_sensitivity,
            throw_error_if_u0_is_outside_ocp_bounds=throw_error_if_u0_is_outside_ocp_bounds,
        )


def compute_linear_velocity_jacobian(q, l, m):
    n_link = q.shape[0]
    Jvc = []

    # First joint at the origin
    joint_pos = ca.SX.zeros(2)
    # Remaining joints
    for i in range(n_link):
        cm_pos = joint_pos + 0.5 * l * ca.vertcat(
            ca.cos(ca.sum1(q[: i + 1])), ca.sin(ca.sum1(q[: i + 1]))
        )

        joint_pos = joint_pos + l * ca.vertcat(
            ca.cos(ca.sum1(q[: i + 1])), ca.sin(ca.sum1(q[: i + 1]))
        )

        Jvc.append(ca.jacobian(cm_pos, q))

    return Jvc


def compute_coriolis_matrix(theta, thetadot, length, mass):
    n_link = theta.shape[0]
    D = compute_inertia_matrix(theta, length, mass)
    C = ca.SX.zeros(n_link, n_link)
    for k in range(n_link):
        for j in range(n_link):
            for i in range(n_link):
                C[k, j] += (
                    0.5
                    * (
                        ca.jacobian(D[k, j], theta[i])
                        + ca.jacobian(D[k, i], theta[j])
                        - ca.jacobian(D[i, j], theta[k])
                    )
                    * thetadot[i]
                )

    return C


def compute_I(theta, length, mass):
    I_cm = (1 / 12) * mass * length**2
    n_link = theta.shape[0]
    I = ca.SX.zeros(n_link, n_link)
    for i in range(n_link):
        I[: i + 1, : i + 1] += I_cm * ca.SX.ones((i + 1, i + 1))

    return I


def compute_inertia_matrix(theta, length, mass):
    D = ca.SX.zeros(theta.shape[0], theta.shape[0])
    for Jvc_i in compute_linear_velocity_jacobian(theta, length, mass):
        D += mass * Jvc_i.T @ Jvc_i

    D += compute_I(theta, length, mass)

    return D


def forward_kinematics(
    theta: ca.SX | np.ndarray, l: ca.SX | float
) -> tuple[float, float]:
    """
    Calculate the joint positions given joint angles.

    Parameters:
    -----------
    theta : array-like
        Joint angles [θ₁, θ₂, ..., θₙ]

    Returns:
    --------
        Joint positions in Cartesian space
    """

    # Check if theta is a CasADi SX object
    if isinstance(theta, ca.SX):
        xy = ca.SX.zeros(theta.shape[0], 2)
        xy[0, :] = l * ca.vertcat(ca.cos(theta[0]), ca.sin(theta[0]))
        for i in range(1, theta.shape[0]):
            xy[i, 0] = xy[i - 1, 0] + l * ca.cos(ca.sum1(theta[0 : i + 1]))
            xy[i, 1] = xy[i - 1, 1] + l * ca.sin(ca.sum1(theta[0 : i + 1]))

        return xy

    # Check if theta is a numpy array
    if isinstance(theta, np.ndarray):
        xy = np.zeros((theta.shape[0], 2))
        xy[0, :] = l * np.array([np.cos(theta[0]), np.sin(theta[0])])
        for i in range(1, theta.shape[0]):
            xy[i, 0] = xy[i - 1, 0] + l * np.cos(np.sum(theta[0 : i + 1]))
            xy[i, 1] = xy[i - 1, 1] + l * np.sin(np.sum(theta[0 : i + 1]))

        return xy

    raise ValueError("theta must be either a CasADi SX object or a numpy array.")


def get_f_expl_expr(ocp: AcadosOcp) -> ca.SX:
    """
    Compute the right-hand side of the dynamics equations.

    M(θ) * dθ̈ + C(θ, dθ̇ ) * dθ̇ = τ
    where:
    - M(θ) is the inertia matrix
    - C(θ, dθ̇ ) is the Coriolis matrix
    - τ is the joint torques
    """
    # Extract joint angles from x
    theta = ocp.model.x[0 : ocp.dims.nx // 2]
    # Extract joint velocities from x
    dtheta = ocp.model.x[ocp.dims.nx // 2 :]
    # Get m, l, I from p or p_global
    m, l = find_param_in_p_or_p_global(["m", "l"], ocp.model).values()
    # Get inertia matrix M(θ)
    M = compute_inertia_matrix(theta=theta, length=l, mass=m)

    # Get Coriolis matrix C(θ, dθ̇ )
    C = compute_coriolis_matrix(theta=theta, thetadot=dtheta, length=l, mass=m)

    # Get joint torques τ from u
    tau = ocp.model.u

    # Compute angular accelerations using the equation of motion M(θ) * dθ̈ + C(θ, dθ̇ ) * dθ̇ = τ
    ddtheta = ca.solve(M, tau - ca.mtimes(C, dtheta))
    # ddtheta = ca.mtimes(ca.inv(M), tau - ca.mtimes(C, dtheta))

    return ca.vertcat(dtheta, ddtheta)


def get_disc_dyn_expr(ocp: AcadosOcp) -> ca.SX:
    pass


def create_diag_matrix(
    v_diag: np.ndarray | ca.SX,
) -> np.ndarray | ca.SX:
    if any(isinstance(i, ca.SX) for i in [v_diag]):
        return ca.diag(v_diag)
    else:
        return np.diag(v_diag)


def compute_position_cost(ocp: AcadosOcp) -> ca.SX:
    """
    Compute the cost associated with the end-effector position.

    Parameters:
    -----------
    ocp : AcadosOcp
        The OCP object containing the model and parameters.

    Returns:
    --------
    cost : casadi.SX
        The cost associated with the end-effector position.
    """

    # Get the end-effector position using forward kinematics
    q_sqrt_diag = find_param_in_p_or_p_global(["q_sqrt_diag"], ocp.model)["q_sqrt_diag"]
    Q = create_diag_matrix(q_sqrt_diag**2)

    # Compute end-effector position
    l = find_param_in_p_or_p_global(["l"], ocp.model)["l"]
    theta = ocp.model.x[0 : ocp.model.x.shape[0] // 2]
    xy_e = forward_kinematics(theta, l)[-1, :]  # Get the last position

    # Reshape xy_e to column vector
    xy_e = ca.reshape(xy_e, -1, 1)

    # Extract desired end-effector position from p_global
    xy_e_ref = find_param_in_p_or_p_global(["xy_e_ref"], ocp.model)["xy_e_ref"]

    return ca.mtimes(ca.mtimes((xy_e - xy_e_ref).T, Q), (xy_e - xy_e_ref))


def compute_torque_cost(ocp: AcadosOcp) -> ca.SX:
    """
    Compute the cost associated with the joint torques.

    Parameters:
    -----------
    ocp : AcadosOcp
        The OCP object containing the model and parameters.

    Returns:
    --------
    cost : casadi.SX
        The cost associated with the joint torques.
    """
    r_sqrt_diag = find_param_in_p_or_p_global(["r_sqrt_diag"], ocp.model)["r_sqrt_diag"]
    R = create_diag_matrix(r_sqrt_diag**2)

    return ca.mtimes(ca.mtimes((ocp.model.u).T, R), ocp.model.u)


def get_intermediate_cost_expr(ocp: AcadosOcp) -> ca.SX:
    # TODO: Think about using the inertia matrix M(θ) to compute the torque cost?
    # TODO: Think about using the inverse kinematics solution from endeffector to joint
    # angles to compute the cost?

    pos_cost = compute_position_cost(ocp=ocp)
    torque_cost = compute_torque_cost(ocp=ocp)

    return pos_cost + torque_cost
    # return compute_position_cost(ocp=ocp) + compute_torque_cost(ocp=ocp)


def get_terminal_cost_expr(ocp: AcadosOcp) -> ca.SX:
    # TODO: Think about using the inertia matrix M(θ) to compute the torque cost?
    # TODO: Think about using the inverse kinematics solution from endeffector to joint
    # angles to compute the cost?

    return compute_position_cost(ocp=ocp)


def export_parametric_ocp(
    nominal_param: dict[str, np.ndarray],
    name: str = "n_link_robot",
    learnable_params: list[str] | None = None,
    N_horizon: int = 50,
    tf: float = 5.0,
    n_link: int = 2,
) -> AcadosOcp:
    ocp = AcadosOcp()

    ocp.solver_options.tf = tf
    ocp.solver_options.N_horizon = N_horizon

    ocp.model.name = name

    ocp.model.x = ca.vertcat(
        ca.SX.sym("theta", n_link),
        ca.SX.sym("dtheta", n_link),
    )

    ocp.model.u = ca.SX.sym("tau", n_link)

    ocp.dims.nx = ocp.model.x.shape[0]
    ocp.dims.nu = ocp.model.u.shape[0]

    # ocp.model.p = struct_symSX([entry("u_wind", shape=(2, 1))])

    ocp = translate_learnable_param_to_p_global(
        nominal_param=nominal_param, learnable_param=learnable_params, ocp=ocp
    )

    ocp.model.f_expl_expr = get_f_expl_expr(ocp=ocp)
    # ocp.model.disc_dyn_expr = get_disc_dyn_expr(ocp=ocp)

    # ocp.model.cost_expr_ext_cost_0 = get_cost_expr_nonlinear_ls_cost(ocp=ocp)
    # ocp.cost.cost_type_0 = "NONLINEAR_LS"

    if False:
        ocp.model.cost_expr_ext_cost = get_intermediate_cost_expr(ocp=ocp)
        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_e = get_terminal_cost_expr(ocp=ocp)
        ocp.cost.cost_type_e = "EXTERNAL"
    else:
        xy = forward_kinematics(
            theta=ocp.model.x[0 : ocp.dims.nx // 2],
            l=find_param_in_p_or_p_global(["l"], ocp.model)["l"],
        )

        xy_ee = ca.reshape(xy[-1, :], -1, 1)

        theta_ref = np.array([np.deg2rad(10.0)] * n_link)

        xy_ee_ref = forward_kinematics(theta=theta_ref, l=1.0)[-1, :]

        ocp.dims.ny = xy_ee.shape[0] + ocp.dims.nu
        ocp.dims.ny_e = xy_ee.shape[0]
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.W = np.diag(np.array([10.0, 10.0] + [1.0] * ocp.dims.nu))
        ocp.model.cost_y_expr = ca.vertcat(xy_ee, ocp.model.u)
        ocp.cost.yref = np.concatenate([xy_ee_ref, np.zeros(ocp.dims.nu)])
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W_e = np.diag(np.array([10.0, 10.0]))
        ocp.model.cost_y_expr_e = xy_ee
        ocp.cost.yref_e = xy_ee_ref

    # Initial state for n_link robot. All angles and velocities should be zero.
    ocp.constraints.x0 = np.zeros((ocp.dims.nx,))

    # # Box constraints on u
    # ocp.constraints.lbu = np.array([-100] * ocp.dims.nu)  # [Nm] Minimum torque.
    # ocp.constraints.ubu = np.array([+100] * ocp.dims.nu)  # [Nm] Maximum torque.
    # ocp.constraints.idxbu = np.arange(0, ocp.dims.nu)

    # # Box constraints on x
    # ocp.constraints.lbx = np.array(
    #     [
    #         [np.deg2rad(-120)]  # [rad] Minimum angle constraint.
    #         * np.ones(ocp.dims.nx // 2),
    #         [np.deg2rad(-120)]  # [rad/s] Minimum angular velocity constraint.
    #         * np.ones(ocp.dims.nx // 2),
    #     ]
    # )
    # ocp.constraints.ubx = -ocp.constraints.lbx
    # ocp.constraints.idxbx = np.arange(0, ocp.dims.nx)

    # Nonlinear constraints on x for joint positions

    # TODO: Define the joint positions via foward kinematics. Remain inside the workspace.
    # TODO: Define the obstacle avoidance constraints.

    # xy = forward_kinematics(
    #     theta=ocp.model.x[0 : ocp.dims.nx // 2],
    #     l=find_param_in_p_or_p_global(["l"], ocp.model)["l"],
    # )
    # x = ca.reshape(xy[0, :], -1, 1)
    # y = ca.reshape(xy[1, :], -1, 1)

    # xmin = -10.0
    # xmax = +10.0
    # ymin = -10.0
    # ymax = +10.0

    # ocp.model.con_h_expr = ca.vertcat(x, y)
    # ocp.constraints.lh = np.array(
    #     [xmin] * n_link + [ymin] * n_link
    # )  # Define lower bounds for joint positions
    # ocp.constraints.uh = np.array(
    #     [xmax] * n_link + [ymax] * n_link
    # )  # Define upper bounds for joint positions

    # #############################
    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp


def configure_ocp_solver(ocp: AcadosOcp, exact_hess_dyn: bool):
    # ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    # ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.with_value_sens_wrt_params = False
    ocp.solver_options.with_solution_sens_wrt_params = False
    ocp.solver_options.with_batch_functionality = True
