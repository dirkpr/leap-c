from pathlib import Path

import casadi as ca
import numpy as np
from acados_template import AcadosOcp
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
        N_horizon: int = 20,
        T_horizon: float = 2.0,
        discount_factor: float = 0.99,
        n_batch: int = 64,
        export_directory: Path | None = None,
        export_directory_sensitivity: Path | None = None,
        throw_error_if_u0_is_outside_ocp_bounds: bool = True,
    ):
        params = (
            {
                "m": 1.0,
                "l": 1.0,
                "I": 1.0,
                "xy_e_ref": np.array([1.0, 1.0]),
                "q_sqrt_diag": np.array([1.0, 1.0]),
                "r_sqrt_diag": np.array([1.0, 1.0]),
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
        )
        configure_ocp_solver(ocp=ocp, exact_hess_dyn=True)

        self.given_default_param_dict = params
        super().__init__(
            ocp=ocp,
            n_batch_max=n_batch,
            export_directory=export_directory,
            export_directory_sensitivity=export_directory_sensitivity,
            throw_error_if_u0_is_outside_ocp_bounds=throw_error_if_u0_is_outside_ocp_bounds,
        )


def compute_inertia_matrix(
    theta: Iterable, n_link: int, length: float, mass: float, inertia: float
) -> np.ndarray:
    """
    Calculate the inertia matrix M(θ).

    Parameters:
    -----------
    theta : array-like
        Joint angles [θ₁, θ₂, ..., θₙ]
    n_link : int
        Number of links
    length : float
        Length of each link
    mass : float
        Mass of each link
    inertia : float
        Moment of inertia of each link

    Note: Every link has the same length, mass, and moment of inertia.

    Returns:
    --------
    M : ca.SX
        n×n inertia matrix
    """
    M = ca.SX.zeros(n_link, n_link)

    for i in range(n_link):
        for j in range(n_link):
            # Start from max(i,j) to account for coupling inertias
            k_start = max(i, j)

            for k in range(k_start, n_link):
                # Sum of angles from m=i+1 to k
                sum_angles = ca.sum1(theta[k_start : k + 1])
                M[i, j] += mass * (length**2) * ca.cos(sum_angles)

            # Add moment of inertia term (only for diagonal elements)
            if i == j:
                M[i, j] += inertia

    return M


def compute_coriolis_matrix(M_expr, theta, theta_dot):
    """
    Compute the Coriolis matrix using the property that Ṁ - 2C is skew-symmetric, which means Ṁ - 2C = -(Ṁ - 2C)ᵀ.

    Parameters:
    -----------
    M_expr : casadi.SX
        Symbolic expression for the inertia matrix M(θ)
    theta : casadi.SX
        Symbolic joint angle vector
    theta_dot : casadi.SX
        Symbolic joint velocity vector

    Returns:
    --------
    C : casadi.SX
        Coriolis matrix expression
    """
    n = theta.shape[0]  # Number of joints

    # Initialize the Coriolis matrix
    C = ca.SX.zeros(n, n)

    # Use the property that Ṁ - 2C is skew-symmetric
    # Therefore: 2C = Ṁ - S where S is a skew-symmetric matrix
    # Since C has to satisfy the energy conservation principle, it can be shown that:
    # C[i,j] = Σₖ (0.5 * (∂M[i,j]/∂θₖ + ∂M[i,k]/∂θⱼ - ∂M[j,k]/∂θᵢ)) * θ̇ₖ

    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Compute the Christoffel symbols and form the Coriolis matrix
                dM_ij_dtheta_k = ca.jacobian(M_expr[i, j], theta[k])
                dM_ik_dtheta_j = ca.jacobian(M_expr[i, k], theta[j])
                dM_jk_dtheta_i = ca.jacobian(M_expr[j, k], theta[i])

                C[i, j] += (
                    0.5
                    * (dM_ij_dtheta_k + dM_ik_dtheta_j - dM_jk_dtheta_i)
                    * theta_dot[k]
                )

    return C


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
    m, l, I = find_param_in_p_or_p_global(["m", "l", "I"], ocp.model).values()
    # Get inertia matrix M(θ)
    M = compute_inertia_matrix(
        theta=theta, n_link=ocp.dims.nx // 2, length=l, mass=m, inertia=I
    )
    # Get Coriolis matrix C(θ, dθ̇ )
    C = compute_coriolis_matrix(M_expr=M, theta=theta, theta_dot=dtheta)

    # Get joint torques τ from u
    tau = ocp.model.u

    # Compute angular accelerations using the equation of motion M(θ) * dθ̈ + C(θ, dθ̇ ) * dθ̇ = τ
    ddtheta = ca.solve(M, tau - ca.mtimes(C, dtheta))

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

    return compute_position_cost(ocp=ocp) + compute_torque_cost(ocp=ocp)
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

    ocp.dims.nx = 2 * n_link
    ocp.dims.nu = n_link

    # ocp.model.p = struct_symSX([entry("u_wind", shape=(2, 1))])

    ocp = translate_learnable_param_to_p_global(
        nominal_param=nominal_param, learnable_param=learnable_params, ocp=ocp
    )

    ocp.model.f_expl_expr = get_f_expl_expr(ocp=ocp)
    # ocp.model.disc_dyn_expr = get_disc_dyn_expr(ocp=ocp)

    # ocp.model.cost_expr_ext_cost_0 = get_cost_expr_nonlinear_ls_cost(ocp=ocp)
    # ocp.cost.cost_type_0 = "NONLINEAR_LS"
    ocp.model.cost_expr_ext_cost = get_intermediate_cost_expr(ocp=ocp)
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = get_terminal_cost_expr(ocp=ocp)
    ocp.cost.cost_type_e = "EXTERNAL"

    # Initial state for n_link robot. All angles and velocities should be zero.
    ocp.constraints.x0 = np.zeros((ocp.dims.nx,))

    # Box constraints on u
    ocp.constraints.lbu = np.array([-100] * ocp.dims.nu)  # [Nm] Minimum torque.
    ocp.constraints.ubu = np.array([+100] * ocp.dims.nu)  # [Nm] Maximum torque.
    ocp.constraints.idxbu = np.arange(0, ocp.dims.nu)

    # Box constraints on x
    ocp.constraints.lbx = np.array(
        [
            [np.deg2rad(-120)]  # [rad] Minimum angle constraint.
            * np.ones(ocp.dims.nx // 2),
            [np.deg2rad(-120)]  # [rad/s] Minimum angular velocity constraint.
            * np.ones(ocp.dims.nx // 2),
        ]
    )
    ocp.constraints.ubx = -ocp.constraints.lbx
    ocp.constraints.idxbx = np.arange(0, ocp.dims.nx)

    # Nonlinear constraints on x for joint positions

    # TODO: Define the joint positions via foward kinematics. Remain inside the workspace.
    # TODO: Define the obstacle avoidance constraints.

    xy = forward_kinematics(
        theta=ocp.model.x[0 : ocp.dims.nx // 2],
        l=find_param_in_p_or_p_global(["l"], ocp.model)["l"],
    )
    x = ca.reshape(xy[0, :], -1, 1)
    y = ca.reshape(xy[1, :], -1, 1)

    xmin = -10.0
    xmax = +10.0
    ymin = -10.0
    ymax = +10.0

    ocp.model.con_h_expr = ca.vertcat(x, y)
    ocp.constraints.lh = np.array(
        [xmin] * n_link + [ymin] * n_link
    )  # Define lower bounds for joint positions
    ocp.constraints.uh = np.array(
        [xmax] * n_link + [ymax] * n_link
    )  # Define upper bounds for joint positions

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
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.with_value_sens_wrt_params = False
    ocp.solver_options.with_solution_sens_wrt_params = False
    ocp.solver_options.with_batch_functionality = True
