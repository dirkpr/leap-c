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
        learnable_params: list[str] | None = ["m"],
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


def get_inertia_matrix(
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


def get_coriolis_matrix(
    theta: ca.SX,
    dtheta: ca.SX,
    n_link: int,
    length: ca.SX,
    mass: ca.SX,
):
    """
    Calculate the Coriolis matrix C(θ, θ̇ ).

    Parameters:
    -----------
    theta : array-like
        Joint angles [θ₁, θ₂, ..., θₙ]
    theta_dot : array-like
        Joint angular velocities [θ̇₁, θ̇₂, ..., θ̇ₙ]
    n_link : int
        Number of links
    length : float
        Length of each link
    mass : float
        Mass of each link
    inertia : float
        Moment of inertia of each link

    Returns:
    --------
    C : ndarray
        n×n Coriolis matrix
    """
    C = ca.SX.zeros((n_link, n_link))

    # Pre-compute derivatives of inertia matrix elements
    dM_dtheta = ca.SX.zeros(n_link, n_link, n_link)

    # TODO: Compute dM_dtheta using the symbolic expression for M(θ) instead
    for i in range(n_link):
        for j in range(n_link):
            for k in range(n_link):
                # Only compute derivatives for relevant elements
                if k >= max(i, j):
                    # Sum of angles for the sine term
                    sum_angles = ca.sum1(theta[max(i, j) : k + 1])
                    dM_dtheta[i, j, k] = -mass * (length**2) * ca.sin(sum_angles)

    # Compute Coriolis matrix using Christoffel symbols
    for i in range(n_link):
        for j in range(n_link):
            for k in range(n_link):
                C[i, j] += (
                    0.5
                    * (dM_dtheta[i, j, k] + dM_dtheta[i, k, j] - dM_dtheta[j, k, i])
                    * dtheta[k]
                )

    return C


def forward_kinematics(theta: ca.SX, l: ca.SX) -> tuple[float, float]:
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
    # Cumulative sum of angles for each link
    cum_theta = ca.cumsum(theta)

    # Calculate positions for all joints including end-effector
    # return l * np.cumsum(np.vstack((np.cos(cum_theta), np.sin(cum_theta))), axis=1)
    # TODO: Check that the above euqation is correctly turned into casadi
    return l * ca.vertcat(ca.cos(cum_theta), ca.sin(cum_theta))


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
    M = get_inertia_matrix(
        theta=theta, n_link=ocp.dims.nx // 2, length=l, mass=m, inertia=I
    )
    # Get Coriolis matrix C(θ, dθ̇ )
    C = get_coriolis_matrix(
        theta=theta,
        dtheta=dtheta,
        n_link=ocp.dims.nx // 2,
        length=l,
        mass=m,
        inertia=I,
    )
    # Get joint torques τ from u
    tau = ocp.model.u

    # Compute angular accelerations using the equation of motion M(θ) * dθ̈ + C(θ, dθ̇ ) * dθ̇ = τ
    ddtheta = ca.mtimes(ca.inv(M), tau - ca.mtimes(C, dtheta))

    # Alternatively, can we use the solve function to compute ddtheta?
    # ddtheta = ca.solve(M, tau - ca.mtimes(C, dtheta))

    return ca.vertcat(dtheta, ddtheta)


def get_disc_dyn_expr(ocp: AcadosOcp) -> ca.SX:
    pass


def get_cost_expr_nonlinear_ls_cost(ocp: AcadosOcp) -> ca.SX:
    # TODO: Extract end-effector position from x
    n_link = ocp.model.x // 2
    # TODO: Extract desired end-effector position from p_global
    # TODO: Define the cost function as the weighted squared difference between the
    # end-effector position and the desired position
    # TODO: Add the weighted squared torque cost
    # TODO: Think about using the inertia matrix M(θ) to compute the torque cost?
    # TODO: Think about using the inverse kinematics solution from endeffector to joint
    # angles to compute the cost?
    pass


def get_cost_expr_nonlinear_ls_cost_e(ocp: AcadosOcp) -> ca.SX:
    # TODO: Same as regular stage cost, but no torque cost
    pass


def export_parametric_ocp(
    nominal_param: dict[str, np.ndarray],
    name: str = "n_link_robot",
    learnable_params: list[str] | None = None,
    N_horizon: int = 50,
    tf: float = 2.0,
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

    ocp.model.cost_expr_ext_cost_0 = get_cost_expr_nonlinear_ls_cost(ocp=ocp)
    ocp.cost.cost_type_0 = "NONLINEAR_LS"
    ocp.model.cost_expr_ext_cost = get_cost_expr_nonlinear_ls_cost(ocp=ocp)
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.model.cost_expr_ext_cost_e = get_cost_expr_nonlinear_ls_cost_e(ocp=ocp)
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    # Initial state for n_link robot. All angles and velocities should be zero.
    ocp.constraints.x0 = np.zeros((ocp.dims.nx,))

    # Box constraints on u
    ocp.constraints.lbu = ...  # TODO: Minimum torque constraint. Add later.
    ocp.constraints.ubu = ...  # TODO: Maximum torque constraint. Add later.
    ocp.constraints.idxbu = np.arange(0, ocp.dims.nu)

    # Box constraints on x
    ocp.constraints.lbx = (
        ...
    )  # TODO: Minimum angle constraint. Minimum angular velocity constraint. Add later.
    ocp.constraints.ubx = (
        ...
    )  # TODO: Maximum angle constraint. Maximum angular velocity constraint. Add later.
    ocp.constraints.idxbx = np.arange(0, ocp.dims.nx)

    # Nonlinear constraints on x for joint positions

    # TODO: Define the joint positions via foward kinematics. Remain inside the workspace.
    # TODO: Define the obstacle avoidance constraints.
    ocp.model.con_h_expr = ...
    ocp.constraints.lh = ...  # Define lower bounds for joint positions
    ocp.constraints.uh = ...  # Define upper bounds for joint positions

    # #############################
    if isinstance(ocp.model.p, struct_symSX):
        ocp.model.p = ocp.model.p.cat if ocp.model.p is not None else []

    if isinstance(ocp.model.p_global, struct_symSX):
        ocp.model.p_global = (
            ocp.model.p_global.cat if ocp.model.p_global is not None else None
        )

    return ocp


def configure_ocp_solver(ocp: AcadosOcp, exact_hess_dyn: bool):
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.exact_hess_dyn = exact_hess_dyn
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.with_value_sens_wrt_params = True
    ocp.solver_options.with_solution_sens_wrt_params = True
    ocp.solver_options.with_batch_functionality = True
