import casadi as ca
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from scipy.linalg import eigvals, solve_discrete_are

from leap_c.examples.lqr.env import LqrEnv, LqrEnvConfig
from leap_c.ocp.acados.parameters import AcadosParameter


def sample_pd_matrix(n: int, rng: np.random.Generator, eps: float = 1e-3) -> np.ndarray:
    """Sample a random positive-definite matrix of size nxn."""
    M = rng.standard_normal((n, n))
    return M.T @ M + eps * np.eye(n)


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    cols = [B]
    for _ in range(n - 1):
        cols.append(A @ cols[-1])
    return np.hstack(cols)


def is_controllable(A: np.ndarray, B: np.ndarray, tol: float = 1e-8) -> bool:
    return np.linalg.matrix_rank(controllability_matrix(A, B), tol=tol) == A.shape[0]


def sample_lqr_matrices(
    n: int,
    m: int,
    seed: int | None = None,
    max_attempts: int = 1000,
    pd_eps: float = 1e-3,
) -> dict:
    """Sample random LQR matrices (A, B, Q, R).

    Parameters
    ----------
    n           : state dimension
    m           : input dimension
    seed        : random seed for reproducibility
    max_attempts: max resampling attempts for (A, B) controllability
    pd_eps      : regularisation added to M'M to enforce strict PD for Q and R

    Returns:
    -------
    dict with keys 'A', 'B', 'Q', 'R', 'P' (solution to DARE), 'K' (optimal gain)
    """
    rng = np.random.default_rng(seed)

    for attempt in range(max_attempts):
        print(attempt)
        A = rng.standard_normal((n, n))
        A /= np.max(np.abs(np.linalg.eig(A)[0]))
        B = 0.2 * rng.standard_normal((n, m))
        if is_controllable(A, B):
            break
    else:
        raise RuntimeError(
            f"Could not sample a controllable (A, B) pair in {max_attempts} attempts."
        )

    Q = sample_pd_matrix(n, rng, eps=pd_eps)
    R = sample_pd_matrix(m, rng, eps=pd_eps)
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)  # optimal gain: u = -K x

    assert np.all(np.abs(eigvals(A - B @ K)) < 1), (
        "Closed-loop system is unstable (eigenvalues outside unit disk)."
    )

    return A, B, Q, R, P, K


def make_default_lqr_params(N_horizon: int = 100) -> tuple[AcadosParameter, ...]:
    """Return a tuple of default parameters for the LQR planner.

    Args:
        N_horizon: The number of steps in the MPC horizon

    Returns:
        Tuple of AcadosParameter objects for the mass-spring-damper system.

    Note: The default parameters do not match the true parameter values used in the environment.
    """
    params = []
    params.extend(
        [
            AcadosParameter(
                name="q_diag_sqrt",
                default=np.sqrt(np.array([5.0, 0.2])),
                space=gym.spaces.Box(
                    low=np.sqrt(np.array([0.1, 0.01])),
                    high=np.sqrt(np.array([10.0, 1.0])),
                    dtype=np.float64,
                ),
            ),
            AcadosParameter(
                name="r_diag_sqrt",
                default=np.sqrt(np.array([0.08])),
                space=gym.spaces.Box(
                    low=np.sqrt(np.array([0.001])),
                    high=np.sqrt(np.array([0.1])),
                    dtype=np.float64,
                ),
            ),
            AcadosParameter(
                name="p_diag_sqrt",
                default=np.sqrt(np.array([5.0, 0.5])),
                space=gym.spaces.Box(
                    low=np.sqrt(np.array([1.0, 0.1])),
                    high=np.sqrt(np.array([10.0, 1.0])),
                    dtype=np.float64,
                ),
                interface="learnable",
            ),
        ]
    )

    return tuple(params)


def is_PSD(A, tol=1e-8):
    E = np.linalg.eigvalsh(A)
    return np.all(E > -tol)


def make_psd(A, delta=1e-6):
    B = (A + A.T) / 2.0
    w, Q = np.linalg.eigh(B)
    w = np.maximum(w, delta)
    B_psd = (Q * w) @ Q.T
    return (B_psd + B_psd.T) / 2.0


def export_parametric_ocp(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    P: np.ndarray,
    N: np.ndarray,
    N_horizon: int,
    x0: np.ndarray | None = None,
    name: str = "lqr",
) -> AcadosOcp:
    """Export the LQR OCP.

    Args:
        A: State transition matrix.
        B: Control input matrix.
        Q: State cost matrix.
        R: Control cost matrix.
        P: Terminal state cost matrix.
        N: Cross-term cost matrix.
        N_horizon: Number of time steps in the horizon.
        name: Name of the OCP model.
        x0: Initial state. If None, a default value is used.

    Returns:
        AcadosOcp: The configured OCP object.
    """
    ocp = AcadosOcp()

    # param_manager.assign_to_ocp(ocp)

    # Model
    ocp.model.name = name

    ocp.model.x = ca.SX.sym("x", A.shape[1])
    ocp.model.u = ca.SX.sym("u", B.shape[1])

    ocp.model.disc_dyn_expr = A @ ocp.model.x + B @ ocp.model.u

    # Cost function
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = np.block([[Q, N], [N.T, R]])
    ocp.model.cost_y_expr = ca.vertcat(ocp.model.x, ocp.model.u)
    ocp.cost.yref = np.zeros((ocp.cost.W.shape[0],))

    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = P
    ocp.model.cost_y_expr_e = ocp.model.x
    ocp.cost.yref_e = np.zeros((ocp.cost.W_e.shape[0],))

    # Initial condition
    if x0 is None:
        ocp.constraints.x0 = np.zeros((A.shape[1],))

    # Solver options
    ocp.solver_options.tf = N_horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

    return ocp


def plot_ocp_solution():
    N_horizon = 5
    n = 2
    m = 1
    x0 = np.array([1.5, -1.0])

    A, B, Q, R, P, K = sample_lqr_matrices(n=n, m=m, seed=None)

    def cost(x, u):
        return 0.5 * x.T @ Q @ x + 0.5 * u.T @ R @ u

    ocp = export_parametric_ocp(A, B, Q, R, P, N_horizon, x0)
    ocp.translate_initial_cost_term_to_external(cost_hessian=ocp.solver_options.hessian_approx)
    ocp.translate_intermediate_cost_term_to_external(cost_hessian=ocp.solver_options.hessian_approx)
    ocp.translate_terminal_cost_term_to_external(cost_hessian=ocp.solver_options.hessian_approx)

    ocp_solver = AcadosOcpSolver(ocp)

    _ = ocp_solver.solve()

    X = np.array([ocp_solver.get(i, "x") for i in range(N_horizon + 1)])
    U = np.array([ocp_solver.get(i, "u") for i in range(N_horizon)])

    costs = np.array([cost(X[i], U[i]) for i in range(N_horizon)])

    plt.figure()
    plt.plot(costs, marker="o")
    plt.xlabel("Time step")
    plt.ylabel("Cost")
    plt.grid(True, alpha=0.3)
    plt.show()


def run_ocp_solver_closed_loop(
    env: LqrEnv,
    ocp_solver: AcadosOcpSolver,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Run the OCP solver in closed-loop on the LQR environment.

    At each step the current state is set as the initial condition of the OCP,
    the first optimal control input is applied to the environment, and the
    resulting trajectory is plotted.

    Args:
        env: LQR environment.
        ocp_solver: An AcadosOcpSolver object configured with the LQR OCP.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, U, rewards) where X has shape (T+1, n), U has shape (T, m),
        and rewards is a list of length T.
    """
    obs, _ = env.reset(seed=seed)
    x0 = obs.astype(np.float64)

    X: list[np.ndarray] = [x0.copy()]
    U: list[np.ndarray] = []
    rewards: list[float] = []

    terminated = truncated = False
    while not (terminated or truncated):
        x = obs.astype(np.float64)

        # Update the initial-state equality constraint
        ocp_solver.set(0, "lbx", x)
        ocp_solver.set(0, "ubx", x)

        status = ocp_solver.solve()
        if status != 0:
            print(f"Warning: OCP solver returned status {status} at step {len(U)}")

        u = ocp_solver.get(0, "u")
        obs, reward, terminated, truncated, _ = env.step(u)

        X.append(obs.astype(np.float64))
        U.append(u.copy())
        rewards.append(float(reward))

    env.close()

    X_arr = np.array(X)
    U_arr = np.array(U)

    return X_arr, U_arr, rewards


def run_state_feedback_closed_loop(
    env: LqrEnv,
    K: np.ndarray,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    obs, _ = env.reset(seed=seed)

    X: list[np.ndarray] = [obs.copy()]
    U: list[np.ndarray] = []
    rewards: list[float] = []

    terminated = truncated = False
    while not (terminated or truncated):
        x = obs.astype(np.float64)
        u = -K @ x

        obs, reward, terminated, truncated, _ = env.step(u)

        X.append(obs.astype(np.float64))
        U.append(u.copy())
        rewards.append(float(reward))

    env.close()

    X_arr = np.array(X)
    U_arr = np.array(U)

    return X_arr, U_arr, rewards


if __name__ == "__main__":
    seed = 0
    n = 10
    m = 10
    N_horizon = 10

    A, B, Q, R, P, K = sample_lqr_matrices(n=n, m=m, seed=seed)
    N = np.zeros((n, m))  # No cross-term in the cost

    A_hat = A + 0.1 * np.random.default_rng(seed).standard_normal(A.shape)
    A_hat = A_hat / np.max(np.abs(np.linalg.eig(A_hat)[0]))
    B_hat = B + 0.1 * np.random.default_rng(seed).standard_normal(B.shape)

    N_hat = N + A.T @ P @ B - A_hat.T @ P @ B_hat
    Q_hat = Q + A.T @ P @ A - A_hat.T @ P @ A_hat
    R_hat = R + B.T @ P @ B - B_hat.T @ P @ B_hat

    P_hat = solve_discrete_are(A_hat, B_hat, Q_hat, R_hat, s=N_hat)

    K_hat = np.linalg.solve(R_hat + B_hat.T @ P_hat @ B_hat, B_hat.T @ P_hat @ A_hat + N_hat.T)

    eig_hat = eigvals(A_hat - B_hat @ K_hat)

    assert np.all(np.abs(eig_hat) < 1), "Mismatched model is unstable"

    assert np.allclose(P_hat, P, atol=1e-6), (
        "Mismatched model has significantly different optimal cost-to-go matrix"
    )

    assert np.allclose(K_hat, K, atol=1e-6), (
        "Mismatched model has significantly different optimal gain matrix"
    )

    assert np.allclose(N_hat, N_hat.T, atol=1e-6), "Cross-term matrix N_hat is not symmetric"

    # Ensure the cost matrices are PSD
    W_hat = np.block([[Q_hat, N_hat], [N_hat.T, R_hat]])

    if not is_PSD(W_hat):
        W_hat = make_psd(W_hat, delta=1e-6)

    N_hat = W_hat[:n, n:]
    Q_hat = W_hat[:n, :n]
    R_hat = W_hat[n:, n:]

    ####

    cfg = LqrEnvConfig(A=A, B=B, Q=Q, R=R, P=P, dt=1.0, max_time=20.0, init_state_bound=1.0)
    env = LqrEnv(cfg=cfg)

    # Simulate the OCP solution with the true model parameters
    X_ocp, U_ocp, rewards_ocp = run_ocp_solver_closed_loop(
        env,
        AcadosOcpSolver(
            export_parametric_ocp(
                A,
                B,
                Q,
                R,
                P,
                N,
                N_horizon,
            )
        ),
        seed=seed,
    )

    # Simulate the OCP solution with the mismatched model parameters
    X_ocp_hat, U_ocp_hat, rewards_ocp_hat = run_ocp_solver_closed_loop(
        env,
        AcadosOcpSolver(
            export_parametric_ocp(
                A_hat,
                B_hat,
                Q_hat,
                R_hat,
                P_hat,
                N_hat,
                N_horizon,
            )
        ),
        seed=seed,
    )

    # Simulate state feedback u=-kX in closed-loop with the true system dynamics and the optimal K
    X_sf, U_sf, rewards_sf = run_state_feedback_closed_loop(env, K, seed=seed)

    n_rows = n + m + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 2 * n_rows), sharex=True)

    t_x_ocp = np.arange(len(X_ocp))
    t_x_ocp_hat = np.arange(len(X_ocp_hat))
    t_x_sf = np.arange(len(X_sf))
    for i in range(n):
        axes[i].plot(t_x_ocp, X_ocp[:, i], label="OCP")
        axes[i].plot(t_x_sf, X_sf[:, i], "--", label="State feedback")
        axes[i].plot(t_x_ocp_hat, X_ocp_hat[:, i], ":k", label="OCP (mismatched)")
        axes[i].axhline(0, color="k", linestyle=":", alpha=0.3)
        axes[i].set_ylabel(f"$x_{{{i}}}$")
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend()

    t_u_ocp = np.arange(len(U_ocp))
    t_u_sf = np.arange(len(U_sf))
    t_u_ocp_hat = np.arange(len(U_ocp_hat))
    for j in range(m):
        axes[n + j].step(t_u_ocp, U_ocp[:, j], where="post", label="OCP")
        axes[n + j].step(t_u_sf, U_sf[:, j], "--", where="post", label="State feedback")
        axes[n + j].step(t_u_ocp_hat, U_ocp_hat[:, j], ":k", where="post", label="OCP (mismatched)")
        axes[n + j].axhline(0, color="k", linestyle=":", alpha=0.3)
        axes[n + j].set_ylabel(f"$u_{{{j}}}$")
        axes[n + j].grid(True, alpha=0.3)
        if j == 0:
            axes[n + j].legend()

    axes[-1].plot(rewards_ocp, label="OCP")
    axes[-1].plot(rewards_sf, "--", label="State feedback")
    axes[-1].plot(rewards_ocp_hat, ":k", label="OCP (mismatched)")
    axes[-1].axhline(0, color="k", linestyle=":", alpha=0.3)
    axes[-1].set_ylabel("Reward")
    axes[-1].set_xlabel("Time step")
    axes[-1].grid(True, alpha=0.3)
    axes[-1].legend()

    fig.suptitle(f"LQR: OCP vs State Feedback with N={N_horizon}", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"lqr_ocp_solution_N_{N_horizon}.png", dpi=300)
    plt.show()
