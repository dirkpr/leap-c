import numpy as np
from scipy.linalg import eigvals, solve_discrete_are


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
        B = rng.standard_normal((n, m))
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
