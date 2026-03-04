from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.lines import Line2D

from leap_c.examples.utils.matplotlib_env import MatplotlibRenderEnv


@dataclass(kw_only=True)
class LqrEnvConfig:
    """Configuration for the LQR environment.

    Attributes:
        A: Discrete-time state transition matrix (n x n).
        B: Discrete-time input matrix (n x m).
        Q: Positive semi-definite state cost matrix (n x n).
        R: Positive definite control cost matrix (m x m).
        P: Positive definite terminal cost matrix (n x n), typically the DARE solution.
        dt: Duration of one discrete-time step [s]. Used for time tracking only.
        max_time: Maximum simulation time before truncation [s].
        init_state_bound: Symmetric bound for uniform sampling of the initial state.
        state_bound: Symmetric per-dimension bound; episode terminates if exceeded. np.inf disables.
        action_bound: Symmetric per-dimension bound; actions are clipped to this range.
            np.inf disables.
        goal_radius: L2-norm radius of the goal ball around the origin. Episode terminates
            with success when the state enters this ball.
        goal_reward: Extra reward added when the state enters the goal ball.
    """

    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray
    dt: float = 0.1
    max_time: float = 5.0
    init_state_bound: float = 1.0
    state_bound: float = np.inf
    action_bound: float = np.inf
    goal_radius: float = 0.01
    goal_reward: float = 0.0


class LqrEnv(MatplotlibRenderEnv):
    """A gymnasium environment for discrete-time linear quadratic regulation (LQR).

    The dynamics follow the discrete-time linear system:
    ```
        x[k+1] = A x[k] + B u[k]
    ```

    Observation Space:
    ------------------
    The observation is a `ndarray` with shape `(n,)` and dtype `np.float32` representing the
    full state vector, bounded by `[-state_bound, state_bound]` per dimension (`np.inf` if
    `state_bound` is not finite).

    Action Space:
    -------------
    The action is a `ndarray` with shape `(m,)` and dtype `np.float32` representing the control
    input, bounded by `[-action_bound, action_bound]` per dimension (`np.inf` if `action_bound`
    is not finite).

    Reward:
    -------
    The step reward is the negative LQR running cost:
    ```
        r = -(x^T Q x + u^T R u)
    ```
    At truncation (end of episode), the negative terminal cost is added:
    ```
        r_terminal = -(x^T P x)
    ```

    Termination:
    ------------
    The episode terminates if:
    - Any state component exceeds `state_bound` in absolute value (only when finite), or
    - The L2 norm of the state falls within `goal_radius` (goal reached).

    Truncation:
    -----------
    The episode is truncated after `max_time / dt` steps.

    Info:
    -----
    The info dictionary at episode end contains:
    - `"task"`: {`"violation"`: bool, `"success"`: bool}
      - violation: True if state left the bounded region.
      - success: True if the L2 norm of the state is within `goal_radius`.

    Attributes:
        cfg: Configuration object.
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
        state: Current state of shape `(n,)`.
        time: Elapsed time in the current episode [s].
        trajectory: List of states visited during the episode.
        actions: List of actions taken during the episode.
    """

    cfg: LqrEnvConfig
    observation_space: spaces.Box
    action_space: spaces.Box
    state: np.ndarray | None
    time: float
    trajectory: list[np.ndarray]
    actions: list[np.ndarray]

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        cfg: LqrEnvConfig,
        render_mode: str | None = None,
    ):
        """Initialize the LQR environment.

        Args:
            cfg: Configuration holding the system matrices and episode parameters.
            render_mode: Rendering mode; one of `"human"`, `"rgb_array"`, or `None`.
        """
        super().__init__(render_mode=render_mode)

        self.cfg = cfg

        n = cfg.A.shape[0]
        m = cfg.B.shape[1]

        obs_bound = np.full(n, cfg.state_bound, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_bound, obs_bound, dtype=np.float32)

        act_bound = np.full(m, cfg.action_bound, dtype=np.float32)
        self.action_space = spaces.Box(-act_bound, act_bound, dtype=np.float32)

        self.state = None
        self.time = 0.0
        self.trajectory: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []

        # Rendering line objects (initialised in _render_setup)
        self._state_lines: list[Line2D] = []
        self._action_lines: list[Line2D] = []

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step within the environment.

        Args:
            action: Control input of shape `(m,)`.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self.state is None:
            raise ValueError("Environment must be reset before stepping.")

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        ).reshape(-1)

        x = self.state
        u = action.astype(np.float64)

        # Running cost
        state_cost = float(x @ self.cfg.Q @ x)
        control_cost = float(u @ self.cfg.R @ u)
        reward = -(state_cost + control_cost)

        # Dynamics
        self.state = self.cfg.A @ x + self.cfg.B @ u
        self.time += self.cfg.dt
        self.trajectory.append(self.state.copy())
        self.actions.append(u.copy())

        # Termination and truncation
        out_of_bounds = np.isfinite(self.cfg.state_bound) and bool(
            np.any(np.abs(self.state) > self.cfg.state_bound)
        )
        terminated = bool(np.linalg.norm(self.state) <= self.cfg.goal_radius)
        truncated = self.time >= self.cfg.max_time or out_of_bounds

        if truncated and not terminated:
            reward -= float(self.state @ self.cfg.P @ self.state)

        info: dict = {}
        if terminated or truncated:
            info["task"] = {
                "violation": out_of_bounds,
                "success": terminated,
            }

        return self._observation(), reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to a randomly sampled initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Optional dict with:
                - `"state"`: fixed initial state array to use instead of random sampling.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)
        self.time = 0.0
        self.trajectory.clear()
        self.actions.clear()

        if options is not None and "state" in options:
            self.state = np.asarray(options["state"], dtype=np.float64)
        else:
            bound = self.cfg.init_state_bound
            self.state = self.np_random.uniform(-bound, bound, size=(self.cfg.A.shape[0],))

        self.trajectory.append(self.state.copy())

        return self._observation(), {}

    def _observation(self) -> np.ndarray:
        """Return the current state as a float32 array."""
        return self.state.astype(np.float32, copy=True)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Rendering (Matplotlib)
    # ------------------------------------------------------------------

    def _render_setup(self) -> None:
        """One-time initialisation of the Matplotlib figure."""
        if self.render_mode == "human":
            plt.ion()

        n_rows = self.n + 1  # one row per state dim + one for all actions
        self._fig, axes = plt.subplots(n_rows, 1, figsize=(8, 2 * n_rows))
        if n_rows == 1:
            axes = [axes]
        self._ax = axes

        max_steps = int(self.cfg.max_time / self.cfg.dt) + 1
        self._state_lines = []
        for i in range(self.n):
            (line,) = axes[i].plot([], [], linewidth=2)
            axes[i].axhline(0, color="k", linestyle="--", alpha=0.3)
            axes[i].set_xlim(0, max_steps)
            axes[i].set_ylabel(f"$x_{{{i}}}$")
            axes[i].grid(True, alpha=0.3)
            self._state_lines.append(line)

        ax_u = axes[self.n]
        self._action_lines = []
        for j in range(self.m):
            (line,) = ax_u.step([], [], where="post", linewidth=2, label=f"$u_{{{j}}}$")
            self._action_lines.append(line)
        ax_u.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax_u.set_xlim(0, max_steps)
        ax_u.set_xlabel("Time step")
        ax_u.set_ylabel("Action")
        if self.m > 1:
            ax_u.legend(fontsize="small")
        ax_u.grid(True, alpha=0.3)

        self._fig.suptitle("LQR Trajectory", fontsize=13)
        self._fig.tight_layout()

    def _render_frame(self) -> None:
        """Update the Matplotlib figure with the current trajectory."""
        if len(self.trajectory) == 0:
            return

        traj = np.array(self.trajectory)
        steps = np.arange(len(traj))

        for i, line in enumerate(self._state_lines):
            line.set_data(steps, traj[:, i])
            ax = self._ax[i]  # type: ignore[index]
            y_abs_max = max(np.abs(traj[:, i]).max(), 0.1)
            ax.set_ylim(-y_abs_max * 1.2, y_abs_max * 1.2)

        if len(self.actions) > 0:
            acts = np.array(self.actions)
            act_steps = np.arange(len(acts))
            for j, line in enumerate(self._action_lines):
                line.set_data(act_steps, acts[:, j])
            ax_u = self._ax[self.n]  # type: ignore[index]
            y_abs_max = max(np.abs(acts).max(), 0.1)
            ax_u.set_ylim(-y_abs_max * 1.2, y_abs_max * 1.2)
