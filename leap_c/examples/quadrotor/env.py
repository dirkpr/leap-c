from os.path import abspath, dirname

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation as R

from leap_c.examples.quadrotor.casadi_models import (
    get_rhs_quadrotor,
    integrate_one_step,
)
from leap_c.examples.quadrotor.utils import read_from_yaml


W_R_ALIVE = 0.2
W_R_PROGRESS = 1.
W_R_CONSTRAINT = 5.
W_R_CONTROL_REG = 0.
W_R_CONTROL_CONS = 0.
W_R_CLOSE = 1.


class QuadrotorStop(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
            self,
            render_mode: str | None = None,
            verbose: bool = False,
            difficulty: str = "easy",
            obs_with_action: bool = True,
    ):
        self.fig, self.axes = None, None
        self.verbose = verbose
        self.uref = np.array([970.437] * 4)

        self.obs_with_action = obs_with_action

        if difficulty == "easy":
            scale_disturbances = 0.0
        elif difficulty == "medium":
            scale_disturbances = 0.0004
        elif difficulty == "hard":
            scale_disturbances = 0.001
        else:
            raise ValueError("Difficulty must be one of easy, medium, or hard.")

        self.model_params = read_from_yaml(dirname(abspath(__file__)) + "/model_params.yaml")

        self.sim_params = {
            "dt": 0.04,
            "t_sim": 5.0
        }
        x, u, p, rhs, self.rhs_func = get_rhs_quadrotor(self.model_params,
                                                        model_fidelity="high",
                                                        scale_disturbances=scale_disturbances,
                                                        sym_params=False)

        x_high = np.array(
            [
                4.0, 4.0, 4.0,  # position
                1.5, 1.5, 1.5, 1.5,  # quaternion
                50, 50, 50,  # velocity
                50, 50, 50,  # angular velocity
            ],
            dtype=np.float32,
        )
        x_low = np.array(
            [
                -4.0, -4.0, -4.0,  # position
                -1, -1, -1, -1,  # quaternion
                -50, -50, -50,  # velocity
                -50, -50, -50,  # angular velocity
            ],
            dtype=np.float32,
        )
        self.x_low, self.x_high = x_low, x_high

        u_high = np.array([self.model_params["motor_omega_max"]] * 4, dtype=np.float32)
        u_low = np.array([0.0] * 4, dtype=np.float32)

        self.max_u_dist = np.linalg.norm(u_high - u_low)

        self.action_space = spaces.Box(u_low, u_high, dtype=np.float32)
        if obs_with_action:
            obs_high = np.concatenate((x_high, u_high))
            obs_low = np.concatenate((x_low, u_low))
        else:
            obs_high = x_high
            obs_low = x_low
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.reset_needed = True
        self.trajectory, self.time_steps, self.action_trajectory = None, None, None
        self.t = 0
        self.x = None
        self.prev_a = None
        self.last_dist = None

        # For rendering
        if not (render_mode is None or render_mode in self.metadata["render_modes"]):
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute the dynamics of the pendulum on cart."""
        if self.reset_needed:
            raise Exception("Call reset before using the step method.")
        dt = self.sim_params["dt"]

        action = np.clip(action, self.action_space.low, self.action_space.high)  # type: ignore
        self.x = integrate_one_step(self.rhs_func, self.x, action, dt).full().flatten()
        self.x[3:3 + 4] = self.x[3:3 + 4] / np.linalg.norm(self.x[3:3 + 4])
        self.t += dt
        self.trajectory = [self.x] if self.trajectory is None else self.trajectory + [self.x]
        self.time_steps = [self.t] if self.time_steps is None else self.time_steps + [self.t]
        self.action_trajectory = [action] if self.action_trajectory is None else self.action_trajectory + [action]


        term = False
        trunc = False

        if bool(np.isnan(self.x).sum()):  # or (self.x[7:10].sum() <= 1000) and (self.x[7:10].sum() >= -1000):
            self.verbose and (f"Truncation due to nans at time {self.t} with state {self.x}")
            trunc = True
            term = True

        elif (any(self.x > self.x_high) or any(self.x < self.x_low)):
            self.verbose and print(f"Truncation due to state limits at time {self.t} with state {self.x}")
            trunc = True
            term = True

        # rewards similar to
        # https://gymnasium.farama.org/environments/mujoco/ant/
        r_progress = 0
        dist = np.linalg.norm(self.x[:3])
        if self.last_dist is not None:
            r_progress = W_R_PROGRESS * (self.last_dist - dist) / 100
        self.last_dist = np.linalg.norm(self.x[:3])

        r_close = W_R_CLOSE / (np.linalg.norm(self.x[:3]) + 1)

        violates_contraint = min(np.sign(-self.x[2] + self.model_params["safety_dist"]), 0)
        r_constraint = W_R_CONSTRAINT * violates_contraint
        if violates_contraint:
            term = True

        r_control = -W_R_CONTROL_REG * np.linalg.norm(action - self.uref) / self.max_u_dist
        r_alive = W_R_ALIVE
        r_consistent = -W_R_CONTROL_CONS * np.linalg.norm(action - self.prev_a) / self.max_u_dist

        print("r_progress:", r_progress)
        print("r_constraint:", r_constraint)
        print("r_alive:", r_alive)
        print("r_control:", r_control)
        print("r_close:", r_close)
        print("r_consistent:", r_consistent)


        # todo: add cost for control, action, dcontrol, and daction
        r = r_progress + r_constraint + r_alive + r_control + r_close + r_consistent
        r *= 10 * dt

        if self.t >= self.sim_params["t_sim"]:
            term = True

        self.reset_needed = trunc or term
        self.prev_a = action

        return self.obs(), r, term, trunc, {}

    def reset(
            self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:  # type: ignore
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
            self.observation_space.seed(seed)
            self.action_space.seed(seed)
        if self._np_random is None:
            raise RuntimeError("The first reset needs to be called with a seed.")
        self.t = 0
        self.last_dist = None

        Rxy = 2  # Radius of the sphere
        Rz = 4

        # Generate random angles
        phi = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle [0, 2π]
        cos_theta = np.random.uniform(-1, 0)  # Cosine of polar angle [-1,1]
        theta = np.arccos(cos_theta)  # Convert to theta

        # Convert to Cartesian coordinates
        px = Rxy * np.cos(phi) * np.sin(theta)
        py = Rxy * np.sin(phi) * np.sin(theta)
        pz = Rz * np.cos(theta)

        self.x = np.array([px, py, pz,
                           1, 0, 0, 0,
                           0, 0, 0,
                           0, 0, 0], dtype=np.float32)
        self.prev_a = np.zeros(4, dtype=np.float32)
        self.reset_needed = False

        self.trajectory, self.time_steps, self.action_trajectory = [self.x], [self.t], None
        return self.obs(), {}

    def obs(self) -> np.ndarray:
        if self.obs_with_action:
            return np.concatenate((self.x, self.prev_a))  # type: ignore
        else:
            return self.x  # type: ignore

    def render(self):

        if self.render_mode == "human":
            self.fig, self.axes = plt.subplots(4, 3, figsize=(12, 12))
            # Share y-axis only for specific rows
            for col in range(3):  # Iterate through columns
                self.axes[0, col].sharey(self.axes[0, 0])  # Share y-axis between row 0 and row 1
                self.axes[1, col].sharey(self.axes[1, 0])
                self.axes[2, col].sharey(self.axes[2, 0])
            fig, axes = self.fig, self.axes

            trajectory = np.array(self.trajectory)
            action_trajectory = np.array(self.action_trajectory)
            axes[0, 0].plot(self.time_steps, trajectory[:, 0])
            axes[0, 0].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
            axes[0, 0].set_title(r"position $p_x$")
            axes[0, 0].set_xlabel("time (s)")
            axes[0, 0].set_ylabel("position (m)")

            axes[0, 1].plot(self.time_steps, trajectory[:, 1])
            axes[0, 1].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
            axes[0, 1].set_title(r"position $p_y$")
            axes[0, 1].set_xlabel("time (s)")

            axes[0, 2].plot(self.time_steps, trajectory[:, 2])
            axes[0, 2].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
            # axes[0, 2].hlines(self.model_params["bound_z"], 0, self.sim_params["t_sim"], colors='tab:red',
            #                  linestyles='dashed')
            axes[0, 2].set_title(r"position $p_z$")
            axes[0, 2].set_xlabel("time (s)")

            axes[1, 0].plot(self.time_steps, trajectory[:, 7])
            axes[1, 0].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
            axes[1, 0].set_xlabel("time (s)")
            axes[1, 0].set_ylabel(r"velocity ($\frac{m}{s}$)")
            axes[1, 0].set_title(r"velocity $v_x$")

            axes[1, 1].plot(self.time_steps, trajectory[:, 8])
            axes[1, 1].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
            axes[1, 1].set_xlabel("time (s)")
            axes[1, 1].set_title(r"velocity $v_y$")

            axes[1, 2].plot(self.time_steps, trajectory[:, 9])
            axes[1, 2].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
            axes[1, 2].set_xlabel("time (s)")
            axes[1, 2].set_title(r"velocity $v_z$")

            axes[2, 0].plot(self.time_steps, trajectory[:, 10])
            axes[2, 0].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
            axes[2, 0].set_xlabel("time (s)")
            axes[2, 0].set_ylabel(r"angular velocity ($\frac{m}{s}$)")
            axes[2, 0].set_title(r"angular velocity $\omega_x$")

            axes[2, 1].plot(self.time_steps, trajectory[:, 11])
            axes[2, 1].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
            axes[2, 1].set_xlabel("time (s)")
            axes[2, 1].set_title(r"angular velocity $\omega_y$")

            axes[2, 2].plot(self.time_steps, trajectory[:, 12])
            axes[2, 2].hlines(0, 0, self.sim_params["t_sim"], colors='tab:green', linestyles='dashed')
            axes[2, 2].set_xlabel("time (s)")
            axes[2, 2].set_title(r"angular velocity $\omega_z$")

            axes[3, 0].plot(self.time_steps[:-1], action_trajectory[:, 0])
            axes[3, 0].plot(self.time_steps[:-1], action_trajectory[:, 1])
            axes[3, 0].plot(self.time_steps[:-1], action_trajectory[:, 2])
            axes[3, 0].plot(self.time_steps[:-1], action_trajectory[:, 3])
            axes[3, 0].set_xlabel("time (s)")
            axes[3, 0].set_title(r"Revolution speeds")

            axes[3, 1].plot(self.time_steps, trajectory[:, 3], label="qw")
            axes[3, 1].plot(self.time_steps, trajectory[:, 4], label="qx")
            axes[3, 1].legend()
            axes[3, 1].set_xlabel("time (s)")
            axes[3, 1].set_title(r"Quaternions w, x")

            axes[3, 2].plot(self.time_steps, trajectory[:, 5], label="qy")
            axes[3, 2].plot(self.time_steps, trajectory[:, 6], label="qz")
            axes[3, 2].legend()
            axes[3, 2].set_xlabel("time (s)")
            axes[3, 2].set_title(r"Quaternions y, z")

            plt.tight_layout()
            plt.show()
        else:
            image_size = (2 * 512, 2 * 512)
            x, y, z, qw, qx, qy, qz = self.x[0:7]

            # Convert quaternion to rotation matrix
            r = R.from_quat([qx, qy, qz, qw])
            rotation_matrix = r.as_matrix()

            # Define drone body (a simple cube centered at (x, y, z))
            drone_size = 0.5  # Arbitrary size
            size_half = drone_size / 2
            height_hlf = 0.1
            cube_vertices = np.array([
                [-size_half, -size_half, -height_hlf],
                [size_half, -size_half, -height_hlf],
                [size_half, size_half, -height_hlf],
                [-size_half, size_half, -height_hlf],
                [-size_half, -size_half, height_hlf],
                [size_half, -size_half, height_hlf],
                [size_half, size_half, height_hlf],
                [-size_half, size_half, height_hlf]
            ])

            # Rotate and translate the cube
            rotated_vertices = (rotation_matrix @ cube_vertices.T).T + np.array([x, y, z])

            # Define cube faces
            faces = [
                [0, 1, 2, 3], [4, 5, 6, 7],  # Top & Bottom
                [0, 1, 5, 4], [2, 3, 7, 6],  # Sides
                [1, 2, 6, 5], [4, 7, 3, 0]  # More sides
            ]

            fig = plt.figure(figsize=(image_size[0] / 100, image_size[1] / 100))
            ax = fig.add_subplot(111, projection='3d')

            # Draw the cube
            for face in faces:
                ax.add_collection3d(Poly3DCollection([rotated_vertices[face]], color='blue', alpha=1))

            # Draw drone orientation axes
            axis_length = drone_size * 1.5
            origin = np.array([x, y, z])
            axes = np.array([
                [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]
            ])

            transformed_axes = (rotation_matrix @ axes.T).T + origin
            ax.quiver(*origin, *(transformed_axes[0] - origin), color='r', label="X-axis")
            ax.quiver(*origin, *(transformed_axes[1] - origin), color='g', label="Y-axis")
            ax.quiver(*origin, *(transformed_axes[2] - origin), color='b', label="Z-axis")
            ax.scatter(0, 0, 0, color='black', label="Origin")

            # Draw planar surface at z = 0.25
            plane_size = 2.0  # Size of the ground plane
            plane_z = self.model_params["lower_bound_z"] - drone_size / 2  # Fixed height for the plane
            plane_vertices = np.array([
                [-plane_size, -plane_size, plane_z],
                [plane_size, -plane_size, plane_z],
                [plane_size, plane_size, plane_z],
                [-plane_size, plane_size, plane_z]
            ])

            plane_face = [[0, 1, 2, 3]]
            plane = Poly3DCollection([plane_vertices[face] for face in plane_face], color='lightgray', alpha=0.01)
            ax.add_collection3d(plane)

            # Set limits and labels
            ax.view_init(elev=10,
                         azim=45)  # Set elevation to -30° to look from below, azimuth to 45° for an angled view
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')

            ax.set_xlim([-2.5, 2.5])
            ax.set_ylim([-2.5, 2.5])
            ax.set_zlim([-2.5, 2.5])  # Ensure space above the plane
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Drone 3D Visualization")

            # Render to an RGB array
            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

            plt.close(fig)  # Close the figure to avoid memory leaks
            return image


# execute as main to test
if __name__ == "__main__":
    env = QuadrotorStop()
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action = [970.437] * 4
        obs, reward, done, _, _ = env.step(action)
        print(f"reward:{reward}")
    env.render()
