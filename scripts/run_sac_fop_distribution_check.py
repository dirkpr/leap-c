"""Main script to run experiments."""

from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path

from leap_c.examples import create_controller
from leap_c.run import default_controller_code_path, default_output_path
from leap_c.torch.nn.extractor import ExtractorName
from leap_c.torch.rl.sac_fop import SacFopTrainerConfig


from leap_c.torch.nn.gaussian import SquashedGaussian, BoundedBeta
import torch

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


@dataclass
class RunSacFopConfig:
    """Configuration for running SAC-FOP experiments."""

    env: str = "cartpole"
    controller: str = "cartpole"
    trainer: SacFopTrainerConfig = field(default_factory=SacFopTrainerConfig)
    extractor: ExtractorName = "identity"  # for hvac use "scaling"


def create_cfg() -> RunSacFopConfig:
    # ---- Configuration ----
    cfg = RunSacFopConfig()
    cfg.env = "cartpole"
    cfg.controller = "cartpole"
    cfg.extractor = "identity"  # for hvac use "scaling"

    # ---- Section: cfg.trainer ----
    cfg.trainer.seed = 0
    cfg.trainer.train_steps = 1000000
    cfg.trainer.train_start = 0
    cfg.trainer.val_interval = 10000
    cfg.trainer.val_num_rollouts = 20
    cfg.trainer.val_deterministic = True
    cfg.trainer.val_num_render_rollouts = 0
    cfg.trainer.val_render_mode = "rgb_array"
    cfg.trainer.val_render_deterministic = True
    cfg.trainer.val_report_score = "cum"
    cfg.trainer.ckpt_modus = "best"
    cfg.trainer.batch_size = 64
    cfg.trainer.buffer_size = 1000000
    cfg.trainer.gamma = 0.99
    cfg.trainer.tau = 0.005
    cfg.trainer.soft_update_freq = 1
    cfg.trainer.lr_q = 0.001
    cfg.trainer.lr_pi = 0.001
    cfg.trainer.lr_alpha = 0.001
    cfg.trainer.init_alpha = 0.02
    cfg.trainer.target_entropy = None
    cfg.trainer.entropy_reward_bonus = True
    cfg.trainer.num_critics = 2
    cfg.trainer.report_loss_freq = 100
    cfg.trainer.update_freq = 4
    cfg.trainer.noise = "param"
    cfg.trainer.entropy_correction = False

    # ---- Section: cfg.trainer.log ----
    cfg.trainer.log.verbose = True
    cfg.trainer.log.interval = 1000
    cfg.trainer.log.window = 10000
    cfg.trainer.log.csv_logger = True
    cfg.trainer.log.tensorboard_logger = True
    cfg.trainer.log.wandb_logger = False
    cfg.trainer.log.wandb_init_kwargs = {}

    # ---- Section: cfg.trainer.critic_mlp ----
    cfg.trainer.critic_mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.critic_mlp.activation = "relu"
    cfg.trainer.critic_mlp.weight_init = "orthogonal"

    # ---- Section: cfg.trainer.actor_mlp ----
    cfg.trainer.actor_mlp.hidden_dims = (256, 256, 256)
    cfg.trainer.actor_mlp.activation = "relu"
    # cfg.trainer.actor_mlp.weight_init = "kaiming_uniform"
    cfg.trainer.actor_mlp.weight_init = "orthogonal"

    return cfg


def create_parameter_samples_at_nominal_param(
    cfg: RunSacFopConfig,
    output_path: str | Path,
    device: str = "cuda",
    reuse_code_dir: Path | None = None,
) -> float:
    controller = create_controller(cfg.controller, reuse_code_dir)
    squashed_gaussian = SquashedGaussian(controller.param_space)

    param_manager = controller.param_manager

    p_nom = torch.tensor(param_manager.p_global_values.cat.full().flatten())
    p_min = torch.tensor(param_manager.lb)
    p_max = torch.tensor(param_manager.ub)

    bounded_beta = BoundedBeta(mode=p_nom, lower_bound=p_min, upper_bound=p_max)

    # Test forward pass for nominal parameters
    n_param = controller.param_space.shape[0]
    mean = torch.tensor(n_param * [0.0], dtype=torch.float32).reshape(1, -1)
    log_std = torch.tensor(n_param * [0.0], dtype=torch.float32).reshape(1, -1)

    squashed_gaussian_samples = []
    beta_samples = []
    for _ in range(10000):
        transformed_x, _, _ = squashed_gaussian.forward(mean=mean, log_std=log_std)
        beta_samples.append(bounded_beta.forward(mode=mean, log_c=log_std))
        squashed_gaussian_samples.append(transformed_x)

    squashed_gaussian_samples = torch.stack(squashed_gaussian_samples)
    beta_samples = torch.stack(beta_samples)

    # Squeeze
    squashed_gaussian_samples = squashed_gaussian_samples.squeeze(1)
    beta_samples = beta_samples.squeeze(1)

    plt.hist(
        squashed_gaussian_samples[:, 0].numpy(),
        bins=50,
        density=True,
        alpha=0.7,
        label="Squashed Gaussian",
        color="skyblue",
        edgecolor="black",
    )
    plt.hist(
        beta_samples[:, 0].numpy(),
        bins=50,
        density=True,
        alpha=0.7,
        label="Bounded Beta",
        color="lightgreen",
        edgecolor="black",
    )
    plt.title("Histogram of parameter samples")
    plt.xlabel("value")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="cartpole")
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument(
        "-r",
        "--reuse_code",
        action="store_true",
        help="Reuse compiled code. The first time this is run, it will compile the code.",
    )
    parser.add_argument("--reuse_code_dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = create_cfg()
    cfg.controller = args.controller if args.controller else args.env
    cfg.env = args.env
    cfg.trainer.seed = args.seed

    if args.output_path is None:
        output_path = default_output_path(
            seed=args.seed, tags=["sac_fop", args.env, args.controller]
        )
    else:
        output_path = args.output_path

    if args.reuse_code and args.reuse_code_dir is None:
        reuse_code_dir = default_controller_code_path() if args.reuse_code else None
    elif args.reuse_code_dir is not None:
        reuse_code_dir = args.reuse_code_dir
    else:
        reuse_code_dir = None

    if args.output_path is None:
        trainer_output_path = default_output_path(
            seed=args.seed, tags=["sac_fop", args.env, args.controller]
        )
    else:
        trainer_output_path = args.output_path

    if args.reuse_code and args.reuse_code_dir is None:
        reuse_code_dir = default_controller_code_path() if args.reuse_code else None
    elif args.reuse_code_dir is not None:
        reuse_code_dir = args.reuse_code_dir
    else:
        reuse_code_dir = None

    create_parameter_samples_at_nominal_param(
        cfg=cfg,
        output_path=trainer_output_path,
        device=args.device,
        reuse_code_dir=reuse_code_dir,
    )
