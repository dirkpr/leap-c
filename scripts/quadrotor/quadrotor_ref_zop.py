"""Script to run sac_zop for the quadrotor tasks."""
import datetime
from argparse import ArgumentParser
from pathlib import Path

from leap_c.run import main
from leap_c.rl.sac_fop import SacFopBaseConfig


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    cfg = SacFopBaseConfig()

    # log config
    cfg.val.num_render_rollouts = 1
    cfg.log.wandb_logger = False
    cfg.log.csv_logger = False
    cfg.log.train_interval = 100
    cfg.log.tensorboard_logger = True

    # train parameter
    cfg.val.interval = 25_000
    cfg.train.steps = 1_000_000

    # sac parameter
    cfg.sac.entropy_reward_bonus = False  # type: ignore
    cfg.sac.update_freq = 4
    cfg.sac.batch_size = 64
    cfg.sac.lr_pi = 1e-5
    cfg.sac.lr_q = 3e-5
    cfg.sac.lr_alpha = 1e-3
    cfg.sac.init_alpha = 0.001
    cfg.sac.num_critics = 5
    cfg.sac.num_threads_mpc = 4

    output_path = Path(f"output/quadrotor_ref_hard/zop_{args.seed}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
    main("sac_zop", "quadrotor_ref_hard", cfg, output_path, args.device)
