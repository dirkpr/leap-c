"""Quick dry run for SAC-FOP on the i4b environment.

Two independent checks, run back to back:

  (a) probe_signal_flow() -- one forward pass through the actor, asserting the
      shape/bounds at every hop of the policy:

          obs (Dict)
            -> I4bExtractor          (1, 51)
            -> Mlp                    heads (mean, log_std), each (1, 13)
            -> SquashedGaussian       param sample (1, 13), p_global = per-stage
                                      Qdot_gains in [0, 8000] W
            -> I4bPlanner / acados    differentiable OCP solve
            -> action                 (1, 1) in [-1, 1]

      It also prints the parameter-manager split so the ``p`` (non-learnable,
      set from observations/forecast) vs ``p_global`` (learnable, NN-predicted)
      distinction is visible at runtime.

  (b) run_short_training() -- a ~20-step SAC-FOP loop that exercises
      collect -> buffer -> update (incl. the FOP Jacobian path) -> validate
      (incl. the I4bSacFopTrainer per-episode callback).

The first run compiles the acados solver (~1-2 min). Code reuse is ON by
default (``--reuse-code``); pass ``--no-reuse-code`` to force a fresh compile.

Run from this directory (it shares run_sac_fop.py's ``from trainer import``
convention):

    python scripts/i4b/dry_run_sac_fop.py            # probe + short loop
    python scripts/i4b/dry_run_sac_fop.py --probe-only
    python scripts/i4b/dry_run_sac_fop.py --no-reuse-code
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# run_sac_fop.py / trainer.py are imported by bare name (matching run_sac_fop.py's
# ``from trainer import ...``); make that work regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_sac_fop import create_cfg, run  # noqa: E402

from leap_c.examples import create_controller, create_env  # noqa: E402
from leap_c.run import default_controller_code_path  # noqa: E402
from leap_c.torch.rl.buffer import ReplayBuffer  # noqa: E402
from leap_c.torch.rl.mpc_actor import HierachicalMPCActor  # noqa: E402

_TOL = 1e-3  # float slack for bound checks


def probe_signal_flow(reuse_code_dir: Path | None, seed: int = 0) -> None:
    """Run one actor forward pass and assert the shape/bounds at every hop."""
    print("=" * 72)
    print("SIGNAL-FLOW PROBE: obs -> NN -> distribution -> param (p_global) -> action")
    print("=" * 72)

    cfg = create_cfg(seed=seed)
    env = create_env("i4b")
    controller = create_controller("i4b", reuse_code_dir)

    actor = HierachicalMPCActor(
        cfg.trainer.actor, env.observation_space, env.action_space, controller
    )
    actor.eval()

    # --- parameter-manager split: p (non-learnable) vs p_global (learnable) ---
    param_manager = controller.planner.param_manager
    print("\nparameter-manager split:")
    for p in param_manager.parameters.values():
        lo = np.asarray(p.space.low).ravel() if p.space is not None else None
        hi = np.asarray(p.space.high).ravel() if p.space is not None else None
        bounds = f"[{lo[0]:.0f}, {hi[0]:.0f}]" if lo is not None else "(unbounded)"
        target = "p_global" if p.interface == "learnable" else "p"
        stages = f" x{len(p.end_stages)} stages" if p.end_stages else ""
        print(f"  {p.name:<12} interface={p.interface:<13} -> {target:<8} {bounds}{stages}")

    # --- param space (NN output target) ---
    param_space = controller.param_space
    param_dim = int(np.prod(param_space.shape))
    print(
        f"\ncontroller.param_space: shape={tuple(param_space.shape)} "
        f"low={param_space.low.min():.0f} high={param_space.high.max():.0f}"
    )
    N = controller.planner.cfg.N_horizon
    assert param_dim == N + 1, f"expected param_dim == N+1 == {N + 1}, got {param_dim}"

    # --- collate a single env observation into a batch (as the trainer does) ---
    buffer = ReplayBuffer(1, "cpu", torch.float32, controller.collate_fn_map)
    obs, _ = env.reset(seed=seed)
    obs_b = buffer.collate([obs])

    with torch.no_grad():
        # hop 1: feature extractor
        e = actor.extractor(obs_b)
        # hop 2: MLP -> distribution-parameter heads
        dist_params = actor.mlp(e)
        # hop 3: anchor (residual learning) + squashed-Gaussian sample
        anchor = controller.default_param(obs_b) if actor.residual else None
        param_s, log_prob, _ = actor.param_distribution(
            *dist_params, deterministic=False, anchor=anchor
        )
        # hop 4+5: authoritative end-to-end pass (param -> OCP solve -> action)
        out = actor(obs_b, deterministic=False)

    extractor_out = int(actor.extractor.output_size)

    print("\nper-hop shapes / bounds:")
    print(f"  extractor out    : {tuple(e.shape)}        (expected (1, {extractor_out}))")
    print(
        f"  mlp heads        : {len(dist_params)} x {tuple(dist_params[0].shape)} "
        f"(mean, log_std)  (expected 2 x (1, {param_dim}))"
    )
    print(f"  residual anchor  : {None if anchor is None else tuple(anchor.shape)}")
    print(
        f"  sampled param    : {tuple(param_s.shape)}  range "
        f"[{param_s.min():.1f}, {param_s.max():.1f}]  (p_global = Qdot_gains)"
    )
    print(
        f"  action           : {tuple(out.action.shape)}  range "
        f"[{out.action.min():.3f}, {out.action.max():.3f}]"
    )
    print(f"  solver status    : {np.asarray(out.status).ravel().tolist()}  (0 = success)")

    # --- assertions ---
    assert tuple(e.shape) == (1, extractor_out), tuple(e.shape)
    assert len(dist_params) == 2, len(dist_params)
    assert all(tuple(d.shape) == (1, param_dim) for d in dist_params), [
        d.shape for d in dist_params
    ]
    assert tuple(param_s.shape) == (1, param_dim), tuple(param_s.shape)
    assert (param_s >= param_space.low.min() - _TOL).all() and (
        param_s <= param_space.high.max() + _TOL
    ).all(), "sampled param out of parameter-space bounds"
    assert tuple(out.action.shape) == (1, 1), tuple(out.action.shape)
    assert (out.action >= -1 - _TOL).all() and (out.action <= 1 + _TOL).all(), (
        "action out of [-1, 1]"
    )
    assert (np.asarray(out.status).ravel() == 0).all(), f"solver failed: {out.status}"

    env.close()
    print("\nPROBE OK -- every hop matches the expected shape/bounds.\n")


def run_short_training(
    reuse_code_dir: Path | None, steps: int, output_path: Path, seed: int
) -> float:
    """Run a tiny SAC-FOP training loop to exercise the full pipeline."""
    print("=" * 72)
    print(f"SHORT TRAINING LOOP: {steps} steps (collect -> buffer -> update -> validate)")
    print("=" * 72)

    cfg = create_cfg(seed=seed)
    cfg.trainer.train_steps = steps
    cfg.trainer.train_start = 0
    cfg.trainer.update_freq = 1
    cfg.trainer.batch_size = 4
    cfg.trainer.val_freq = max(1, steps // 2)
    cfg.trainer.val_num_rollouts = 1
    cfg.trainer.val_num_render_rollouts = 0
    cfg.trainer.buffer_size = 1_000
    cfg.trainer.ckpt_modus = "none"
    cfg.trainer.log.csv_logger = False
    cfg.trainer.log.tensorboard_logger = False
    cfg.trainer.log.wandb_logger = False

    score = run(
        cfg,
        output_path=output_path,
        device="cpu",
        reuse_code_dir=reuse_code_dir,
        with_val=True,
    )
    print(f"\nTRAINING LOOP OK -- final validation score: {score}")
    print(f"(output written to {output_path})\n")
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick dry run for SAC-FOP on i4b.")
    parser.add_argument(
        "-r",
        "--reuse-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse compiled acados code (default: on). Use --no-reuse-code to recompile.",
    )
    parser.add_argument("--probe-only", action="store_true", help="Run only the signal-flow probe.")
    parser.add_argument("--steps", type=int, default=20, help="Training steps for the short loop.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    reuse_code_dir = default_controller_code_path() if args.reuse_code else None

    probe_signal_flow(reuse_code_dir, seed=args.seed)

    if args.probe_only:
        return

    output_path = args.output_path or Path(tempfile.mkdtemp(prefix="i4b_dryrun_"))
    run_short_training(reuse_code_dir, args.steps, output_path, seed=args.seed)


if __name__ == "__main__":
    main()
