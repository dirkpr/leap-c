"""High-level i4b ablation orchestrator.

Launches the ``{algorithm x reward x seed}`` training grid (plus one MPC baseline
per reward) as isolated subprocesses, with W&B logging and per-run output
directories, and writes a top-level ``manifest.json``.

Each run is a separate process so acados code generation and long RL training
stay isolated -- one crash does not abort the grid.  Subprocesses run with
``cwd`` set to this script's directory so the run scripts' relative imports
(``trainer``, ``reward_setup``, ``channels``) resolve.

W&B project/entity default to the i4b convention (project ``i4b``, entity
``leap-c``) from ``run_i4b_experiment.sh``; group is auto-derived per combo
(``<algo>-<reward>`` / ``baseline-<reward>``).  Override with --wandb-project /
--wandb-entity, or disable logging with --no-use-wandb.

Run from the repo root::

    # 1) dry run: print the full command grid without executing anything
    python scripts/i4b/run_ablation.py --dry-run

    # 2) smoke pass: short runs that validate the whole pipeline end to end
    python scripts/i4b/run_ablation.py --train-steps 2000

    # 3) full study (raise the step budget)
    python scripts/i4b/run_ablation.py --train-steps 200000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from env_setup import add_env_noise_args, env_noise_to_cli, stochasticity_from_args

SCRIPT_DIR = Path(__file__).resolve().parent

# SAC algorithm name -> run script.
ALGOS = {
    "sac_fop": "run_sac_fop.py",
    "sac_zop": "run_sac_zop.py",
}
BASELINE_SCRIPT = "run_baseline.py"
REWARDS = ("R0", "R1", "R2", "R3")  # mirrors reward_setup.REWARD_NAMES


@dataclass
class Run:
    """One leaf of the ablation grid (a single subprocess invocation)."""

    kind: str  # "sac_fop" | "sac_zop" | "baseline"
    reward: str
    seed: int
    output_path: Path
    wandb_group: str
    cmd: list[str]
    returncode: int | None = None
    status: str = "pending"

    @property
    def label(self) -> str:
        return f"{self.kind}/{self.reward}/seed_{self.seed}"

    def to_record(self) -> dict:
        return {
            "label": self.label,
            "kind": self.kind,
            "reward": self.reward,
            "seed": self.seed,
            "output_path": str(self.output_path),
            "wandb_group": self.wandb_group,
            "cmd": self.cmd,
            "returncode": self.returncode,
            "status": self.status,
        }


def _wandb_args(args: argparse.Namespace, group: str) -> list[str]:
    """W&B CLI flags shared by every run script (empty if logging is off)."""
    if not args.use_wandb:
        return []
    flags = [
        "--use-wandb",
        "--wandb-project",
        args.wandb_project,
        "--wandb-group",
        group,
        "--append-start-time",
    ]
    if args.wandb_entity is not None:
        flags += ["--wandb-entity", args.wandb_entity]
    return flags


def build_runs(args: argparse.Namespace, run_dir: Path) -> list[Run]:
    """Expand the algorithm x reward x seed grid (+ per-reward baselines)."""
    runs: list[Run] = []

    # One stochasticity config applied uniformly to every run (SAC + baseline) so the
    # whole grid is trained/evaluated on the same noise/uncertainty level.
    noise_flags = env_noise_to_cli(stochasticity_from_args(args))

    for algo in args.algos:
        script = ALGOS[algo]
        for reward in args.rewards:
            for seed in args.seeds:
                out = run_dir / algo / reward / f"seed_{seed}"
                group = f"{algo}-{reward}"
                cmd = [
                    sys.executable,
                    script,
                    "--seed",
                    str(seed),
                    "--reward",
                    reward,
                    "--device",
                    args.device,
                    "--output-path",
                    str(out),
                ]
                if args.train_steps is not None:
                    cmd += ["--train-steps", str(args.train_steps)]
                if args.with_val:
                    cmd += ["--with-val"]
                if args.reuse_code:
                    # Each reward -> different OCP (different derived ws), and fop/zop
                    # differ in sensitivity codegen: never share generated code.
                    cmd += ["--reuse-code-dir", str(run_dir / "code" / algo / reward)]
                cmd += noise_flags
                cmd += _wandb_args(args, group)
                runs.append(Run(algo, reward, seed, out, group, cmd))

    if args.baseline:
        for reward in args.rewards:
            out = run_dir / "baseline" / reward / f"seed_{args.baseline_seed}"
            group = f"baseline-{reward}"
            cmd = [
                sys.executable,
                BASELINE_SCRIPT,
                "--seed",
                str(args.baseline_seed),
                "--reward",
                reward,
                "--policy-type",
                "controller",
                "--device",
                args.device,
                "--output-path",
                str(out),
            ]
            if args.reuse_code:
                cmd += ["--reuse-code-dir", str(run_dir / "code" / "baseline" / reward)]
            cmd += noise_flags
            cmd += _wandb_args(args, group)
            runs.append(Run("baseline", reward, args.baseline_seed, out, group, cmd))

    return runs


def write_manifest(path: Path, args: argparse.Namespace, run_id: str, runs: list[Run]) -> None:
    """Persist the run table; rewritten after every run so partial progress survives."""
    manifest = {
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "algos": args.algos,
            "rewards": args.rewards,
            "seeds": args.seeds,
            "baseline": args.baseline,
            "train_steps": args.train_steps,
            "with_val": args.with_val,
            "device": args.device,
            "use_wandb": args.use_wandb,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "reuse_code": args.reuse_code,
            "max_parallel": args.max_parallel,
            "stochasticity": asdict(stochasticity_from_args(args)),
        },
        "runs": [r.to_record() for r in runs],
    }
    path.write_text(json.dumps(manifest, indent=2))


def run_one(run: Run, stream: bool) -> Run:
    """Execute a single run; stream output live (sequential) or tee to console.log."""
    run.output_path.mkdir(parents=True, exist_ok=True)
    run.status = "running"
    if stream:
        proc = subprocess.run(run.cmd, cwd=SCRIPT_DIR)
    else:
        log_path = run.output_path / "console.log"
        with log_path.open("w") as f:
            proc = subprocess.run(run.cmd, cwd=SCRIPT_DIR, stdout=f, stderr=subprocess.STDOUT)
    run.returncode = proc.returncode
    run.status = "ok" if proc.returncode == 0 else "failed"
    return run


def print_grid(runs: list[Run]) -> None:
    print("=" * 78)
    print(f"i4b ablation grid: {len(runs)} runs")
    print("-" * 78)
    for i, r in enumerate(runs, 1):
        print(f"  [{i:2d}/{len(runs)}] {r.label:28s} group={r.wandb_group}")
    print("=" * 78)


def print_summary(runs: list[Run]) -> None:
    ok = sum(r.status == "ok" for r in runs)
    print("\n" + "=" * 78)
    print(f"Ablation finished: {ok}/{len(runs)} ok")
    print("-" * 78)
    for r in runs:
        flag = "ok " if r.status == "ok" else f"FAIL({r.returncode})"
        print(f"  {flag:>10}  {r.label:28s} -> {r.output_path}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the i4b algorithm x reward ablation grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        choices=list(ALGOS),
        default=list(ALGOS),
        help="SAC algorithms to include.",
    )
    parser.add_argument(
        "--rewards",
        nargs="+",
        choices=list(REWARDS),
        default=list(REWARDS),
        help="Reward scenarios to include.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds per SAC combination.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=2000,
        help="Training steps per SAC run (default 2000 = smoke; raise for the full study).",
    )
    parser.add_argument(
        "--baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run one MPC baseline per reward as a reference.",
    )
    parser.add_argument(
        "--baseline-seed",
        type=int,
        default=0,
        help="Seed for the MPC baseline runs.",
    )
    parser.add_argument(
        "--with-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable validation rollouts + per-episode logging for SAC runs.",
    )
    parser.add_argument("--device", type=str, default="cpu")

    group = parser.add_argument_group("W&B logging")
    group.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log every run to Weights & Biases.",
    )
    # W&B project/entity follow the i4b convention from run_i4b_experiment.sh.
    group.add_argument("--wandb-project", type=str, default="i4b")
    group.add_argument("--wandb-entity", type=str, default="leap-c")

    group = parser.add_argument_group("Output and execution")
    group.add_argument("--output-root", type=Path, default=Path("output/i4b_ablation"))
    group.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Sub-directory name under --output-root (default: timestamp).",
    )
    group.add_argument(
        "--reuse-code",
        action="store_true",
        help="Reuse compiled acados code (keyed per kind/reward to stay correct).",
    )
    group.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Concurrent runs. 1 = sequential with live output; >1 tees to console.log.",
    )
    group.add_argument("--dry-run", action="store_true", help="Print the grid and exit.")

    add_env_noise_args(parser)

    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (args.output_root / run_id).resolve()

    runs = build_runs(args, run_dir)
    print_grid(runs)

    if args.dry_run:
        print("\n--dry-run: commands that would execute (cwd=%s):\n" % SCRIPT_DIR)
        for r in runs:
            print(f"# {r.label}")
            print("  " + " ".join(r.cmd) + "\n")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    write_manifest(manifest_path, args, run_id, runs)
    print(f"\nManifest: {manifest_path}\n")

    if args.max_parallel <= 1:
        for i, run in enumerate(runs, 1):
            print(f"\n>>> [{i}/{len(runs)}] {run.label}\n")
            run_one(run, stream=True)
            write_manifest(manifest_path, args, run_id, runs)
    else:
        with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
            futures = {ex.submit(run_one, run, False): run for run in runs}
            done = 0
            for fut in as_completed(futures):
                run = fut.result()
                done += 1
                print(f"  [{done}/{len(runs)}] {run.label}: {run.status}")
                write_manifest(manifest_path, args, run_id, runs)

    write_manifest(manifest_path, args, run_id, runs)
    print_summary(runs)


if __name__ == "__main__":
    main()
