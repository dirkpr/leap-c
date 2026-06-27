"""Showcase the configurable, OCP-aligned i4b reward scenarios on an MPC baseline.

For each reward scenario in ``REWARD_SCENARIOS`` (R0 energy, R1 cost, R2 comfort,
R3 combined) this runs a closed-loop MPC rollout (``I4bPlanner`` + ``I4bEnv``) in
which the env reward and the planner OCP cost are driven by the *same*
``RewardConfig`` -- so reward and MPC stay aligned (see
``i4b/gym_interface/reward.py`` and the wiki note ``i4b-reward-design``).

It illustrates, per scenario, the per-term reward decomposition and the total reward
alongside the state/action trajectories, writes a combined cross-scenario comparison
figure, and prints + saves a summary table.

Run from the repo root:
    python scripts/i4b/showcase_rewards.py
    python scripts/i4b/showcase_rewards.py --days 3 --n-horizon 24
"""

import csv
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from i4b.gym_interface import BUILDING_NAMES2CLASS
from i4b.gym_interface.env import I4bEnv, I4bEnvConfig
from i4b.gym_interface.reward import compute_refs, make_scenarios
from i4b.models.model_hvac import Heatpump_AW

from leap_c.examples.i4b.planner import I4bPlanner, I4bPlannerConfig

# ── Reward terms tracked per step (from info["reward_terms"]) ───────────────────
_TERM_KEYS = (
    "reward",
    "r_energy",
    "r_comfort",
    "r_overheat",
    "r_shaping",
    "price",
    "cost_eur",
    "d_under",
    "grid_signal",
)
# Pretty labels/colors for the additive reward components.
_COMPONENTS = {
    "r_energy": ("energy/cost term", "tab:blue"),
    "r_comfort": ("comfort term", "tab:red"),
    "r_overheat": ("over-heating term", "tab:orange"),
    "r_shaping": ("shaping term", "tab:green"),
}


def _obs_to_tensor(obs_np: dict, dtype=torch.float64) -> dict:
    """Recursively convert a numpy Dict obs to a batched (1, ...) torch Dict."""
    return {
        k: _obs_to_tensor(v, dtype)
        if isinstance(v, dict)
        else torch.tensor(v, dtype=dtype).unsqueeze(0)
        for k, v in obs_np.items()
    }


def rollout(env: I4bEnv, planner: I4bPlanner, max_steps: int, seed: int) -> dict:
    """Closed-loop MPC rollout; returns a dict of per-step numpy arrays."""
    obs_np, _ = env.reset(seed=seed)
    ctx = None
    logs: dict[str, list] = defaultdict(list)

    for step in range(max_steps):
        T_amb_now = float(obs_np["disturbances"]["T_amb"].flat[0])
        obs_t = _obs_to_tensor(obs_np)

        with torch.no_grad():
            ctx, u0_norm, _, _, _ = planner(obs_t, ctx=ctx)

        action_np = u0_norm.squeeze(0).cpu().numpy()
        obs_np, reward, terminated, truncated, info = env.step(action_np)
        terms = info["reward_terms"]

        logs["T_amb"].append(T_amb_now)
        logs["T_room"].append(info["T_room"])
        logs["T_hp_sup"].append(info["T_hp_sup"])
        logs["E_el_kWh"].append(info["E_el_kWh"])
        logs["T_set_lower"].append(float(obs_np["setpoints"]["T_set_lower"].flat[0]))
        logs["T_set_upper"].append(float(obs_np["setpoints"]["T_set_upper"].flat[0]))
        for k in _TERM_KEYS:
            logs[k].append(terms[k])

        if step % 16 == 0:
            print(
                f"    step {step:4d} | T_room={info['T_room']:.1f}degC"
                f" T_HP={info['T_hp_sup']:.1f}degC E={info['E_el_kWh'] * 1e3:.0f}Wh"
                f" r={reward:+.4f}"
            )

        if terminated or truncated:
            break

    return {k: np.asarray(v, dtype=float) for k, v in logs.items()}


def _scenario_title(name: str, rcfg) -> str:
    return (
        f"{name}: lam={rcfg.lam:g} w_c={rcfg.w_c:g} "
        f"pi_ref={rcfg.pi_ref:.3g} E_ref={rcfg.E_ref:.3g} phi={rcfg.penalty_shape}"
    )


def plot_scenario(name: str, rcfg, logs: dict, dt_h: float, out_path: Path) -> None:
    """3-panel figure: temps+comfort band, action/energy, reward terms+total."""
    t = np.arange(len(logs["T_room"])) * dt_h
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # Panel 0 — temperatures + comfort band + ambient.
    ax = axes[0]
    ax.fill_between(
        t, logs["T_set_lower"], logs["T_set_upper"], alpha=0.10, color="green", label="comfort band"
    )
    ax.step(t, logs["T_set_lower"], where="post", color="green", lw=0.8, ls="--")
    ax.step(t, logs["T_set_upper"], where="post", color="green", lw=0.8, ls="--")
    ax.plot(t, logs["T_room"], color="tab:red", lw=1.4, label="T_room")
    ax.plot(t, logs["T_amb"], color="tab:gray", lw=0.9, label="T_amb")
    ax.set_ylabel("Temperature [degC]")
    ax.set_title(_scenario_title(name, rcfg))
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 1 — action (T_HP supply) + electrical energy on a twin axis.
    ax = axes[1]
    ax.step(t, logs["T_hp_sup"], where="post", color="tab:purple", lw=1.2, label="T_HP supply")
    ax.set_ylabel("T_HP [degC]", color="tab:purple")
    ax.tick_params(axis="y", labelcolor="tab:purple")
    ax.grid(True, alpha=0.3)
    axr = ax.twinx()
    axr.step(t, logs["E_el_kWh"], where="post", color="tab:brown", lw=0.9, alpha=0.7)
    axr.set_ylabel("E_el [kWh/step]", color="tab:brown")
    axr.tick_params(axis="y", labelcolor="tab:brown")

    # Panel 2 — reward components + total, with price on a twin axis.
    ax = axes[2]
    for key, (label, color) in _COMPONENTS.items():
        arr = logs[key]
        if np.any(arr != 0.0):
            ax.plot(t, arr, color=color, lw=1.0, label=label)
    ax.plot(t, logs["reward"], color="black", lw=1.6, label="total reward")
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_ylabel("reward / step")
    ax.set_xlabel("time [h]")
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    ax.grid(True, alpha=0.3)
    axr = ax.twinx()
    axr.plot(t, logs["price"], color="tab:cyan", lw=0.9, alpha=0.8)
    axr.set_ylabel("price [EUR/kWh]", color="tab:cyan")
    axr.tick_params(axis="y", labelcolor="tab:cyan")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_comparison(results: dict, dt_h: float, out_path: Path) -> None:
    """Combined figure: cumulative reward and T_room across all scenarios."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    for name, logs in results.items():
        t = np.arange(len(logs["reward"])) * dt_h
        ax.plot(t, np.cumsum(logs["reward"]), lw=1.3, label=name)
    ax.set_ylabel("cumulative reward")
    ax.set_title("Reward scenarios — cumulative reward (not comparable across scenarios)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    any_logs = next(iter(results.values()))
    t = np.arange(len(any_logs["T_room"])) * dt_h
    ax.fill_between(
        t,
        any_logs["T_set_lower"],
        any_logs["T_set_upper"],
        alpha=0.10,
        color="green",
        label="comfort band",
    )
    for name, logs in results.items():
        ax.plot(t, logs["T_room"], lw=1.1, label=f"T_room {name}")
    ax.set_ylabel("T_room [degC]")
    ax.set_xlabel("time [h]")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


def summarize(results: dict, dt_h: float) -> list[dict]:
    """Per-scenario totals: reward, energy [kWh], cost [EUR], comfort [K*h]."""
    rows = []
    for name, logs in results.items():
        rows.append(
            {
                "scenario": name,
                "total_reward": float(logs["reward"].sum()),
                "total_energy_kWh": float(logs["E_el_kWh"].sum()),
                "total_cost_eur": float(logs["cost_eur"].sum()),
                "total_comfort_Kh": float(logs["d_under"].sum() * dt_h),
            }
        )
    return rows


def main() -> None:
    parser = ArgumentParser(
        description="Showcase i4b reward scenarios on an MPC baseline.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--building", type=str, default="i4c")
    parser.add_argument("--method", type=str, default="4R3C")
    parser.add_argument("--mdot-hp", type=float, default=0.25)
    parser.add_argument("--delta-t", type=int, default=900)
    parser.add_argument("--n-horizon", type=int, default=24, help="MPC horizon (steps).")
    parser.add_argument("--days", type=int, default=3, help="Rollout length in days.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-date", type=str, default="2025-01-06")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/i4b_rewards"))
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dt_h = args.delta_t / 3600.0
    steps_per_day = 24 * int(3600 / args.delta_t)
    max_steps = args.days * steps_per_day

    hp_model = Heatpump_AW(mdot_HP=args.mdot_hp)
    building_params = BUILDING_NAMES2CLASS[args.building]

    def make_env(rcfg, dataset=None) -> I4bEnv:
        cfg = I4bEnvConfig(
            building_params=building_params,
            hp_model=hp_model,
            method=args.method,
            mdot_hp=args.mdot_hp,
            delta_t=args.delta_t,
            days=args.days,
            N_forecast=args.n_horizon + 1,  # full horizon coverage for the planner
            start_date=args.start_date,
            seed=args.seed,  # deterministic internal-gain draw -> identical disturbances
            reward=rcfg,
        )
        return I4bEnv(cfg=cfg, dataset=dataset)

    # Probe env to compute hvac-style normalization references from the dataset.
    probe = make_env(make_scenarios()["R0"])
    pi_ref, E_ref = compute_refs(probe.dataset, hp_model, float(args.delta_t))
    print(f"References: pi_ref={pi_ref:.4f} EUR/kWh  E_ref={E_ref:.4f} kWh")
    scenarios = make_scenarios(pi_ref=pi_ref, E_ref=E_ref)
    shared_dataset = probe.dataset  # reuse the loaded+augmented dataset across scenarios

    results: dict[str, dict] = {}
    for name, rcfg in scenarios.items():
        print(f"\n=== Scenario {name} | ws(auto)-> see planner | {_scenario_title(name, rcfg)} ===")
        env = make_env(rcfg, dataset=shared_dataset)
        planner_cfg = I4bPlannerConfig(
            building_params=building_params,
            method=args.method,
            mdot_hp=args.mdot_hp,
            N_horizon=args.n_horizon,
            delta_t=float(args.delta_t),
            reward=rcfg,  # ws auto-derived from this reward config
        )
        planner = I4bPlanner(
            building_model=env.bldg_model,
            hp_model=hp_model,
            cfg=planner_cfg,
            export_directory=out_dir / "acados" / name,
        )
        print(f"  derived OCP ws = {planner.cfg.ws:.4g}")
        logs = rollout(env, planner, max_steps, args.seed)
        results[name] = logs
        plot_scenario(name, rcfg, logs, dt_h, out_dir / f"{name}.png")

    # Combined comparison + summary.
    plot_comparison(results, dt_h, out_dir / "compare.png")

    rows = summarize(results, dt_h)
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 78)
    print(
        f"{'scenario':>10} | {'tot_reward':>12} | {'energy[kWh]':>12} | "
        f"{'cost[EUR]':>10} | {'comfort[K*h]':>12}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r['scenario']:>10} | {r['total_reward']:>12.4f} | "
            f"{r['total_energy_kWh']:>12.3f} | {r['total_cost_eur']:>10.3f} | "
            f"{r['total_comfort_Kh']:>12.3f}"
        )
    print("=" * 78)
    print(f"\nSummary CSV: {csv_path}")
    print("NB: totals are not directly comparable across scenarios (different reward units).")


if __name__ == "__main__":
    main()
