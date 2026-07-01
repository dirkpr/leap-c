"""Minimal closed-loop rollout of the BOPTEST heat-pump MPC (no learning).

This confirms the end-to-end signal wiring: the ``BoptestPlanner`` reads the
zone temperature and the perfect forecasts (ambient temperature, solar, price,
comfort bounds) exposed by ``BoptestEnv``, solves the R1C1 OCP, and applies the
first heat-pump modulation back to BOPTEST. At the end it prints the BOPTEST
core KPIs and saves a plot of the zone temperature vs. the comfort band and the
applied action.

Requires a running BOPTEST server. For a local BOPTEST v0.8.0 deployment, run in
the ``external/project1-boptest`` submodule::

    docker compose up web worker provision

and then (default ``--url http://127.0.0.1``)::

    python scripts/boptest/run_baseline.py
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch

from leap_c.examples.boptest.env import BoptestEnv, BoptestEnvConfig
from leap_c.examples.boptest.planner import BoptestPlanner, BoptestPlannerConfig
from leap_c.planner import ControllerFromPlanner

KELVIN = 273.15


def _to_batched(obs: dict) -> dict:
    """Turn a single env observation into a batch-of-1 of float64 tensors."""
    state = torch.as_tensor(obs["state"], dtype=torch.float64).unsqueeze(0)
    forecast = {
        name: torch.as_tensor(values, dtype=torch.float64).unsqueeze(0)
        for name, values in obs["forecast"].items()
    }
    return {"state": state, "forecast": forecast}


def run(url: str, n_horizon: int, step_period: int, episode_hours: float, output: str) -> None:
    env = BoptestEnv(
        cfg=BoptestEnvConfig(
            url=url,
            N_forecast=n_horizon,
            step_period=step_period,
            max_episode_length=int(episode_hours * 3600),
        )
    )
    planner = BoptestPlanner(
        cfg=BoptestPlannerConfig(N_horizon=n_horizon, step_period=float(step_period))
    )
    controller = ControllerFromPlanner(planner)

    obs, _ = env.reset()
    ctx = None
    times, t_zone, t_low, t_up, actions, rewards = [], [], [], [], [], []

    step = 0
    while True:
        # Record the state and comfort band at the current time (pred_0).
        t_zone.append(float(obs["state"][0]) - KELVIN)
        t_low.append(float(obs["forecast"]["T_lower"][0]) - KELVIN)
        t_up.append(float(obs["forecast"]["T_upper"][0]) - KELVIN)
        times.append(step * step_period / 3600.0)

        ctx, action = controller(_to_batched(obs), ctx=ctx)
        action = action.squeeze(0).detach().cpu().numpy()
        actions.append(float(action[0]))

        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        step += 1
        if terminated or truncated:
            break

    kpis = env.get_kpis()
    print("\nBOPTEST core KPIs:")
    for key in ("cost_tot", "tdis_tot", "ener_tot", "emis_tot", "pele_tot"):
        if key in kpis:
            print(f"  {key:>10}: {kpis[key]:.4f}")
    print(f"  total reward: {np.sum(rewards):.4f}")

    fig, (ax_t, ax_u) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax_t.plot(times, t_zone, label="zone temperature", color="tab:red")
    ax_t.fill_between(times, t_low, t_up, color="tab:green", alpha=0.15, label="comfort band")
    ax_t.set_ylabel("temperature [degC]")
    ax_t.legend(loc="best")
    ax_t.set_title("BOPTEST bestest_hydronic_heat_pump - R1C1 MPC baseline")

    ax_u.step(times, actions, where="post", color="tab:blue")
    ax_u.set_ylabel("heat-pump modulation [-]")
    ax_u.set_xlabel("time [h]")
    ax_u.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(output, dpi=120)
    print(f"\nSaved plot to {output}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1", help="BOPTEST server URL")
    parser.add_argument("--N", type=int, default=12, help="MPC horizon (steps)")
    parser.add_argument("--step", type=int, default=900, help="Control step period [s]")
    parser.add_argument("--episode-hours", type=float, default=24.0, help="Episode length [h]")
    parser.add_argument("--output", default="boptest_baseline.png", help="Output plot path")
    args = parser.parse_args()

    run(args.url, args.N, args.step, args.episode_hours, args.output)
