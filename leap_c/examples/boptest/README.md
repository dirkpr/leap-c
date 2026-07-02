# BOPTEST example (minimal, no learning)

A minimal MPC example wiring leap-c to [BOPTEST](https://github.com/ibpsa/project1-boptest)
(a high-fidelity building-emulation benchmark) through its gymnasium interface
[boptest-gym](https://github.com/ibpsa/project1-boptest-gym). The goal at this
stage is to **confirm signal wiring end to end**, not to control well or learn.

Test case: **`bestest_hydronic_heat_pump`** — a single-zone BESTEST case-900
building with an air-to-water heat pump. It is used because its control input is
a single **continuous modulation signal** `oveHeaPumY_u ∈ [0, 1]`, which maps
cleanly onto one MPC action. (The physically simpler `bestest_hydronic` is
steered indirectly through setpoint overwrites that feed internal PI loops, so
it is a poorer fit for a first MPC example.)

## Files

| File | Role |
|------|------|
| `env.py` | `BoptestEnv`: thin adapter reusing `BoptestGymEnv`; exposes a leap-c `Dict` observation `{state, forecast:{...}}` and a `Box(0,1)` action. The heavy `BoptestGymEnv` (needs a server) is created lazily on the first `reset()`. |
| `acados_ocp.py` | `export_parametric_ocp`: R1C1 zone model + electricity-cost / soft comfort-band cost. |
| `planner.py` | `BoptestPlanner`: differentiable-MPC planner over the OCP. |

Registered as `boptest` in `leap_c/examples/__init__.py` (env + planner).

## Prerequisites

```bash
git submodule update --init external/project1-boptest external/project1-boptest-gym
pip install -e ".[boptest]"     # or: uv sync --extra boptest
```

The `boptest` extra pulls boptest-gym's runtime deps (incl. `stable-baselines3`,
which `boptestGymEnv` imports). boptest-gym is a flat repo (no packaging); the
env adds its submodule path to `sys.path` at import time.

## Running the server (local Docker)

BOPTEST is pinned to **v0.8.0** to match boptest-gym v0.8.0 (their versions must
be kept in lock-step). From the boptest submodule:

```bash
cd external/project1-boptest
docker compose up -d web worker provision
```

The `web` service listens on **port 80**, so the default URL is
`http://127.0.0.1`. On a cold start `web` can exit with `InvalidAccessKeyId`
because it races the one-shot `mc` container that provisions the local minio
(S3) credentials — simply re-run once the `mc` container has exited:

```bash
docker compose up -d web provision
```

Stop everything with `docker compose down` (built images stay cached).

## Running the example

```bash
python scripts/boptest/run_baseline.py --url http://127.0.0.1 --N 12 --episode-hours 4
```

This runs a closed-loop rollout (planner reads state + forecasts, applies the
heat-pump modulation), prints the BOPTEST core KPIs, and saves a plot of the
zone temperature vs. the comfort band and the applied action.

## Offline tests (no server)

The planner / OCP is testable without a server:

```bash
TEST_ENV=boptest uv run --extra test python -m pytest tests/leap_c/examples/test_boptest.py
```

The env round-trip test is skipped unless `BOPTEST_URL` is set. The generic
`tests/leap_c/test_scripts.py` smoke test likewise skips `boptest` unless
`BOPTEST_URL` is set.

## Parameters and forecasts

- **Differentiable (tunable) parameters:** R1C1 gains `R, C, a_sol, heat_gain,
  cop` and the comfort weight `w_comfort`. Registered but left at defaults (no
  learning yet); exposed via `param_space` for later tuning.
- **Forecast / exogenous inputs** (`T_amb`, `solar`, `price`, comfort bounds
  `T_lower`/`T_upper`) come from BOPTEST as perfect forecasts over the horizon
  and are injected as **non-differentiable stagewise** parameters: disturbances
  enter the dynamics, price and comfort bounds enter the cost. The current zone
  temperature is the OCP initial state `x0`.

## Caveats

- The R1C1 values are **rough physics, not identified** from data — sufficient
  to confirm wiring, not for accurate control. System identification of the RC
  model is the natural next step before any learning — see
  [`SYSID_PLAN.md`](SYSID_PLAN.md).
- Comfort bounds are enforced as a **soft cost penalty** (not slacked box
  constraints) so the time-varying bounds stay in the parameter machinery.
