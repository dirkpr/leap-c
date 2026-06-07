"""Resolve an i4b reward scenario (R0..R3) into a dataset-normalized RewardConfig.

Centralizes the ``showcase_rewards.py`` pattern (probe env -> ``compute_refs`` ->
``make_scenarios``) so ``run_sac_fop`` / ``run_sac_zop`` / ``run_baseline`` select a
reward identically, keeping the env reward and the planner OCP cost aligned.
"""

from leap_c.examples.i4b.env import BUILDING_NAMES2CLASS, Heatpump_AW, I4bEnv, I4bEnvConfig
from leap_c.examples.i4b.reward import RewardConfig, compute_refs, make_scenarios

REWARD_NAMES = ("R0", "R1", "R2", "R3")


def resolve_reward(reward_name: str) -> RewardConfig:
    """Return the dataset-normalized ``RewardConfig`` for a scenario name (R0..R3).

    Builds a probe ``I4bEnv`` to read the dataset, derives the price/energy
    normalizers via ``compute_refs``, then returns the requested scenario from
    ``make_scenarios``.  For R0/R1/R2 the references are unused, but probing keeps
    the path uniform and matches ``showcase_rewards.py``.
    """
    if reward_name not in REWARD_NAMES:
        raise ValueError(f"Unknown reward '{reward_name}'; choose from {REWARD_NAMES}.")
    hp_model = Heatpump_AW(mdot_HP=0.25)
    probe = I4bEnv(cfg=I4bEnvConfig(building_params=BUILDING_NAMES2CLASS["i4c"], hp_model=hp_model))
    pi_ref, E_ref = compute_refs(probe.dataset, hp_model, float(probe.cfg.delta_t))
    return make_scenarios(pi_ref=pi_ref, E_ref=E_ref)[reward_name]
