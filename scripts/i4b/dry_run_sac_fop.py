"""Quick dry run for SAC-FOP on the i4b environment.

Two independent checks, run back to back (shared logic in ``dry_run_common.py``):

  (a) probe_signal_flow() -- one forward pass through the actor, asserting the
      shape/bounds at every hop of the policy:

          obs (Dict)
            -> I4bExtractor          features (1, n_feat)
            -> Mlp                    heads (mean, log_std), each (1, n_param)
            -> SquashedGaussian       param sample (1, n_param) = p_global, the
                                      learnable parameters laid out over the horizon
            -> I4bPlanner / acados    differentiable OCP solve
            -> action                 (1, n_act) in [-1, 1]

      It also prints the concrete per-hop dims, the parameter bounds and the
      parameter-manager split, so the ``p`` (non-learnable, set from
      observations/forecast) vs ``p_global`` (learnable, NN-predicted)
      distinction -- and the exact shapes sketched above (n_feat, n_param,
      n_act) -- is visible at runtime.

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

import sys
from pathlib import Path

# run_sac_fop.py / trainer.py are imported by bare name (matching run_sac_fop.py's
# ``from trainer import ...``); make that work regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dry_run_common import main  # noqa: E402
from run_sac_fop import create_cfg, run  # noqa: E402

if __name__ == "__main__":
    main(create_cfg, run, description="Quick dry run for SAC-FOP on i4b.")
