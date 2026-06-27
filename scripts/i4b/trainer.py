"""Custom SAC-FOP / SAC-ZOP trainers for i4b with per-episode validation logging.

The per-episode validation logging is algorithm-agnostic (both SAC-FOP and
SAC-ZOP drive the same ``HierachicalMPCActor`` and predict the same per-stage
``Qdot_gains``), so it lives in ``I4bValLoggingMixin`` and is mixed into both
``I4bSacFopTrainer`` and ``I4bSacZopTrainer``. This mirrors the cartpole
convention (``CartPoleValChannelsMixin`` in
``scripts/cartpole/sac_cartpole_mixin.py``).
"""

from __future__ import annotations

from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from leap_c.controller import CtxType
from leap_c.torch.rl.mpc_actor import (
    StochasticMPCActorOutput,
)
from leap_c.torch.rl.sac_fop import SacFopTrainer
from leap_c.torch.rl.sac_zop import SacZopTrainer


class I4bValLoggingMixin:
    """Per-episode validation logging for the i4b environment.

    Meant to be mixed in *before* a concrete SAC trainer, e.g.
    ``class I4bSacFopTrainer(I4bValLoggingMixin, SacFopTrainer): ...``.

    Adds three capabilities on top of the concrete trainer:

    1. ``act()`` stashes ``pi_output.param`` so the per-step callback can access
       the predicted ``Qdot_gains`` without requiring changes to the trainer
       signature.

    2. ``_make_val_step_callback()`` collects per-step records (true and
       predicted Qdot_gains, temperature, setpoints, solver status) and detects
       episode boundaries by watching the step counter reset to 1.

    3. ``validate()`` calls ``_on_episode_end()`` once for each completed
       episode (including the last one, which the step-boundary detector cannot
       see) and then ``_on_validation_end()`` once after all episodes.

    State (``_last_act_param``, ``_pending_episode_flush``) is initialised
    lazily by the methods below, so no ``__init__`` override is needed.
    """

    # ------------------------------------------------------------------
    # act() — stash predicted param for the callback
    # ------------------------------------------------------------------

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        state: CtxType | None = None,
    ) -> tuple[np.ndarray, CtxType | None, dict[str, float] | None]:
        obs = self.buffer.collate([obs])
        t0 = default_timer()
        with torch.inference_mode():
            pi_output: StochasticMPCActorOutput = self.pi(obs, state, deterministic)
        # Wall-clock of the full policy evaluation (MLP + extra parameter setting +
        # solve); stashed on ctx so the per-step recorder logs it raw.
        if pi_output.ctx is not None:
            pi_output.ctx.policy_time_s = default_timer() - t0
        action = pi_output.action.cpu().numpy()[0]  # type: ignore[union-attr]
        self._last_act_param = pi_output.param.cpu().numpy()[0]
        return action, pi_output.ctx, pi_output.stats

    # ------------------------------------------------------------------
    # per-step callback — detects episode boundaries
    # ------------------------------------------------------------------

    def _make_val_step_callback(self):
        episode_records: list[dict] = []
        prev_step_ref = [0]
        episode_ref = [0]
        # Raw per-step records across all validation episodes, persisted by
        # validate() for offline / dashboard analysis (aggregation happens in
        # scripts/i4b/analyze_timings.py, not here).
        self._all_records: list[dict] = []

        def _flush():
            if episode_records:
                self._on_episode_end(list(episode_records))
                episode_records.clear()

        # Expose flush so validate() can call it for the last episode.
        self._pending_episode_flush = _flush

        def callback(step: int, obs, action, reward, info, ctx) -> None:
            # step resets to 1 at the start of every new episode.
            if step == 1 and prev_step_ref[0] > 0:
                _flush()
                episode_ref[0] += 1
            prev_step_ref[0] = step

            last_act_param = getattr(self, "_last_act_param", None)
            Qdot_gains_pred = (
                float(last_act_param[0]) if last_act_param is not None else float("nan")
            )
            Qdot_gains_true = float(info.get("Qdot_gains", float("nan")))
            T_set_lower = float(obs["setpoints"]["T_set_lower"].flat[0])
            T_set_upper = float(obs["setpoints"]["T_set_upper"].flat[0])
            T_room = float(info.get("T_room", float("nan")))

            # Raw acados solver statistics for this solve (stored verbatim).
            solver_stats: dict[str, float] = {}
            if ctx is not None and getattr(ctx, "stats", None):
                s = ctx.stats[0] if isinstance(ctx.stats, list) else ctx.stats
                for k in ("time_tot", "time_lin", "time_qp", "sqp_iter"):
                    v = s.get(k)
                    if v is not None and np.isscalar(v):
                        solver_stats[f"solver.{k}"] = float(v)

            record = {
                "step": step,
                # compute time of the full policy evaluation (overall cost)
                "policy_time_s": float(getattr(ctx, "policy_time_s", float("nan"))),
                # applied action (normalised T_HP)
                "action": float(np.asarray(action).flat[0]) if action is not None else float("nan"),
                # learnable parameter
                "Qdot_gains_pred": Qdot_gains_pred,
                "Qdot_gains_true": Qdot_gains_true,
                # context for the plot
                "T_amb": float(obs["disturbances"]["T_amb"].flat[0]),
                "quarter_hour": int(obs["forecast"]["quarter_hour"].flat[0]),
                "day_of_year": int(obs["forecast"]["day_of_year"].flat[0]),
                "day_of_week": int(obs["forecast"]["day_of_week"].flat[0]),
                # thermal comfort
                "T_set_lower": T_set_lower,
                "T_set_upper": T_set_upper,
                "T_room": T_room,
                "T_set_violated": int(T_room < T_set_lower or T_room > T_set_upper),
                # economics / reliability
                "E_el_kWh": float(info.get("E_el_kWh", float("nan"))),
                "reward": float(reward),
                "solver_status": (
                    int(ctx.status.flat[0]) if ctx is not None and hasattr(ctx, "status") else -1
                ),
                **solver_stats,
            }
            episode_records.append(record)
            self._all_records.append({"episode": episode_ref[0], **record})

        return callback

    # ------------------------------------------------------------------
    # episode-level hook — override to add custom logging/plotting
    # ------------------------------------------------------------------

    def _on_episode_end(self, records: list[dict]) -> None:
        """Called once per completed validation episode.

        ``records`` is a list of per-step dicts (one entry per env step).
        Override or extend this method to add custom metrics or plots.

        The default implementation:
        * logs aggregate scalars via ``report_stats`` (→ WandB / TensorBoard / CSV)
        * sends a matplotlib figure to WandB if a run is active
        """
        if not records:
            return

        df = pd.DataFrame(records)

        # --- scalar metrics -------------------------------------------------
        mae = float((df["Qdot_gains_true"] - df["Qdot_gains_pred"]).abs().mean())
        bias = float((df["Qdot_gains_pred"] - df["Qdot_gains_true"]).mean())
        n_violations = int(df["T_set_violated"].sum())
        solver_failures = int((df["solver_status"] != 0).sum())

        self.report_stats(
            "val_i4b",
            {
                "Qdot_gains_mae": mae,
                "Qdot_gains_bias": bias,
                "T_set_violations": n_violations,
                "solver_failures": solver_failures,
            },
            with_smoothing=False,
        )

        # --- wandb figure ---------------------------------------------------
        try:
            import wandb

            if wandb.run is not None:
                fig = plot_qdot_gains_episode(df)
                wandb.log(
                    {"val_i4b/qdot_gains": wandb.Image(fig)},
                    step=self.state.step,
                )
                plt.close(fig)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # validation-level hook — aggregate across all episodes
    # ------------------------------------------------------------------

    def _on_validation_end(self) -> None:
        """Called once after all validation episodes have completed.

        Override to add cross-episode aggregation.  The default is a no-op.
        """

    # ------------------------------------------------------------------
    # validate() override — wire episode / validation end hooks
    # ------------------------------------------------------------------

    def validate(self) -> float:
        self._pending_episode_flush = None
        self._all_records = []
        score = super().validate()
        # Flush the last episode (the step-boundary detector inside the callback
        # cannot see it because no subsequent episode starts after it).
        if self._pending_episode_flush is not None:
            self._pending_episode_flush()
        self._on_validation_end()
        self._save_val_records()
        return score

    def _save_val_records(self) -> None:
        """Persist the raw per-step validation records as a flat parquet table.

        One row per env step across all validation episodes (tagged with an
        ``episode`` index). Aggregation is left to post-processing
        (scripts/i4b/analyze_timings.py).
        """
        records = getattr(self, "_all_records", None)
        if not records:
            return
        path = self.output_path / f"val_records_step{self.state.step}.parquet"
        pd.DataFrame(records).to_parquet(path)
        print(f"Per-step table saved to: {path}")


class I4bSacFopTrainer(I4bValLoggingMixin, SacFopTrainer):
    """SacFopTrainer with i4b per-episode validation logging."""


class I4bSacZopTrainer(I4bValLoggingMixin, SacZopTrainer):
    """SacZopTrainer with i4b per-episode validation logging."""


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_qdot_gains_episode(df: pd.DataFrame) -> plt.Figure:
    """Return a two-panel matplotlib figure for a single validation episode.

    Top panel  : predicted vs true Qdot_gains [W], with T_amb on a secondary y-axis.
    Bottom panel: T_room vs comfort-band [T_set_lower, T_set_upper].

    Args:
        df: DataFrame with one row per env step, as produced by the step callback.
            Required columns: step, Qdot_gains_pred, Qdot_gains_true, T_amb,
            T_room, T_set_lower, T_set_upper.

    Returns:
        A ``matplotlib.figure.Figure``.  The caller is responsible for closing it
        (``plt.close(fig)``).
    """
    steps = df["step"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # --- top: Qdot_gains ---------------------------------------------------
    ax1 = axes[0]
    ax1.plot(steps, df["Qdot_gains_true"], label="Qdot_gains true", color="steelblue", lw=1.5)
    ax1.plot(
        steps,
        df["Qdot_gains_pred"],
        label="Qdot_gains pred",
        color="tomato",
        lw=1.5,
        ls="--",
    )
    ax1.set_ylabel("Heat gains [W]")
    ax1.legend(loc="upper left", fontsize=8)

    ax1r = ax1.twinx()
    ax1r.plot(steps, df["T_amb"], color="grey", lw=1, alpha=0.5, label="T_amb")
    ax1r.set_ylabel("T_amb [degC]", color="grey")
    ax1r.tick_params(axis="y", labelcolor="grey")

    mae = (df["Qdot_gains_true"] - df["Qdot_gains_pred"]).abs().mean()
    ax1.set_title(f"Qdot_gains  (MAE = {mae:.1f} W)")

    # --- bottom: thermal comfort -------------------------------------------
    ax2 = axes[1]
    ax2.plot(steps, df["T_room"], label="T_room", color="steelblue", lw=1.5)
    ax2.fill_between(
        steps,
        df["T_set_lower"],
        df["T_set_upper"],
        alpha=0.15,
        color="green",
        label="comfort band",
    )
    ax2.plot(steps, df["T_set_lower"], color="green", lw=0.8, ls=":")
    ax2.plot(steps, df["T_set_upper"], color="green", lw=0.8, ls=":")

    n_viol = int(df["T_set_violated"].sum())
    ax2.set_title(f"Thermal comfort  ({n_viol} violations)")
    ax2.set_ylabel("Temperature [degC]")
    ax2.set_xlabel("Step")
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig
