"""Serve a COFACTOR apartment-electricity column as an indoor heat disturbance.

``assets/elimp_per_apartment.parquet`` holds hourly, area-normalized
apartment electricity import [W/m^2], one column per building. ``GainSampler``
picks a random building on ``reset()`` and serves that signal as the indoor heat
disturbance ``q_indoor_disturbance`` acting on the indoor temperature node.

The data file is not tracked due to size, contact dirk.p.reinhardt@ntnu.no for access.

The hourly series is linearly interpolated onto the environment step grid
(``step_size_s``), so ``sample(time_index)`` returns the W/m^2 value at env step
``time_index``. This is a plant disturbance: keep it OUT of the MPC control model
to preserve the benchmark's plant-model mismatch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PARQUET = Path(__file__).resolve().parent / "assets" / "elimp_per_apartment.parquet"


class DisturbanceSampler:
    """Serve a random building's electricity column as an indoor heat disturbance."""

    def __init__(self, parquet_path=DEFAULT_PARQUET, step_size_s: float = 900.0, seed=None):
        self.step_size_s = float(step_size_s)
        self._data = pd.read_parquet(parquet_path)  # hourly, W/m^2, tz-aware
        self.building_id: str | None = None
        self._signal: np.ndarray | None = None  # step-resolution W/m^2
        self._index: pd.DatetimeIndex | None = None
        self.reset(seed)  # ready to use immediately

    def reset(self, rng=None) -> str:
        """Pick a random building column and build its step-resolution signal.

        ``rng`` may be an int seed, a numpy Generator, or None (fresh entropy).
        Returns the chosen building id.
        """
        gen = np.random.default_rng(rng)
        self.building_id = str(gen.choice(self._data.columns.to_numpy()))
        hourly = self._data[self.building_id]
        step = pd.Timedelta(seconds=self.step_size_s)
        fine = pd.date_range(hourly.index[0], hourly.index[-1], freq=step)
        self._index = fine
        self._signal = (
            hourly.reindex(hourly.index.union(fine))
            .interpolate("time", limit_direction="both")
            .reindex(fine)
            .to_numpy()
        )
        return self.building_id

    def sample(self, time_index: int) -> float:
        """Indoor heat disturbance [W/m^2] at environment step ``time_index``."""
        return float(self._signal[time_index])

    @property
    def n_steps(self) -> int:
        return 0 if self._signal is None else len(self._signal)
