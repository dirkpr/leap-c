import contextlib
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ----------------------------------------------------------------------------
# Hermetic stubs for the gitignored hvac data assets.
#
# The hvac example loads weather/price CSVs that are .gitignore'd, so a fresh
# CI checkout has none of them. Without these stubs the example tests fall
# back to live open-meteo/energy-charts requests (which crash).
# ----------------------------------------------------------------------------


def _make_synthetic_weather() -> pd.DataFrame:
    """Synthetic 15-min weather frame matching ``get_open_meteo_data`` output."""
    index = pd.date_range("2020-01-01", "2021-12-31 23:45", freq="15min", tz="UTC")
    index.name = "Timestamp"
    rng = np.random.default_rng(0)
    n = len(index)
    return pd.DataFrame(
        {
            "date": index,
            "temperature_2m": rng.uniform(-10, 20, n).astype(np.float32),
            "apparent_temperature": rng.uniform(-15, 18, n).astype(np.float32),
            "shortwave_radiation": rng.uniform(0, 200, n).astype(np.float32),
            "direct_normal_irradiance": rng.uniform(0, 300, n).astype(np.float32),
            "diffuse_radiation": rng.uniform(0, 150, n).astype(np.float32),
        },
        index=index,
    )


def _make_synthetic_price() -> pd.DataFrame:
    """Synthetic price frame matching ``get_energy_charts_data`` output."""
    index = pd.date_range("2020-01-01", "2021-12-31 23:45", freq="15min", tz="UTC")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "Timestamp": index,
            "price": rng.uniform(0.0, 0.5, len(index)).astype(np.float32),
        }
    )


@pytest.fixture(autouse=True)
def _stub_external_assets():
    """Make the hvac example tests hermetic when the data assets are absent.

    ``get_open_meteo_data`` / ``get_energy_charts_data`` are only invoked on a
    missing CSV (``FileNotFoundError``), so patching them is inert when the real
    CSVs exist locally and returns synthetic data (no network) in CI.
    """
    patches = [
        patch(
            "leap_c.examples.hvac.dataset.get_open_meteo_data",
            side_effect=lambda *a, **k: _make_synthetic_weather(),
        ),
        patch(
            "leap_c.examples.hvac.dataset.get_energy_charts_data",
            side_effect=lambda *a, **k: _make_synthetic_price(),
        ),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield
