from pathlib import Path

import pytest

# I4bEnv imports the i4b submodule at construction; skip cleanly when absent.
pytest.importorskip("i4b")

from i4b.disturbances import get_int_gains as get_int_gains_din  # noqa: E402

from leap_c.examples.i4b.env import (  # noqa: E402
    _DIN_INT_GAIN_PEAK_W_PER_M2,
    _INT_GAIN_PEAK_JITTER,
    BUILDING_NAMES2CLASS,
    Heatpump_AW,
    I4bEnv,
    I4bEnvConfig,
)

# DIN EN 16798-1 ResidentialFlat profile the env's peak constant is extracted from.
_RESIDENTIALFLAT_CSV = (
    Path(__file__).parents[3] / "external/i4b/i4b_data/profiles/InternalGains/ResidentialFlat.csv"
)


def test_int_gains_max_scaled_to_din_peak():
    """The COFACTOR internal-gain peak matches the DIN EN 16798-1 peak within +/-10%."""
    cfg = I4bEnvConfig(
        building_params=BUILDING_NAMES2CLASS["i4c"],
        hp_model=Heatpump_AW(mdot_HP=0.25),
        seed=0,
    )
    env = I4bEnv(cfg=cfg)

    area_floor = cfg.building_params["area_floor"]
    cofactor_max = float(env.dataset.data["Qdot_int_tot"].max())

    din = get_int_gains_din(env.dataset.data.index, _RESIDENTIALFLAT_CSV, bldg_area=area_floor)
    din_max = float(din["Qdot_tot"].max())

    # The env constant reproduces the standard's peak on the same floor area.
    assert din_max == pytest.approx(_DIN_INT_GAIN_PEAK_W_PER_M2 * area_floor, rel=1e-3)

    # Both internal-gain peaks agree within the +/-jitter band.
    ratio = cofactor_max / din_max
    assert 1.0 - _INT_GAIN_PEAK_JITTER - 1e-3 <= ratio <= 1.0 + _INT_GAIN_PEAK_JITTER + 1e-3
