"""Parametric acados OCP for the minimal BOPTEST heat-pump example.

The prediction model is a hand-crafted single-state R1C1 thermal model of the
BESTEST case-900 zone (see the ``bestest_hydronic_heat_pump`` BOPTEST case).
BOPTEST itself exposes no state-space model (the emulator is a black-box
Modelica FMU), so this rough grey-box is used purely to confirm signal wiring;
its R/C/gain values and the cost weights are registered as *differentiable*
parameters so they can later be tuned/learned.

Exogenous inputs that come from the environment as perfect forecasts over the
horizon (ambient temperature, solar irradiance, electricity price and the
time-varying comfort bounds) are registered as *non-differentiable stagewise*
parameters and set per solve by the planner.
"""

from collections import OrderedDict

import casadi as ca
import gymnasium as gym
import numpy as np
from acados_template import AcadosOcp

from leap_c.ocp.acados.parameters import AcadosParameterManager

# Canonical order of the forecast / exogenous quantities fed to the OCP as
# non-differentiable stagewise parameters. The environment decodes its flat
# observation into forecast arrays keyed by exactly these names.
FORECAST_PARAM_NAMES = ["T_amb", "solar", "price", "T_lower", "T_upper"]

# Nominal (fallback) forecast values, used when the planner does not overwrite a
# stagewise parameter (e.g. in offline unit tests). Temperatures in Kelvin.
FORECAST_DEFAULTS = {
    "T_amb": 278.15,  # 5 degC ambient
    "solar": 0.0,  # W/m2 global horizontal irradiance
    "price": 0.2,  # EUR/kWh electricity price
    "T_lower": 294.15,  # 21 degC lower comfort bound
    "T_upper": 297.15,  # 24 degC upper comfort bound
}


def export_parametric_ocp(
    name: str = "boptest",
    N_horizon: int = 12,
    step_period: float = 900.0,
) -> tuple[AcadosOcp, AcadosParameterManager, gym.spaces.Dict, dict[str, np.ndarray]]:
    """Build the parametric acados OCP for the BOPTEST heat-pump example.

    Args:
        name: Name of the acados model / generated solver.
        N_horizon: Number of shooting intervals in the MPC horizon.
        step_period: Sampling time in seconds (matches the BOPTEST step period).

    Returns:
        A tuple ``(ocp, manager, param_space, default_param)`` mirroring the
        other examples: the OCP, the parameter manager, the differentiable
        parameter space and the corresponding default parameter dictionary.
    """
    ocp = AcadosOcp()
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = N_horizon * step_period

    dt = step_period

    manager = AcadosParameterManager(N_horizon=N_horizon)

    # ---------------------------------------------------------------------
    # Differentiable (tunable / learnable) global parameters.
    # ---------------------------------------------------------------------
    spaces: OrderedDict[str, gym.spaces.Box] = OrderedDict()
    defaults: dict[str, np.ndarray] = {}

    def register_differentiable(pname: str, default: float, low: float, high: float) -> ca.SX:
        default_arr = np.array([default])
        symbol = manager.register_parameter(
            pname, default=default_arr, differentiable=True, splits="global"
        )
        spaces[pname] = gym.spaces.Box(low=np.array([low]), high=np.array([high]), dtype=np.float64)
        defaults[pname] = default_arr
        return symbol

    # R1C1 thermal model gains (rough BESTEST case-900 values).
    R = register_differentiable("R", default=0.01, low=0.005, high=0.05)  # K/W
    C = register_differentiable("C", default=3.0e6, low=1.0e6, high=1.0e7)  # J/K
    a_sol = register_differentiable("a_sol", default=2.0, low=0.0, high=10.0)  # m2
    heat_gain = register_differentiable(
        "heat_gain", default=5000.0, low=1000.0, high=10000.0
    )  # W thermal at full modulation
    cop = register_differentiable("cop", default=3.0, low=1.5, high=5.0)  # -
    w_comfort = register_differentiable("w_comfort", default=1.0, low=0.0, high=100.0)

    # ---------------------------------------------------------------------
    # Non-differentiable stagewise parameters (set from the env forecast).
    # ---------------------------------------------------------------------
    for pname in FORECAST_PARAM_NAMES:
        manager.register_parameter(
            pname,
            default=np.array([FORECAST_DEFAULTS[pname]]),
            differentiable=False,
        )
    T_amb = manager.get("T_amb")
    solar = manager.get("solar")
    price = manager.get("price")
    T_lower = manager.get("T_lower")
    T_upper = manager.get("T_upper")

    # ---------------------------------------------------------------------
    # Model: single state (zone temperature), single input (HP modulation).
    # ---------------------------------------------------------------------
    ocp.model.name = name
    ocp.dims.nx = 1
    ocp.dims.nu = 1
    T = ca.SX.sym("T", 1)
    u = ca.SX.sym("u", 1)  # heat-pump modulation in [0, 1] (oveHeaPumY_u)
    ocp.model.x = T
    ocp.model.u = u

    # Exact zero-order-hold discretization of the R1C1 dynamics
    #   C dT/dt = (T_amb - T)/R + a_sol * solar + heat_gain * u.
    # Steady state for constant inputs: T_ss = T_amb + R * q, with total heat
    # input q = a_sol * solar + heat_gain * u.
    q = a_sol * solar + heat_gain * u
    tau = R * C
    ad = ca.exp(-dt / tau)
    T_ss = T_amb + R * q
    ocp.model.disc_dyn_expr = ad * T + (1.0 - ad) * T_ss

    # ---------------------------------------------------------------------
    # Cost (EXTERNAL): electricity cost + soft comfort-band penalty.
    # ---------------------------------------------------------------------
    # Electrical energy this step [kWh] = heat_gain * u / cop * dt / 3.6e6.
    energy_cost = price * (heat_gain * u / cop) * (dt / 3.6e6)
    comfort = w_comfort * (ca.fmax(0.0, T_lower - T) ** 2 + ca.fmax(0.0, T - T_upper) ** 2)
    ocp.cost.cost_type = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = energy_cost + comfort
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost_e = w_comfort * (
        ca.fmax(0.0, T_lower - T) ** 2 + ca.fmax(0.0, T - T_upper) ** 2
    )

    # ---------------------------------------------------------------------
    # Constraints: hard box on the modulation, placeholder initial state.
    # ---------------------------------------------------------------------
    ocp.constraints.x0 = np.array([293.15])
    ocp.constraints.lbu = np.array([0.0])
    ocp.constraints.ubu = np.array([1.0])
    ocp.constraints.idxbu = np.array([0])

    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_ric_alg = 1

    return ocp, manager, gym.spaces.Dict(spaces), defaults
