"""Energy balance of the solver over one converged cycle.

At periodic steady state nothing is stored from one cycle to the next, so over
a cycle the energy in (absorbed sunlight plus geothermal heat) equals the
energy out (thermal emission), and the column's heat content returns to its
starting value. This is the same check the project applies to surrogate
outputs (planning package, Section 5), computed the same way: from the
forcing and the surface temperature at the sample times.
"""

import numpy as np
import pytest
from synthetic_forcing import equatorial_absorbed_flux

from lunar_thermal import solver as S
from lunar_thermal.properties import STEFAN_BOLTZMANN, RegolithParams

N_SAMPLES = 708  # hourly sampling of one synodic month (dt = 3602 s)
PARAMS = RegolithParams()


def _cycle_means(flux, result, params):
    """Cycle-mean absorbed, geothermal and emitted fluxes, W m^-2, from samples."""
    emitted = params.emissivity * STEFAN_BOLTZMANN * result.surface_temperature**4
    return float(np.mean(flux)), params.geothermal_flux, float(np.mean(emitted))


@pytest.fixture(scope="module")
def converged():
    flux, dt = equatorial_absorbed_flux(N_SAMPLES)
    return flux, dt, S.solve_periodic(flux, dt, PARAMS)


def test_energy_in_equals_energy_out(converged):
    """Mean absorbed + geothermal = mean emitted, from the sampled series.

    Tolerance: 0.005 W m^-2, about a quarter of the geothermal flux and 1.4e-5
    of the mean absorbed flux (354 W m^-2), so the check would catch a lost
    or doubled geothermal term. Measured: 0.0009 W m^-2. The remaining
    imbalance comes from sampling: emission is averaged over hourly samples,
    while the solver steps three times per hour.
    """
    flux, _, result = converged
    absorbed, geothermal, emitted = _cycle_means(flux, result, PARAMS)
    assert result.converged
    assert abs(absorbed + geothermal - emitted) < 0.005


def test_stored_heat_returns_to_start(converged):
    """Column heat content at the end of the cycle equals that at its start.

    Tolerance: 1e-4 of the heat content's swing over the day (1.0e7 J m^-2).
    Measured: 1.5e-5, set by the spin-up tolerance (0.01 K).
    """
    _, _, result = converged
    H = S.column_heat_content(result.grid, result.node_temperature, PARAMS)
    H_end = S.column_heat_content(result.grid, result.end_profile, PARAMS)
    swing = H.max() - H.min()
    assert swing > 1e6  # the day actually stores and releases heat
    assert abs(H_end - H[0]) < 1e-4 * swing


def test_discrete_energy_conservation():
    """With one internal step per sample, net boundary flux equals the stored-heat change.

    This checks that the scheme conserves energy, as opposed to merely being
    close to steady state. A loose spin-up tolerance is used on purpose, so
    the column is still drifting and the heat-content change is not
    negligible; a scheme that lost energy would not match it.

    Summed over the N steps of a cycle, the BDF2 updates (3 h^{n+1} - 4 h^n +
    h^{n-1}) / 2 telescope to 1.5 (h^N - h^0) - 0.5 (h^{N-1} - h^{-1}): the
    cycle's heat change, measured from two starting points one step apart.
    When the drift is smooth these agree, so net boundary flux times the
    period equals H_end - H_start. Tolerance: 1e-6 W m^-2 on the difference
    of the two rates, against a stored rate of ~6e-5 W m^-2. Measured: 2e-9.
    """
    flux, dt = equatorial_absorbed_flux(N_SAMPLES)
    result = S.solve_periodic(
        flux,
        dt,
        PARAMS,
        grid=S.GridSettings(max_step=2 * dt),
        spinup=S.SpinupSettings(tolerance=0.05),
    )
    assert result.substeps == 1
    absorbed, geothermal, emitted = _cycle_means(flux, result, PARAMS)
    period = N_SAMPLES * dt
    H_start = S.column_heat_content(result.grid, result.node_temperature[0], PARAMS)
    H_end = S.column_heat_content(result.grid, result.end_profile, PARAMS)
    stored_rate = (H_end - H_start) / period
    net_rate = absorbed + geothermal - emitted
    assert stored_rate != 0.0
    assert net_rate == pytest.approx(stored_rate, abs=1e-6)
