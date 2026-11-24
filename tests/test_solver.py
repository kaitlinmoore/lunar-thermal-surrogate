"""Analytic and convergence tests for lunar_thermal.solver.

Each test isolates one piece of the solver:

(a) interior conduction, against the exact damped temperature wave
    (constant properties, prescribed sinusoidal surface temperature);
(b) the radiative surface and geothermal bottom boundaries, against the exact
    steady state (constant properties, no sunlight);
(c) temperature- and depth-dependent conductivity, against an independent
    integration of K(z, T) dT/dz = Q;
(d) spin-up bookkeeping: converged runs say so, capped runs say they aren't;
(e) self-convergence: refining space and time changes little.

Plus the solver contract (shapes, depth outputs, input checks) and agreement
between the compiled property formulas and lunar_thermal.properties.

Every tolerance is stated in the test with the measured error it guards, so a
change that worsens accuracy shows up as a failure rather than passing
silently.
"""

import numpy as np
import pytest
from synthetic_forcing import SYNODIC_MONTH, equatorial_absorbed_flux

from lunar_thermal import solver as S
from lunar_thermal.properties import (
    STEFAN_BOLTZMANN,
    RegolithParams,
    conductivity,
    heat_capacity,
    specific_enthalpy,
)

N_SAMPLES = 708  # hourly sampling of one synodic month (dt = 3603.7 s)

# Constant-property regolith for the analytic cases: uniform density and
# conductivity, no radiative term, constant heat capacity.
RHO, K, C = 1500.0, 1.0e-3, 600.0
CONSTANT = RegolithParams(rho_s=RHO, rho_d=RHO, k_s=K, k_d=K, chi=0.0, cp_coeffs=(C,))


@pytest.fixture(scope="module")
def equatorial():
    """Flat equatorial forcing and its default-settings solution, shared by tests."""
    flux, dt = equatorial_absorbed_flux(N_SAMPLES)
    result = S.solve_periodic(flux, dt, output_depths=[0.0, 0.05, 0.2])
    return flux, dt, result


# --- (a) Damped temperature wave --------------------------------------------


def test_a_damped_temperature_wave():
    """Constant properties, T(0, t) = T0 + A sin(wt): the exact periodic solution is

        T(z, t) = T0 + A exp(-z/d) sin(wt - z/d),  d = sqrt(2 kappa / w)

    (Carslaw and Jaeger 1959, Section 2.6). The geothermal flux is set to zero
    so the bottom is insulated, matching the semi-infinite solution to within
    exp(-10) at the default bottom depth.

    Tolerance: 0.25 K, i.e. 0.25% of the 100 K amplitude, at depths from d/4 to
    3d. Measured error at default settings: 0.125 K.
    """
    params = RegolithParams(
        rho_s=RHO, rho_d=RHO, k_s=K, k_d=K, chi=0.0, cp_coeffs=(C,), geothermal_flux=0.0
    )
    dt = SYNODIC_MONTH / N_SAMPLES
    t = np.arange(N_SAMPLES) * dt
    w = 2 * np.pi / SYNODIC_MONTH
    d = np.sqrt(2 * K / (RHO * C) / w)
    T0, A = 250.0, 100.0
    depths = np.array([0.25, 0.5, 1.0, 2.0, 3.0]) * d

    result = S._solve_periodic_fixed_surface(
        T0 + A * np.sin(w * t),
        dt,
        params,
        output_depths=depths,
        spinup=S.SpinupSettings(tolerance=1e-4),
    )
    exact = T0 + A * np.exp(-depths / d) * np.sin(w * t[:, None] - depths / d)

    assert result.converged
    np.testing.assert_allclose(result.depth_temperature, exact, rtol=0, atol=0.25)


# --- (b) Steady state, constant properties ----------------------------------


def test_b_steady_state_constant_properties():
    """No sunlight, constant properties: the exact steady state is

        T_s = (Q / (eps sigma))^(1/4),  T(z) = T_s + Q z / K.

    With uniform conductivity the finite-volume fluxes are exact for a linear
    profile, so the only error is spin-up. Tolerance: 1e-3 K, for a spin-up
    tolerance of 1e-4 K. Measured error: 4e-5 K.
    """
    dt = SYNODIC_MONTH / N_SAMPLES
    result = S.solve_periodic(
        np.zeros(N_SAMPLES),
        dt,
        CONSTANT,
        spinup=S.SpinupSettings(tolerance=1e-4, max_cycles=1000),
    )
    Q = CONSTANT.geothermal_flux
    Ts = (Q / (CONSTANT.emissivity * STEFAN_BOLTZMANN)) ** 0.25  # 24.04 K
    exact = Ts + Q * result.grid / K

    assert result.converged
    np.testing.assert_allclose(result.surface_temperature, Ts, rtol=0, atol=1e-3)
    np.testing.assert_allclose(result.end_profile, exact, rtol=0, atol=1e-3)


# --- (c) Steady state, temperature-dependent conductivity -------------------


def _integrate_steady_profile(z_out, Ts, Q, params, h=1e-5):
    """Integrate dT/dz = Q / K(z, T) from T(0) = Ts with classical RK4.

    Independent of the solver: uses only properties.conductivity and a fixed
    step of 10 micrometres, far finer than the solver grid.
    """

    def f(z, T):
        return Q / float(conductivity(z, T, params))

    z, T = 0.0, Ts
    out = [T]
    for target in z_out[1:]:
        while z < target - 1e-15:
            step = min(h, target - z)
            k1 = f(z, T)
            k2 = f(z + step / 2, T + step * k1 / 2)
            k3 = f(z + step / 2, T + step * k2 / 2)
            k4 = f(z + step, T + step * k3)
            T += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            z += step
        out.append(T)
    return np.array(out)


def test_c_steady_state_temperature_dependent_conductivity():
    """No sunlight, standard Hayne et al. profile, geothermal flux raised to 0.5 W m^-2.

    At steady state every layer carries the same flux, so K(z, T) dT/dz = Q
    with T(0) = (Q / (eps sigma))^(1/4). The raised flux spreads temperatures
    from 55 K at the surface to 166 K at the bottom (default H = 0.068 m), so
    conductivity varies with both depth (Kc from Ks to Kd) and temperature
    (the radiative term adds 29% at the bottom). The default Q = 0.018 W m^-2 would leave the
    column near 24-30 K, where the radiative term is negligible and the test
    would not exercise it.

    Tolerance: 0.05 K at every node. Measured: 0.007 K (the solver's
    interface conductivity is the mean of its two nodes, a second-order
    approximation).
    """
    params = RegolithParams(geothermal_flux=0.5)
    dt = SYNODIC_MONTH / N_SAMPLES
    result = S.solve_periodic(
        np.zeros(N_SAMPLES),
        dt,
        params,
        spinup=S.SpinupSettings(tolerance=1e-5, max_cycles=2000),
    )
    Q = params.geothermal_flux
    Ts = (Q / (params.emissivity * STEFAN_BOLTZMANN)) ** 0.25
    expected = _integrate_steady_profile(result.grid, Ts, Q, params)

    assert result.converged
    radiative_share = params.chi * (expected[-1] / 350.0) ** 3
    assert radiative_share > 0.2, "test should exercise the T-dependent term"
    np.testing.assert_allclose(result.end_profile, expected, rtol=0, atol=0.05)


# --- (d) Spin-up reporting ---------------------------------------------------


def test_d_converged_run_reports_convergence(equatorial):
    """Default spin-up converges and reports both criteria below tolerance."""
    _, _, result = equatorial
    assert result.converged
    assert result.residual < result.tolerance
    assert result.drift_estimate < result.tolerance
    assert 3 <= result.n_cycles < S.SpinupSettings().max_cycles


def test_d_converged_result_is_close_to_true_periodic_state(equatorial):
    """The convergence claim is honest: the result is near the true periodic state.

    Reference: plain repetition (no extrapolation) to 1e-6 K. Tolerance: 2x
    the spin-up tolerance, since the projected drift is an estimate. Measured:
    0.008 K for a 0.01 K tolerance (default H = 0.068 m; 0.006 K at 0.06 m).
    """
    flux, dt, result = equatorial
    reference = S.solve_periodic(
        flux,
        dt,
        spinup=S.SpinupSettings(tolerance=1e-6, accelerate=False, max_cycles=3000),
    )
    assert reference.converged
    err = np.max(np.abs(result.node_temperature - reference.node_temperature))
    assert err < 2 * result.tolerance


def test_d_cycle_cap_reports_non_convergence():
    """With too few cycles allowed, the flag is false, the residual says why,
    and a RuntimeWarning is raised."""
    flux, dt = equatorial_absorbed_flux(N_SAMPLES)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        result = S.solve_periodic(flux, dt, spinup=S.SpinupSettings(max_cycles=3))
    assert not result.converged
    assert result.n_cycles == 3
    assert (
        result.residual > result.tolerance or result.drift_estimate > result.tolerance
    )
    assert np.isfinite(result.residual)


def test_d_single_cycle_has_no_residual():
    """One cycle can't be compared with anything: residual is inf, not 0."""
    flux, dt = equatorial_absorbed_flux(N_SAMPLES)
    with pytest.warns(RuntimeWarning):
        result = S.solve_periodic(flux, dt, spinup=S.SpinupSettings(max_cycles=1))
    assert not result.converged
    assert result.residual == np.inf


# --- (e) Self-convergence ----------------------------------------------------


def test_e_self_convergence(equatorial):
    """Halving the grid spacing and the time step barely changes the surface temperature.

    Default grid (m = 20, n = 10, max_step = 1800 s) against m = 40, n = 20,
    max_step = 900 s: the layer thicknesses near the surface and the time
    step both halve, and the growth factor (1 + 1/n) is close to the square
    root of the default's, so every layer is close to half as thick.

    Tolerance: 0.25 K maximum and 0.05 K rms over the cycle. Measured: 0.15 K
    maximum (at sunrise, where the surface warms fastest) and 0.017 K rms, at
    the default H = 0.068 m (the same at 0.06 m).
    """
    flux, dt, coarse = equatorial
    fine = S.solve_periodic(
        flux, dt, grid=S.GridSettings(layers_per_skin_depth=40, growth=20, max_step=900)
    )
    assert fine.grid.size > 1.7 * coarse.grid.size
    diff = fine.surface_temperature - coarse.surface_temperature
    assert np.max(np.abs(diff)) < 0.25
    assert np.sqrt(np.mean(diff**2)) < 0.05


def test_bottom_boundary_is_deep_enough(equatorial):
    """Moving the bottom from 10 to 15 deep skin depths changes nothing that matters.

    Tolerance: 0.01 K at the surface, 5 cm and 20 cm. Measured: 0.0035 K
    (default H = 0.068 m; 0.0015 K at 0.06 m), which is spin-up tolerance
    rather than the bottom boundary.
    """
    flux, dt, default = equatorial
    deeper = S.solve_periodic(
        flux,
        dt,
        grid=S.GridSettings(bottom_skin_depths=15),
        output_depths=[0.0, 0.05, 0.2],
    )
    np.testing.assert_allclose(
        deeper.surface_temperature, default.surface_temperature, atol=0.01
    )
    np.testing.assert_allclose(
        deeper.depth_temperature, default.depth_temperature, atol=0.01
    )


# --- Contract ----------------------------------------------------------------


def test_result_shapes_and_times(equatorial):
    _, dt, result = equatorial
    n_z = result.grid.size
    assert result.times.shape == (N_SAMPLES,)
    np.testing.assert_allclose(result.times, np.arange(N_SAMPLES) * dt)
    assert result.surface_temperature.shape == (N_SAMPLES,)
    assert result.node_temperature.shape == (N_SAMPLES, n_z)
    assert result.end_profile.shape == (n_z,)
    assert result.depth_temperature.shape == (N_SAMPLES, 3)
    assert result.grid[0] == 0.0 and np.all(np.diff(result.grid) > 0)


def test_depth_outputs_interpolate_nodes(equatorial):
    """Depth 0 is the surface; a depth on a node is that node exactly."""
    _, _, result = equatorial
    np.testing.assert_array_equal(
        result.depth_temperature[:, 0], result.surface_temperature
    )
    flux, dt, _ = equatorial
    node_depth = result.grid[5]
    r = S.solve_periodic(flux, dt, output_depths=node_depth)
    np.testing.assert_allclose(
        r.depth_temperature[:, 0], r.node_temperature[:, 5], rtol=0, atol=1e-12
    )


def test_no_depths_gives_none():
    flux, dt = equatorial_absorbed_flux(N_SAMPLES)
    result = S.solve_periodic(flux, dt)
    assert result.depth_temperature is None
    assert result.depths.size == 0


def test_end_of_cycle_matches_start_at_steady_state(equatorial):
    """At periodic steady state, t = period is t = 0 again, within the residual."""
    _, _, result = equatorial
    assert (
        np.max(np.abs(result.end_profile - result.node_temperature[0]))
        <= result.residual + 1e-12
    )


@pytest.mark.parametrize(
    "flux, dt, kwargs",
    [
        (np.array([1.0, -1.0, 1.0]), 100.0, {}),  # negative flux
        (np.array([1.0, np.nan, 1.0]), 100.0, {}),  # non-finite flux
        (np.array([1.0]), 100.0, {}),  # a single sample is not a period
        (np.ones(10), 0.0, {}),  # zero step
        (np.ones(10), 3600.0, {"output_depths": [100.0]}),  # below the grid
        (np.ones(10), 3600.0, {"output_depths": [-0.01]}),  # above the surface
    ],
)
def test_invalid_inputs_rejected(flux, dt, kwargs):
    with pytest.raises(ValueError):
        S.solve_periodic(flux, dt, **kwargs)


# --- Compiled kernels agree with properties.py ------------------------------


def test_kernel_property_formulas_match_properties_module():
    """The solver's compiled cp(T) and h(T) are the same formulas as properties.py."""
    p = RegolithParams()
    cp = np.asarray(p.cp_coeffs)
    for T in [20.0, 95.0, 250.0, 390.0]:
        assert S._cp_scalar(T, cp) == pytest.approx(
            float(heat_capacity(T, p)), rel=1e-12
        )
        assert S._h_scalar(T, cp) == pytest.approx(
            float(specific_enthalpy(T, p)), rel=1e-12
        )
