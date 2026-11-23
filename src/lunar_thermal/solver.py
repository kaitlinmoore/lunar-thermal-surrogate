"""One-dimensional regolith heat solver at periodic steady state.

This is the project's ground truth: given one period of absorbed solar flux at
the surface, it returns the temperatures the regolith settles into when that
period repeats indefinitely. The FNO surrogate is trained to imitate it.

Physics (Hayne et al. 2017, Appendix A; equation numbers from the arXiv
preprint, arXiv:1711.00977):

* heat equation ``rho cp dT/dt = d/dz (K dT/dz)`` (Eq. A1), with density
  ``rho(z)`` (Eq. A2), conductivity ``K(z, T)`` (Eqs. A4, A5) and heat capacity
  ``cp(T)`` (Eq. A6), all from `lunar_thermal.properties`;
* surface: absorbed flux plus conduction from below balances thermal emission,
  ``K dT/dz|_0 + Q_s = eps sigma T_s^4`` (Eq. A7);
* bottom: the geothermal flux ``Q`` enters from below (Eq. A12).

Numerics (this module's own choice, not taken from the paper or `heat1d`):

* **Finite volumes on a nonuniform grid.** Nodes sit at depths ``z_0 = 0 <
  z_1 < ... < z_{N-1}``. Each node owns the slab halfway to its neighbours
  (half slabs at the surface and the bottom). Heat flows between neighbouring
  nodes with the mean of their conductivities. Every joule that leaves one
  slab enters the next, so the scheme conserves energy exactly.
* **Grid tied to the skin depth.** The top spacing is a fraction of the
  diurnal skin depth of the surface material; spacing then grows
  geometrically; the bottom sits several skin depths of the deep material
  down, where the periodic wave has died out (`make_grid`).
* **Implicit time stepping (BDF2) on enthalpy.** Each step solves for the new
  temperatures with the second-order backward differentiation formula,
  applied to heat content ``h(T) = integral of cp dT`` rather than to ``T``.
  Implicit steps are stable for any step size, so the step is set by accuracy,
  not by the thin surface layer; stepping ``h`` keeps the energy budget exact
  even though ``cp`` changes with temperature.
* **Newton iteration.** The radiative boundary (``T^4``) and the
  temperature-dependent ``K`` and ``cp`` make each step nonlinear. Newton's
  method on the whole column (a tridiagonal system) solves it to 1e-8 K,
  usually in two or three iterations.
* **Spin-up.** The period is repeated until every node's temperature at every
  sample time changes by less than a tolerance from one cycle to the next,
  and the drift still to come (projected from how fast the changes are
  shrinking) is below the same tolerance. Once the slowest mode's decay is
  steady, the solver extrapolates to its end, which cuts spin-up about
  fivefold (see `SpinupSettings`). The result carries a converged flag, the
  final change and the projected drift; an unconverged run also raises a
  `RuntimeWarning`.

The public entry point is `solve_periodic`. `make_grid` and
`column_heat_content` are public helpers. Functions with a leading underscore
are internal; `_solve_periodic_fixed_surface` exists for the analytic tests.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray

from lunar_thermal.properties import (
    RADIATIVE_REFERENCE_TEMPERATURE,
    STEFAN_BOLTZMANN,
    RegolithParams,
    contact_conductivity,
    density,
    skin_depth,
    specific_enthalpy,
)

# Newton iteration: stop when the largest temperature update in the column is
# below this, in kelvin. Far below any accuracy that matters, so the result
# doesn't depend on it; loose enough that double precision reaches it.
_NEWTON_TOL = 1e-8
_NEWTON_MAX_ITER = 50

# The heat-capacity polynomial (Eq. A6) goes negative below ~1.3 K, which would
# make the problem ill-posed. Lunar temperatures never approach this (the
# geothermal flux alone holds the surface above ~24 K), so reaching it means
# the iteration has gone wrong; the step is flagged as failed.
_MIN_TEMPERATURE = 2.0

# Spin-up extrapolation (see SpinupSettings). Extrapolate only when the last
# three estimates of the slow mode's decay ratio lie within _RATIO_STABILITY of
# each other, and only if the ratio is at most _MAX_EXTRAPOLATION_RATIO. Both
# guards come from a failure seen in testing: while transients settle, the
# ratio estimate can drift slowly through values near 1, look stable for two
# cycles, and trigger a jump of r / (1 - r) > 100 times the last change that
# overshoots. A ratio cap of 0.98 limits the jump to 49 times the last change;
# runs whose slow mode decays more slowly than that just aren't accelerated.
_RATIO_STABILITY = 0.002
_MAX_EXTRAPOLATION_RATIO = 0.98

_SURFACE_RADIATIVE = 0
_SURFACE_FIXED = 1


@dataclass(frozen=True)
class GridSettings:
    """Space and time discretization. Each default is a logged instrument.

    The grid follows the scheme of Hayne et al. (2017, Eqs. A32-A33): the first
    layer is a fraction of the skin depth thick, and thicknesses then grow by a
    constant factor. Each default is a logged instrument (`docs/changelog.md`),
    chosen from a convergence study on equatorial forcing against a reference
    run with four times finer spacing and a 30-60 s step:

    * Space dominates the error. With the paper's suggested ``m = 10, n = 5``
      the surface temperature is off by up to 0.7 K (at sunrise; rms 0.08 K)
      and the deep mean by 0.5 K. Halving the spacing (``m = 20, n = 10``,
      the defaults) brings that to 0.2 K and 0.14 K, consistent with a
      second-order scheme, for twice the cost.
    * Time contributes little: with hourly forcing, a 1800 s internal step
      adds at most 0.015 K over a 30 s step.
    * Bottom depth: 6 deep skin depths already changes the surface by under
      0.001 K compared with 15; 10 (the paper's choice) costs only a few
      extra nodes.

    Attributes:
        layers_per_skin_depth: ``m``. The first layer is ``zs_top / m`` thick,
            where ``zs_top`` is the skin depth (Eq. A31) of the surface
            material at `reference_temperature`.
        growth: ``n``. Each layer is ``(1 + 1/n)`` times thicker than the
            one above.
        bottom_skin_depths: The column extends to this many skin depths of
            the deep material (``z >> H``) at `reference_temperature`. The
            periodic wave is damped by ``exp(-bottom_skin_depths)`` there.
        max_step: Largest internal time step, s. Each forcing interval ``dt``
            is split into ``ceil(dt / max_step)`` equal substeps; the forcing
            is interpolated linearly between samples.
        reference_temperature: Temperature at which skin depths are
            evaluated, K. Diffusivity is lowest around 200-250 K for the
            standard parameters, so 250 K gives a thin (conservative) top
            layer.
    """

    layers_per_skin_depth: float = 20.0
    growth: float = 10.0
    bottom_skin_depths: float = 10.0
    max_step: float = 1800.0
    reference_temperature: float = 250.0

    def __post_init__(self) -> None:
        if self.layers_per_skin_depth <= 0 or self.growth <= 0:
            raise ValueError("layers_per_skin_depth and growth must be positive")
        if self.bottom_skin_depths <= 0:
            raise ValueError("bottom_skin_depths must be positive")
        if self.max_step <= 0:
            raise ValueError("max_step must be positive")
        if self.reference_temperature <= _MIN_TEMPERATURE:
            raise ValueError("reference_temperature is too low")


@dataclass(frozen=True)
class SpinupSettings:
    """When to stop repeating the period, and how to speed it up.

    Spin-up is slow because of one mode: the deep column warming or cooling
    towards its final mean temperature through the insulating surface layer.
    Once the faster transients have died, each cycle shrinks that mode by a
    nearly constant ratio ``r`` (about 0.95 per lunation for the standard
    parameters at the equator). Two consequences shape this class:

    * A small change between cycles does not mean the run is close to steady
      state: the drift still to come is the geometric sum ``change * r / (1 -
      r)``, about 20 times the last change. Convergence therefore requires both
      the last change and that projected drift to be below the tolerance.
    * The same geometry allows extrapolating straight to the end of the drift
      (vector Aitken extrapolation): once ``r`` is stable over three
      successive estimates, the end-of-cycle state is moved by ``change * r /
      (1 - r)``. For equatorial forcing that cuts spin-up from 140-190 cycles
      to 25-50, and lands closer to the true periodic state (within 0.009 K
      for a 0.01 K tolerance, against 0.0105 K for plain repetition).

    Attributes:
        tolerance: K. Converged when the largest change, over every grid node
            and every sample time, between one cycle and the previous one is
            below this, and the projected remaining drift is too. Logged
            instrument (`docs/changelog.md`).
        max_cycles: Give up after this many cycles. The result then has
            ``converged = False`` and a `RuntimeWarning` is raised.
        min_cycles: Run at least this many cycles. At least 3, since the
            drift projection needs two successive changes.
        accelerate: Apply the extrapolation described above. Off gives plain
            repetition: slower, and its projected drift is only an estimate
            (it lands about 5% past the tolerance on equatorial forcing).
        initial_temperature: Uniform starting temperature, K. ``None`` uses
            the default estimate (see `_initial_temperature`).
    """

    tolerance: float = 0.01
    max_cycles: int = 200
    min_cycles: int = 3
    accelerate: bool = True
    initial_temperature: float | None = None

    def __post_init__(self) -> None:
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.min_cycles < 3:
            raise ValueError("min_cycles must be at least 3")
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be at least 1")
        if (
            self.initial_temperature is not None
            and self.initial_temperature <= _MIN_TEMPERATURE
        ):
            raise ValueError("initial_temperature is too low")


@dataclass(frozen=True)
class SolverResult:
    """Temperatures over the final cycle, and whether spin-up converged.

    Row ``k`` of each time series is the state at ``times[k] = k * dt``, the
    time of forcing sample ``k``. Row 0 is the state at the start of the final
    cycle; `end_profile` is the state at its end (``t = period``). At periodic
    steady state the two agree to within `residual`.

    Attributes:
        times: Sample times, s, shape ``(n_t,)``.
        surface_temperature: Surface temperature at each sample time, K,
            shape ``(n_t,)``.
        depths: The output depths that were requested, m, shape ``(n_d,)``
            (empty if none).
        depth_temperature: Temperature at each requested depth and sample
            time, K, shape ``(n_t, n_d)``, interpolated linearly between grid
            nodes; ``None`` if no depths were requested. Interpolation adds
            error where the profile curves: midway between nodes, up to ~1 K
            in the top centimetre and 0.3-0.4 K at 2-5 cm (equatorial case),
            against ~0.2 K at the nodes themselves.
        grid: Depth of each grid node, m, shape ``(n_z,)``. ``grid[0] = 0``.
        node_temperature: Temperature at every grid node and sample time, K,
            shape ``(n_t, n_z)``.
        end_profile: Node temperatures at the end of the final cycle, K,
            shape ``(n_z,)``.
        n_cycles: Number of periods run, including the final one.
        converged: Whether both `residual` and `drift_estimate` fell below the
            tolerance within ``max_cycles``.
        residual: The largest absolute change, over all nodes and sample
            times, between the final cycle and the one before it, K. ``inf``
            if no two comparable cycles ran.
        drift_estimate: Projected further change if the run continued, K:
            the end-of-cycle change times ``r / (1 - r)`` (see
            `SpinupSettings`). ``inf`` if it couldn't be estimated or the
            run isn't settling.
        tolerance: The spin-up tolerance used, K.
        substeps: Internal time steps per forcing interval.
        extrapolations: How many times spin-up was accelerated by
            extrapolation.
    """

    times: NDArray[np.float64]
    surface_temperature: NDArray[np.float64]
    depths: NDArray[np.float64]
    depth_temperature: NDArray[np.float64] | None
    grid: NDArray[np.float64]
    node_temperature: NDArray[np.float64]
    end_profile: NDArray[np.float64]
    n_cycles: int
    converged: bool
    residual: float
    drift_estimate: float
    tolerance: float
    substeps: int
    extrapolations: int


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def make_grid(
    params: RegolithParams, period: float, settings: GridSettings
) -> NDArray[np.float64]:
    """Node depths for a forcing of the given period, m.

    The first layer is ``zs_top / m`` thick, layers grow by ``(1 + 1/n)``
    (Hayne et al. 2017, Eq. A33), and nodes are added until the column reaches
    ``bottom_skin_depths * zs_deep``. ``zs_top`` and ``zs_deep`` are skin
    depths (Eq. A31) of the surface and deep material at the reference
    temperature. Using the deep material for the bottom matters: it is about
    1.7 times more diffusive than the surface layer for the standard
    parameters, so a column sized by the surface skin depth would be too
    shallow.

    The grid depends on the period, so a longer forcing (a year instead of a
    lunation) automatically gets a deeper column.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    T_ref = settings.reference_temperature
    zs_top = float(skin_depth(0.0, T_ref, period, params))
    # "Deep" means many scale heights down, where density and conductivity
    # have reached their deep values.
    zs_deep = float(skin_depth(50.0 * params.H, T_ref, period, params))
    zs_deep = max(zs_deep, zs_top)
    bottom = settings.bottom_skin_depths * zs_deep
    dz = zs_top / settings.layers_per_skin_depth
    ratio = 1.0 + 1.0 / settings.growth
    z = [0.0]
    while z[-1] < bottom:
        z.append(z[-1] + dz)
        dz *= ratio
    return np.asarray(z, dtype=np.float64)


def column_heat_content(
    grid: ArrayLike, T: ArrayLike, params: RegolithParams
) -> NDArray[np.float64]:
    """Heat content of the column per unit area, J m^-2, relative to 0 K.

    Sums ``rho_i * w_i * h(T_i)`` over the solver's control volumes (``w_i``
    is the thickness of the slab node ``i`` owns), so changes in this quantity
    are exactly the energy the solver has moved across the boundaries. ``T``
    may be one profile (shape ``(n_z,)``) or a time series (shape ``(n_t,
    n_z)``); the result has the leading shape of ``T`` minus the depth axis.
    """
    grid = np.asarray(grid, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    mass = density(grid, params) * _control_widths(grid)
    return np.sum(mass * specific_enthalpy(T, params), axis=-1)


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------


def solve_periodic(
    absorbed_flux: ArrayLike,
    dt: float,
    params: RegolithParams | None = None,
    grid: GridSettings | None = None,
    spinup: SpinupSettings | None = None,
    output_depths: ArrayLike | None = None,
) -> SolverResult:
    """Temperatures at periodic steady state for one period of absorbed flux.

    This is the solver contract used by the forcing and dataset code and by
    evaluation. The forcing is one period sampled at a uniform step; the
    period is ``len(absorbed_flux) * dt``, and the series wraps around (sample
    ``n`` would be sample 0 again). Between samples the flux is taken to vary
    linearly.

    Args:
        absorbed_flux: Absorbed solar flux at the surface, W m^-2, one value
            per sample time ``k * dt``, ``k = 0 .. n_t - 1``. Albedo and
            incidence angle are already applied (that is `forcing.py`'s job).
            Must be finite and non-negative; zeros (night, shadow) are fine.
        dt: Sampling step of the forcing, s.
        params: Regolith parameters; defaults to Hayne et al. (2017).
        grid: Space and time discretization; defaults to `GridSettings()`.
        spinup: Spin-up control; defaults to `SpinupSettings()`.
        output_depths: Depths, m, at which to report temperature in addition
            to the surface. Each must lie within the grid. ``None`` for
            surface only.

    Returns:
        A `SolverResult`. Check ``result.converged``: an unconverged result
        is returned (with a `RuntimeWarning`) rather than raised, so the
        caller can decide whether its residual is acceptable.

    Raises:
        ValueError: on invalid inputs.
        RuntimeError: if a time step's Newton iteration fails, which
            signals a numerical problem rather than slow spin-up.
    """
    flux = np.asarray(absorbed_flux, dtype=np.float64)
    if flux.ndim != 1 or flux.size < 2:
        raise ValueError("absorbed_flux must be a 1D array with at least 2 samples")
    if not np.all(np.isfinite(flux)):
        raise ValueError("absorbed_flux must be finite")
    if np.any(flux < 0):
        raise ValueError("absorbed_flux must be non-negative")
    return _solve(flux, _SURFACE_RADIATIVE, dt, params, grid, spinup, output_depths)


# ---------------------------------------------------------------------------
# Internal: test hook and shared driver
# ---------------------------------------------------------------------------


def _solve_periodic_fixed_surface(
    surface_temperature: ArrayLike,
    dt: float,
    params: RegolithParams | None = None,
    grid: GridSettings | None = None,
    spinup: SpinupSettings | None = None,
    output_depths: ArrayLike | None = None,
) -> SolverResult:
    """Internal, for tests: like `solve_periodic` with a prescribed surface temperature.

    The surface node is held at the given temperature series (a Dirichlet
    boundary) instead of the radiative balance. This isolates the interior
    conduction from the nonlinear boundary, so the result can be compared
    with the analytic damped temperature wave. Not part of the contract.
    """
    Ts = np.asarray(surface_temperature, dtype=np.float64)
    if Ts.ndim != 1 or Ts.size < 2 or np.any(Ts <= _MIN_TEMPERATURE):
        raise ValueError(
            "surface_temperature must be a 1D series of positive temperatures"
        )
    return _solve(Ts, _SURFACE_FIXED, dt, params, grid, spinup, output_depths)


def _initial_temperature(
    series: NDArray[np.float64], surface_mode: int, params: RegolithParams
) -> float:
    """Uniform starting temperature for spin-up, K.

    With a fixed surface, the cycle mean of the prescribed temperature. With
    the radiative boundary, the temperature of a black-ish body in radiative
    balance with the cycle-mean absorbed flux, ``((<F> + Q) / (eps sigma))^(1/4)``,
    scaled by 0.85. The factor accounts for the fact that a surface whose
    emission averages ``eps sigma <T^4>`` has a mean temperature below
    ``<T^4>^(1/4)``; 0.85 was chosen by trial on equatorial forcing, where it
    lands within ~10 K of the converged deep temperature. Any positive value
    works; a good one only shortens spin-up.
    """
    if surface_mode == _SURFACE_FIXED:
        return float(np.mean(series))
    mean_in = float(np.mean(series)) + params.geothermal_flux
    T_rad = (mean_in / (params.emissivity * STEFAN_BOLTZMANN)) ** 0.25
    return max(0.85 * T_rad, 4.0 * _MIN_TEMPERATURE)


def _control_widths(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Thickness of the slab each node owns: halfway to each neighbour."""
    dz = np.diff(z)
    w = np.empty_like(z)
    w[0] = 0.5 * dz[0]
    w[-1] = 0.5 * dz[-1]
    w[1:-1] = 0.5 * (dz[:-1] + dz[1:])
    return w


def _solve(
    series: NDArray[np.float64],
    surface_mode: int,
    dt: float,
    params: RegolithParams | None,
    grid_settings: GridSettings | None,
    spinup: SpinupSettings | None,
    output_depths: ArrayLike | None,
) -> SolverResult:
    """Shared driver: build the grid, repeat cycles until converged, package results."""
    params = params if params is not None else RegolithParams()
    grid_settings = grid_settings if grid_settings is not None else GridSettings()
    spinup = spinup if spinup is not None else SpinupSettings()
    if not (dt > 0 and math.isfinite(dt)):
        raise ValueError("dt must be positive and finite")

    n_t = series.size
    period = n_t * dt
    z = make_grid(params, period, grid_settings)
    n_z = z.size

    depths = (
        np.zeros(0)
        if output_depths is None
        else np.atleast_1d(np.asarray(output_depths, dtype=np.float64))
    )
    if depths.ndim != 1:
        raise ValueError("output_depths must be a 1D sequence of depths")
    if np.any(depths < 0) or np.any(depths > z[-1]):
        raise ValueError(f"output_depths must lie within the grid, 0 to {z[-1]:.4f} m")

    # Per-node constants. Density never changes with time, so slab mass per
    # unit area (rho * w) is fixed; contact conductivity likewise.
    mass = density(z, params) * _control_widths(z)
    kc = contact_conductivity(z, params)
    dz = np.diff(z)
    cp = np.asarray(params.cp_coeffs, dtype=np.float64)
    chi_r = params.chi / RADIATIVE_REFERENCE_TEMPERATURE**3
    eps_sigma = params.emissivity * STEFAN_BOLTZMANN

    n_sub = max(1, math.ceil(dt / grid_settings.max_step - 1e-12))
    dt_sub = dt / n_sub

    T0 = spinup.initial_temperature
    if T0 is None:
        T0 = _initial_temperature(series, surface_mode, params)
    T = np.full(n_z, float(T0))
    if surface_mode == _SURFACE_FIXED:
        T[0] = series[0]
    T_prev = T.copy()
    have_prev = False

    record = np.empty((n_t, n_z))
    prev_record = np.empty((n_t, n_z))
    prev_end = T.copy()
    # `comparable`: prev_record/prev_end hold the cycle just before this one,
    # with no extrapolation in between, so a difference means something.
    comparable = False
    last_change: NDArray[np.float64] | None = None
    recent_ratios: list[float] = []
    # Decay ratio of the slow mode, once seen. Kept across extrapolations:
    # after a jump the next few changes are dominated by fast transients and
    # suggest quick convergence, while what is left of the slow mode still
    # shrinks by only this ratio per cycle.
    slow_ratio = 0.0
    residual = math.inf
    drift = math.inf
    converged = False
    extrapolations = 0
    n_cycles = 0

    for cycle in range(1, spinup.max_cycles + 1):
        status = _run_cycle(
            T,
            T_prev,
            have_prev,
            series,
            n_sub,
            dt_sub,
            mass,
            kc,
            dz,
            chi_r,
            cp,
            eps_sigma,
            params.geothermal_flux,
            surface_mode,
            record,
        )
        have_prev = True
        n_cycles = cycle
        if status != 0:
            raise RuntimeError(
                f"Newton iteration failed in cycle {cycle} (status {status}); "
                "the time step may be too large for this forcing"
            )

        if comparable:
            residual = max(
                float(np.max(np.abs(record - prev_record))),
                float(np.max(np.abs(T - prev_end))),
            )
            change = T - prev_end
            ratio = _decay_ratio(change, last_change)
            if ratio is not None:
                recent_ratios = (recent_ratios + [ratio])[-3:]
            ratio_is_stable = (
                len(recent_ratios) == 3
                and max(recent_ratios) - min(recent_ratios) < _RATIO_STABILITY
            )
            if ratio_is_stable and 0.0 < recent_ratios[-1] < 1.0:
                slow_ratio = recent_ratios[-1]
            drift = _projected_drift(
                change, None if ratio is None else max(abs(ratio), slow_ratio)
            )
            if (
                residual < spinup.tolerance
                and drift < spinup.tolerance
                and cycle >= spinup.min_cycles
            ):
                converged = True
                break
            if (
                spinup.accelerate
                and ratio_is_stable
                and 0.0 < recent_ratios[-1] <= _MAX_EXTRAPOLATION_RATIO
                and cycle < spinup.max_cycles
            ):
                # Jump to the end of the geometric drift (see SpinupSettings).
                # Both BDF2 history levels move together so the step that
                # follows sees no artificial jump in time.
                r = recent_ratios[-1]
                shift = change * (r / (1.0 - r))
                T += shift
                T_prev += shift
                extrapolations += 1
                comparable = False
                last_change = None
                recent_ratios = []
                prev_record, record = record, prev_record
                prev_end[:] = T
                continue
            last_change = change
        prev_record, record = record, prev_record
        prev_end[:] = T
        comparable = True

    if not converged:
        # Every unconverged cycle ends with a swap, so the final cycle's
        # states are now in prev_record.
        record = prev_record
        warnings.warn(
            f"Spin-up did not converge in {n_cycles} cycles: final cycle-to-cycle "
            f"change {residual:.3g} K, projected drift {drift:.3g} K, "
            f"tolerance {spinup.tolerance:.3g} K",
            RuntimeWarning,
            stacklevel=3,
        )

    node_T = record.copy()
    depth_T = None
    if depths.size:
        depth_T = np.empty((n_t, depths.size))
        for j, d in enumerate(depths):
            depth_T[:, j] = _interp_rows(z, node_T, d)

    return SolverResult(
        times=np.arange(n_t) * dt,
        surface_temperature=node_T[:, 0].copy(),
        depths=depths,
        depth_temperature=depth_T,
        grid=z,
        node_temperature=node_T,
        end_profile=T.copy(),
        n_cycles=n_cycles,
        converged=converged,
        residual=residual,
        drift_estimate=drift,
        tolerance=spinup.tolerance,
        substeps=n_sub,
        extrapolations=extrapolations,
    )


def _decay_ratio(
    change: NDArray[np.float64], last_change: NDArray[np.float64] | None
) -> float | None:
    """Per-cycle ratio of the dominant slow mode, from two successive changes.

    The least-squares ``r`` in ``change ~ r * last_change``. ``None`` if there
    is no previous change; 0 if the previous change was exactly zero.
    """
    if last_change is None:
        return None
    denom = float(last_change @ last_change)
    if denom == 0.0:
        return 0.0
    return float(change @ last_change) / denom


def _projected_drift(change: NDArray[np.float64], ratio: float | None) -> float:
    """Further change still to come if the decay continues geometrically, K.

    ``max|change| * |r| / (1 - |r|)``: the sum of ``|r|^k`` times the last
    change over all future cycles. Infinite when ``r`` is unknown or the
    change isn't shrinking.
    """
    if ratio is None:
        return math.inf
    a = abs(ratio)
    if a >= 1.0:
        return math.inf
    return float(np.max(np.abs(change))) * a / (1.0 - a)


def _interp_rows(
    z: NDArray[np.float64], values: NDArray[np.float64], depth: float
) -> NDArray[np.float64]:
    """Linear interpolation in depth of every row of ``values`` (shape (n_t, n_z))."""
    j = int(np.searchsorted(z, depth, side="right")) - 1
    j = min(max(j, 0), z.size - 2)
    w = (depth - z[j]) / (z[j + 1] - z[j])
    return (1.0 - w) * values[:, j] + w * values[:, j + 1]


# ---------------------------------------------------------------------------
# Compiled kernels
#
# Numba compiles these to machine code. They take plain arrays and scalars
# (Numba can't take the dataclasses). The property formulas below repeat those
# in properties.py in scalar form; tests/test_solver.py checks they agree.
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _cp_scalar(T, cp):
    """Heat capacity cp(T), Eq. A6 (Horner's rule)."""
    out = 0.0
    for i in range(cp.size - 1, -1, -1):
        out = out * T + cp[i]
    return out


@numba.njit(cache=True)
def _h_scalar(T, cp):
    """Specific enthalpy h(T), the exact integral of cp from 0 to T."""
    out = 0.0
    for i in range(cp.size - 1, -1, -1):
        out = out * T + cp[i] / (i + 1)
    return out * T


@numba.njit(cache=True)
def _run_cycle(
    T,
    T_prev,
    have_prev,
    series,
    n_sub,
    dt_sub,
    mass,
    kc,
    dz,
    chi_r,
    cp,
    eps_sigma,
    q_geo,
    surface_mode,
    record,
):
    """Advance the column through one period, in place.

    ``T`` holds the current node temperatures and ``T_prev`` those one
    internal step earlier (needed by BDF2); both are updated in place.
    ``record[k]`` receives the state at sample time ``k * dt``: row 0 is the
    state on entry, row ``k`` the state after ``k`` forcing intervals. On
    return ``T`` is the state at the end of the period.

    Returns 0 on success, or a positive code if a Newton solve failed.
    """
    n_t = series.size
    n_z = T.size
    h_now = np.empty(n_z)
    h_old = np.empty(n_z)
    hist = np.empty(n_z)
    T_new = np.empty(n_z)
    res = np.empty(n_z)
    lower = np.empty(n_z)
    diag = np.empty(n_z)
    upper = np.empty(n_z)
    delta = np.empty(n_z)
    K = np.empty(n_z)
    dK = np.empty(n_z)
    c_work = np.empty(n_z)
    d_work = np.empty(n_z)

    record[0, :] = T
    use_bdf2 = have_prev
    for k in range(n_t):
        f0 = series[k]
        f1 = series[(k + 1) % n_t]
        for s in range(n_sub):
            # Forcing at the end of this substep (implicit methods evaluate
            # sources at the new time level).
            frac = (s + 1.0) / n_sub
            forcing = f0 + (f1 - f0) * frac

            # Time-discretization coefficients. BDF2:
            #   (3 h^{n+1} - 4 h^n + h^{n-1}) / (2 dt) = rate
            # written as (a0 h^{n+1} - hist) / dt with a0 = 3/2,
            # hist = 2 h^n - h^{n-1} / 2. Backward Euler (first step ever,
            # when there is no h^{n-1}): a0 = 1, hist = h^n.
            for i in range(n_z):
                h_now[i] = _h_scalar(T[i], cp)
            if use_bdf2:
                a0 = 1.5
                for i in range(n_z):
                    h_old[i] = _h_scalar(T_prev[i], cp)
                    hist[i] = 2.0 * h_now[i] - 0.5 * h_old[i]
                # Initial guess: linear extrapolation from the last two
                # states, which is close to the answer for smooth forcing.
                for i in range(n_z):
                    T_new[i] = 2.0 * T[i] - T_prev[i]
                    if T_new[i] < 0.5 * T[i]:
                        T_new[i] = T[i]
            else:
                a0 = 1.0
                for i in range(n_z):
                    hist[i] = h_now[i]
                    T_new[i] = T[i]
            if surface_mode == _SURFACE_FIXED:
                T_new[0] = forcing

            converged = False
            for _it in range(_NEWTON_MAX_ITER):
                # Conductivity and its temperature derivative at each node
                # (Eq. A4): K = Kc (1 + chi (T/350)^3).
                for i in range(n_z):
                    t = T_new[i]
                    K[i] = kc[i] * (1.0 + chi_r * t * t * t)
                    dK[i] = kc[i] * 3.0 * chi_r * t * t

                # Residual r_i = mass_i (a0 h(T_i) - hist_i) / dt - (net heat
                # into slab i), and its tridiagonal Jacobian. Interface flux
                # G_{i+1/2} = Kbar (T_{i+1} - T_i) / dz_i is the heat flowing
                # up from node i+1 into node i.
                for i in range(n_z):
                    store = mass[i] / dt_sub
                    res[i] = store * (a0 * _h_scalar(T_new[i], cp) - hist[i])
                    diag[i] = store * a0 * _cp_scalar(T_new[i], cp)
                    lower[i] = 0.0
                    upper[i] = 0.0
                for i in range(n_z - 1):
                    kbar = 0.5 * (K[i] + K[i + 1])
                    grad = (T_new[i + 1] - T_new[i]) / dz[i]
                    G = kbar * grad
                    dG_dTi = 0.5 * dK[i] * grad - kbar / dz[i]
                    dG_dTj = 0.5 * dK[i + 1] * grad + kbar / dz[i]
                    # G enters slab i and leaves slab i+1.
                    res[i] -= G
                    diag[i] -= dG_dTi
                    upper[i] -= dG_dTj
                    res[i + 1] += G
                    diag[i + 1] += dG_dTj
                    lower[i + 1] += dG_dTi
                # Bottom: geothermal flux enters the lowest slab (Eq. A12).
                res[n_z - 1] -= q_geo
                if surface_mode == _SURFACE_RADIATIVE:
                    # Surface: absorbed flux in, thermal emission out (Eq. A7).
                    t0 = T_new[0]
                    res[0] -= forcing - eps_sigma * t0 * t0 * t0 * t0
                    diag[0] += 4.0 * eps_sigma * t0 * t0 * t0
                else:
                    # Fixed surface temperature: replace row 0 by T_0 = forcing.
                    res[0] = T_new[0] - forcing
                    diag[0] = 1.0
                    upper[0] = 0.0

                # Solve J delta = -res with the Thomas algorithm (tridiagonal
                # Gaussian elimination without pivoting; J is diagonally
                # dominant here, so pivoting isn't needed).
                c_work[0] = upper[0] / diag[0]
                d_work[0] = -res[0] / diag[0]
                for i in range(1, n_z):
                    m = diag[i] - lower[i] * c_work[i - 1]
                    c_work[i] = upper[i] / m
                    d_work[i] = (-res[i] - lower[i] * d_work[i - 1]) / m
                delta[n_z - 1] = d_work[n_z - 1]
                for i in range(n_z - 2, -1, -1):
                    delta[i] = d_work[i] - c_work[i] * delta[i + 1]

                largest_update = 0.0
                for i in range(n_z):
                    # Damp steps that would more than halve a temperature, so
                    # the iteration can't jump to the unphysical T < 0 branch.
                    d = delta[i]
                    if T_new[i] + d < 0.5 * T_new[i]:
                        d = -0.5 * T_new[i]
                    T_new[i] += d
                    largest_update = max(largest_update, abs(d))
                if largest_update < _NEWTON_TOL:
                    converged = True
                    break

            if not converged:
                return 1
            for i in range(n_z):
                if T_new[i] < _MIN_TEMPERATURE or not np.isfinite(T_new[i]):
                    return 2

            for i in range(n_z):
                T_prev[i] = T[i]
                T[i] = T_new[i]
            use_bdf2 = True

        if k + 1 < n_t:
            record[k + 1, :] = T
    return 0
