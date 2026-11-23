"""Triangulate this repo's regolith solver against heat1d, and check it against Diviner.

    uv run python scripts/triangulate_heat1d.py [--out PATH] [--figures DIR] [--workers N] [--years Y]

Two independent implementations of the same physics (Hayne et al. 2017, Appendix A) are run on the
same absorbed flux and the same regolith parameters:

- this repo's solver, `lunar_thermal.solver.solve_periodic` (finite volumes, BDF2 on enthalpy,
  accelerated spin-up);
- `heat1d` (github.com/phayne/heat1d, MIT; Hayne's own model, pinned in `pyproject.toml`), with its
  explicit solver (the scheme of the paper's Appendix A2), at its default grid and at a refined grid.

**Forcing.** Both solvers get one forcing: heat1d's own absorbed-flux series
(`heat1d.fourier_solver.precompute_diurnal_flux`) for a flat surface at the requested latitude, with the
Moon on a circular orbit at a fixed distance and zero declination (equinox), Bond albedo from Eq. A8.
heat1d's default Moon instead follows Earth's eccentric orbit with local noon at perihelion (0.983 AU)
and a declination that drifts by up to 1.5 degrees; that is not periodic over one lunation and is not
the paper's standard case, so this script circularises the orbit (`circular_moon`). The series is
checked against an independent evaluation of Eqs. A8-A9 (`paper_flux`) before anything runs.

heat1d receives the series through its `flux_series` argument. Its time-stepping solvers always
equilibrate on their built-in insolation, which for the circular orbit is the same function, then run
the output cycles on the supplied series (linearly interpolated between samples, as `solve_periodic`
does). heat1d is equilibrated with its own explicit solver for `--years` years (the paper's procedure,
Appendix A2.4) rather than its default Fourier-matrix shortcut, which leaves the explicit solver's deep
column several kelvin from its own steady state. The last output cycle is compared, and its
cycle-to-cycle change and projected remaining drift are recorded.

**Cases** (`CASES`): flat surface at 0, 30, 60 and 75 degrees with default parameters; at the equator,
H = 0.03 and 0.10 m and A0 = 0.08 and 0.16. Each case runs this solver at its default and a refined grid
(its own discretization error) and heat1d at its default (m = 10, n = 4) and a refined grid (m = 40,
n = 20). The equatorial default case also runs heat1d at m = 20, n = 10, and once with a diagnostic patch
(`midpoint_conductivity`) that replaces heat1d's per-interface conductivity with the mean of the two
nodes, to test what causes the disagreement at heat1d's default grid. The patch is applied in a worker
process only, and only to that run; heat1d itself is not modified.

**Diviner check** (`diviner_check`): this solver's equatorial curve against the Diviner-derived values
quoted in Hayne et al. (2017) Table A2 (from Vasavada et al. 2012; stated for A0 = 0.12, so a
constant-albedo run is included), against the nighttime Diviner points of Hayne et al. Figure A2
(digitized, approximate; `FIG_A2_DIVINER`), and the diurnal means at 0, 20 and 26 degrees against the
model curves of Figure A1 (digitized, approximate; `FIG_A1_MODEL`) and the Apollo 17 and 15 heat-flow
measurements of Table A2. Those run with the Moon at 1.017 AU as the Figure A1 caption states, at 1 AU,
and with the mare albedo heat1d's own validation suite uses for the Apollo sites (`APOLLO_SETTINGS`).

Outputs: one `.npz` results file outside the repo (default
`$HOME/data/lunar-thermal-surrogate/derived/triangulation_heat1d.npz`; a path inside the repo is
refused), PNG figures in `docs/figures/`, and the results tables as Markdown on stdout.

Nothing here reads or evaluates the 23 held-out sites: every case is a flat surface at a latitude.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lunar_thermal.properties import RegolithParams
from lunar_thermal.solver import (
    GridSettings,
    SolverResult,
    SpinupSettings,
    solve_periodic,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = (
    Path.home()
    / "data"
    / "lunar-thermal-surrogate"
    / "derived"
    / "triangulation_heat1d.npz"
)
DEFAULT_FIGURES = REPO_ROOT / "docs" / "figures"

#: heat1d's pinned commit (must match `pyproject.toml`), recorded in the results file.
HEAT1D_SOURCE = (
    "github.com/phayne/heat1d@5b73a6c1ba5047f59f6210185b37539e46e68136 (python/)"
)

#: Samples per lunation: about hourly (dt = 3603.7 s for heat1d's 29.53059 d synodic month).
N_SAMPLES = 708

#: Depths compared below the surface, m.
DEPTHS = (0.10, 0.50)

#: Solar constant at 1 AU, W m^-2 (Table A1; also heat1d's `Moon.S`).
SOLAR_CONSTANT = 1361.0

#: Eq. A8 angle-dependence constants a, b (Table A1; heat1d's `Moon.albedoCoef`).
ALBEDO_COEFFS = (0.06, 0.25)

#: heat1d grids: (m, n, b). m layers per surface skin depth, growth 1 + 1/n, bottom at b skin depths
#: (heat1d's skin depth uses the surface conductivity and density with cp = 600 J/kg/K).
HEAT1D_GRIDS = {
    "default": (10, 4, 20),
    "mid": (20, 10, 20),
    "refined": (40, 20, 20),
}

#: This solver's refined grid, for its own discretization error (the defaults are m = 20, n = 10).
OURS_REFINED = GridSettings(layers_per_skin_depth=40.0, growth=20.0, max_step=900.0)


@dataclass(frozen=True)
class Case:
    """One flat-surface comparison case."""

    name: str
    lat: float  # degrees
    H: float = 0.06  # m
    A0: float = 0.12  # Bond albedo at normal incidence (Eq. A8)

    @property
    def label(self) -> str:
        return f"{self.lat:g}°, H = {self.H:g} m, A0 = {self.A0:g}"


CASES = (
    Case("lat00", 0.0),
    Case("lat30", 30.0),
    Case("lat60", 60.0),
    Case("lat75", 75.0),
    Case("eq_H0.03", 0.0, H=0.03),
    Case("eq_H0.10", 0.0, H=0.10),
    Case("eq_A0.08", 0.0, A0=0.08),
    Case("eq_A0.16", 0.0, A0=0.16),
)

# Diviner-derived constraints at the equator, Hayne et al. (2017) Table A2 (from Vasavada et al. 2012),
# stated for A0 = 0.12. Values in K.
TABLE_A2_EQUATOR = {"noon_max": 385.0, "midnight": 101.0, "predawn_min": 95.0}

# Apollo heat-flow sites, Hayne et al. (2017) Table A2: diurnal mean temperatures (K, +/- 5 K) at the
# surface and at depth. Figure A1 caption: Table A1 parameters, Moon at 1.017 AU.
APOLLO = (
    # name, latitude (deg N), surface mean, depth (m), mean at depth
    ("Apollo 17", 20.0, 216.0, 1.30, 256.0),
    ("Apollo 15", 26.0, 211.0, 0.83, 252.0),
)
APOLLO_DISTANCE_AU = 1.017

# The model curves of Hayne et al. (2017) Figure A1 ("Model: Surface" and "Model: 1 m Depth", diurnal
# mean against latitude), read off the figure (approximate). The curves are vector Bezier paths; each
# was sampled along its segments and averaged within +/- 0.6 deg of the latitude, so the reading error
# is about the stroke width, under 0.5 K. Latitude (deg): (surface, 1 m), K.
FIG_A1_MODEL = {0.0: (217.5, 261.2), 20.0: (213.3, 255.3), 26.0: (210.3, 251.1)}

# Forcing settings for the Apollo / Figure A1 comparison: (label, A0, (a, b), distance in AU).
# The first is the Figure A1 caption (Table A1 parameters, 1.017 AU). The last is what heat1d's own
# validation suite uses for the Apollo sites (`heat1d.validation`: mare albedo 0.06 with a and b scaled
# by the same factor, at 1 AU); it is included to show how far the inputs alone move the result.
APOLLO_SETTINGS = (
    (
        "Table A1, A0 = 0.12, 1.017 AU (Fig. A1 caption)",
        0.12,
        ALBEDO_COEFFS,
        APOLLO_DISTANCE_AU,
    ),
    ("Table A1, A0 = 0.12, 1 AU", 0.12, ALBEDO_COEFFS, 1.0),
    (
        "A0 = 0.06, a, b halved, 1 AU (heat1d's Apollo setting)",
        0.06,
        (0.03, 0.125),
        1.0,
    ),
)

# Nighttime Diviner regolith temperatures read off Hayne et al. (2017) Figure A2 (approximate: values
# read off a figure). The published PDF draws the figure as vectors; each marker's centre was taken from
# its path and converted with the axis tick marks (tick residuals < 0.001 h and < 0.001 K), so the
# reading error is far below the figure's own plotting precision. Local time in hours (24 h clock), K.
# The caption describes the data as equatorial (+/- 1 deg), selected from the 128 ppd gridded data set;
# the legend labels the three series 0, 30 and 60 degrees latitude.
FIG_A2_DIVINER = {
    0.0: (
        (20.47, 107.5), (21.47, 104.6), (22.47, 102.4), (23.47, 100.4), (0.47, 99.0),
        (1.47, 97.7), (2.47, 96.4), (3.47, 95.2), (4.47, 94.3),
    ),
    30.0: (
        (20.50, 105.2), (21.50, 102.2), (22.50, 100.0), (23.50, 98.1), (0.50, 96.8),
        (1.50, 95.4), (2.50, 94.0), (3.50, 92.8), (4.50, 91.9),
    ),
    60.0: (
        (20.50, 94.9), (21.50, 92.5), (22.50, 90.1), (23.50, 88.7), (0.50, 87.3),
        (1.50, 86.2), (2.50, 84.6), (3.50, 83.4), (4.50, 82.6),
    ),
}  # fmt: skip


# ---------------------------------------------------------------------------------------------------
# Forcing
# ---------------------------------------------------------------------------------------------------


def circular_moon(
    H: float = 0.06, A0: float = 0.12, distance_au: float = 1.0, coeffs=ALBEDO_COEFFS
):
    """heat1d's Moon with a circular orbit, zero obliquity and the given parameters.

    heat1d applies `S` at the body's semi-major axis (`rAU`, 1.0000142 AU for the Moon) and scales by
    (rAU / r)^2; with zero eccentricity r = rAU always, so the absorbed flux is `S (1 - A) cos(i)` and
    setting `S = 1361 / d^2` puts the Moon at d AU. Zero obliquity gives zero declination (equinox),
    and the hour angle advances uniformly over the synodic day. `coeffs` are Eq. A8's (a, b); (0, 0)
    gives a constant albedo A0.
    """
    from heat1d import planets

    moon = copy.copy(planets.Moon)
    moon.eccentricity = 0.0
    moon.obliquity = 0.0
    moon.S = SOLAR_CONSTANT / distance_au**2
    moon.H = H
    moon.albedo = A0
    moon.albedoCoef = [float(c) for c in coeffs]
    return moon


def absorbed_flux(
    lat: float, A0: float = 0.12, distance_au: float = 1.0, coeffs=ALBEDO_COEFFS
) -> tuple[np.ndarray, float]:
    """heat1d's absorbed flux for a flat surface at `lat` degrees: (flux in W m^-2, dt in s).

    Sample k is at time k * dt after local noon, over one synodic day (heat1d's 29.53059 d).
    """
    from heat1d.fourier_solver import precompute_diurnal_flux

    moon = circular_moon(A0=A0, distance_au=distance_au, coeffs=coeffs)
    flux, dt = precompute_diurnal_flux(moon, np.deg2rad(lat), N_SAMPLES)
    return np.asarray(flux, dtype=np.float64), float(dt)


def paper_flux(
    lat: float, A0: float = 0.12, distance_au: float = 1.0, coeffs=ALBEDO_COEFFS
) -> np.ndarray:
    """Independent evaluation of Hayne et al. Eqs. A8-A11 at zero declination, for checking.

    cos(theta) = cos(lat) cos(h), h = 2 pi k / N; absorbed = (S / d^2)(1 - A(theta)) psi(theta), with
    A(theta) = A0 + a (theta / (pi/4))^3 + b (theta / (pi/2))^8 and psi the clipped cosine.
    """
    h = 2.0 * np.pi * np.arange(N_SAMPLES) / N_SAMPLES
    cos_t = np.clip(np.cos(np.deg2rad(lat)) * np.cos(h), 0.0, 1.0)
    theta = np.arccos(cos_t)
    a, b = coeffs
    albedo = A0 + a * (theta / (np.pi / 4)) ** 3 + b * (theta / (np.pi / 2)) ** 8
    return SOLAR_CONSTANT / distance_au**2 * (1.0 - albedo) * cos_t


def check_flux_matches_paper(lat: float, **kw) -> float:
    """Largest absolute difference between heat1d's flux and `paper_flux`, W m^-2 (raises if > 1e-6, i.e. beyond rounding)."""
    flux, _ = absorbed_flux(lat, **kw)
    diff = float(np.max(np.abs(flux - paper_flux(lat, **kw))))
    if diff > 1e-6:
        raise AssertionError(
            f"heat1d flux differs from Eqs. A8-A9 by {diff} W m^-2 at {lat} deg"
        )
    return diff


# ---------------------------------------------------------------------------------------------------
# heat1d runs (in worker processes)
# ---------------------------------------------------------------------------------------------------


def _solve_explicit_midpoint(T, dt, rho, cp, k, g1, g2):
    """heat1d's explicit update (Eq. A17) with each interface conductivity the mean of its two nodes.

    DIAGNOSTIC ONLY. heat1d's `solve_explicit` uses `alpha_i = g1_i k_{i-1}` and `beta_i = g2_i k_i`:
    the flux between nodes i and i+1 uses the upper node's conductivity k_i. This copy uses
    (k_i + k_{i+1}) / 2 instead, which is what this repo's solver does. Everything else is unchanged.
    """
    k_mid = 0.5 * (k[:-1] + k[1:])
    alpha = g1 * k_mid[0:-1]
    beta = g2 * k_mid[1:]
    T[1:-1] = T[1:-1] + dt / (rho[1:-1] * cp[1:-1]) * (
        alpha * T[0:-2] - (alpha + beta) * T[1:-1] + beta * T[2:]
    )


def run_heat1d(spec: dict) -> dict:
    """Run heat1d's explicit solver for one case to its periodic state; return its last cycle.

    `spec` keys: lat, H, A0, grid (a `HEAT1D_GRIDS` key), years, out_cycles, patched.
    Returns the surface curve and node temperatures at the forcing sample times, the node depths,
    the last cycle-to-cycle change, the projected remaining drift and the wall time.
    """
    import heat1d.profile as hprofile
    from heat1d.config import Configurator
    from heat1d.model import Model

    m, n, b = HEAT1D_GRIDS[spec["grid"]]
    flux, dt = absorbed_flux(spec["lat"], A0=spec["A0"])
    moon = circular_moon(H=spec["H"], A0=spec["A0"])
    cycles = spec["out_cycles"]
    config = Configurator(
        solver="explicit",
        equil_solver="explicit",
        NYEARSEQ=spec["years"],
        m=m,
        n=n,
        b=b,
        output_interval=dt,
    )
    # One extra sample so heat1d's interpolation never clamps at the end of the series.
    series = np.concatenate([np.tile(flux, cycles), flux[:1]])

    original = hprofile.solve_explicit
    if spec["patched"]:
        hprofile.solve_explicit = _solve_explicit_midpoint
    try:
        t0 = time.perf_counter()
        model = Model(
            planet=moon,
            lat=np.deg2rad(spec["lat"]),
            ndays=cycles,
            config=config,
            flux_series=series,
            flux_dt=dt,
        )
        model.run()
        seconds = time.perf_counter() - t0
    finally:
        hprofile.solve_explicit = original

    T = np.asarray(model.T)
    if T.shape[0] != cycles * N_SAMPLES:
        raise RuntimeError(
            f"heat1d returned {T.shape[0]} rows, expected {cycles * N_SAMPLES}"
        )
    per_cycle = T.reshape(cycles, N_SAMPLES, -1)
    changes = [
        float(np.max(np.abs(per_cycle[i] - per_cycle[i - 1]))) for i in range(1, cycles)
    ]
    # Projected drift as in this repo's spin-up check: change * r / (1 - r), r from the last two changes.
    drift = float("inf")
    if len(changes) >= 2 and changes[-2] > 0:
        r = changes[-1] / changes[-2]
        if r < 1:
            drift = changes[-1] * r / (1 - r)
    # heat1d stores row i at t = (i + 1) dt; roll so row k is the state at sample time k dt, as in
    # SolverResult.
    last = np.roll(per_cycle[-1], 1, axis=0)
    year, day = moon.year, moon.day
    lunations = (
        int(spec["years"] * year // day) + 1 + cycles
    )  # equilibration + warm-up + output
    return {
        "spec": spec,
        "surface": last[:, 0].copy(),
        "nodes": last,
        "z": np.asarray(model.profile.z, dtype=np.float64),
        "changes": changes,
        "drift": drift,
        "seconds": seconds,
        "lunations": lunations,
    }


def heat1d_depth(run: dict, depth: float) -> np.ndarray:
    """heat1d temperature at `depth` m over the cycle, linear between its nodes (as this repo does)."""
    return np.array([np.interp(depth, run["z"], row) for row in run["nodes"]])


# ---------------------------------------------------------------------------------------------------
# This repo's solver
# ---------------------------------------------------------------------------------------------------


def run_ours(
    flux: np.ndarray,
    dt: float,
    params: RegolithParams,
    grid: GridSettings | None = None,
    depths=DEPTHS,
    spinup: SpinupSettings | None = None,
) -> tuple[SolverResult, float]:
    """`solve_periodic` (spin-up defaults unless given); returns (result, wall seconds).

    Raises if spin-up did not converge, so no unconverged curve enters a comparison.
    """
    t0 = time.perf_counter()
    result = solve_periodic(
        flux, dt, params=params, grid=grid, spinup=spinup, output_depths=list(depths)
    )
    seconds = time.perf_counter() - t0
    if not result.converged:
        raise RuntimeError(
            f"solve_periodic did not converge (residual {result.residual} K, "
            f"drift {result.drift_estimate} K)"
        )
    return result, seconds


# ---------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------


def curve_metrics(surface: np.ndarray, flux: np.ndarray) -> dict:
    """Noon maximum, pre-dawn minimum and midnight temperature of one surface curve, K.

    Noon maximum: the cycle's highest temperature (at or just after local noon). Pre-dawn minimum: the
    lowest temperature over the night (flux zero), which falls just before sunrise. Midnight: the
    sample at half the period (sample N/2 is exactly local midnight).
    """
    night = flux <= 0.0
    return {
        "noon_max": float(surface.max()),
        "predawn_min": float(surface[night].min())
        if night.any()
        else float(surface.min()),
        "midnight": float(surface[N_SAMPLES // 2]),
        "mean": float(surface.mean()),
    }


def difference(a: np.ndarray, b: np.ndarray) -> dict:
    """RMSE and largest absolute difference between two curves on the same sample times, K."""
    d = np.asarray(a) - np.asarray(b)
    return {"rmse": float(np.sqrt(np.mean(d**2))), "max_abs": float(np.max(np.abs(d)))}


def hours_after_noon() -> np.ndarray:
    """Sample times in hours after local noon (0 .. 24)."""
    return 24.0 * np.arange(N_SAMPLES) / N_SAMPLES


def at_local_time(surface: np.ndarray, local_hours) -> np.ndarray:
    """Curve value at local times on the 24 h clock (12 = noon), periodic linear interpolation."""
    x = (np.asarray(local_hours, dtype=np.float64) - 12.0) % 24.0
    return np.interp(x, hours_after_noon(), surface, period=24.0)


# ---------------------------------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------------------------------


def run_triangulation(workers: int, years: int, out_cycles: int) -> dict:
    """Run every case through both solvers. Returns a nested dict of curves and metrics."""
    # Before anything else: heat1d's insolation is Eqs. A8-A9 at zero declination.
    flux_check = max(
        check_flux_matches_paper(c.lat, A0=c.A0) for c in CASES
    )  # largest |difference| in W m^-2

    specs = []
    for case in CASES:
        for grid in ("default", "refined"):
            specs.append({"case": case.name, "grid": grid, "patched": False})
    specs.append({"case": "lat00", "grid": "mid", "patched": False})
    specs.append({"case": "lat00", "grid": "default", "patched": True})
    by_name = {c.name: c for c in CASES}
    # Finer heat1d grids relax more slowly per simulated year at high latitude (75 deg, m = 40 still
    # changed by 0.08 K per cycle after 10 years), so the refined grid gets twice the equilibration.
    full_specs = [
        {
            **s,
            "lat": by_name[s["case"]].lat,
            "H": by_name[s["case"]].H,
            "A0": by_name[s["case"]].A0,
            "years": 2 * years if s["grid"] == "refined" else years,
            "out_cycles": out_cycles,
        }
        for s in specs
    ]
    # Compile this solver's kernels once before timing anything.
    warm_flux, warm_dt = absorbed_flux(0.0)
    run_ours(warm_flux, warm_dt, RegolithParams())

    # heat1d runs are slow (Python loop per explicit step), so they go to worker processes while this
    # process runs the fast solver.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_heat1d, s) for s in full_specs]

        ours = {}
        for case in CASES:
            flux, dt = absorbed_flux(case.lat, A0=case.A0)
            params = RegolithParams(H=case.H)
            default, sec_default = run_ours(flux, dt, params)
            refined, sec_refined = run_ours(flux, dt, params, grid=OURS_REFINED)
            ours[case.name] = {
                "flux": flux,
                "dt": dt,
                "default": default,
                "refined": refined,
                "seconds": sec_default,
                "seconds_refined": sec_refined,
            }
            print(
                f"  this solver: {case.name} done ({sec_default:.2f} s)",
                file=sys.stderr,
            )

        heat1d = {}
        for spec, fut in zip(full_specs, futures, strict=True):
            run = fut.result()
            key = spec["grid"] + ("_patched" if spec["patched"] else "")
            heat1d.setdefault(spec["case"], {})[key] = run
            print(
                f"  heat1d: {spec['case']} {key} done ({run['seconds']:.1f} s, "
                f"last change {run['changes'][-1]:.4f} K)",
                file=sys.stderr,
            )

    return {"ours": ours, "heat1d": heat1d, "flux_check": flux_check}


def summarise(tri: dict) -> list[dict]:
    """Per-case metrics: each solver's curve metrics, pairwise differences, depth temperatures."""
    rows = []
    for case in CASES:
        o = tri["ours"][case.name]
        h = tri["heat1d"][case.name]
        flux = o["flux"]
        row = {
            "case": case.name,
            "label": case.label,
            "lat": case.lat,
            "H": case.H,
            "A0": case.A0,
        }
        row["ours"] = curve_metrics(o["default"].surface_temperature, flux)
        row["ours_refined"] = curve_metrics(o["refined"].surface_temperature, flux)
        for key, run in h.items():
            row[f"heat1d_{key}"] = curve_metrics(run["surface"], flux)
            row[f"diff_{key}"] = difference(
                run["surface"], o["default"].surface_temperature
            )
            row[f"heat1d_{key}_convergence"] = {
                "last_change": run["changes"][-1],
                "drift": run["drift"],
                "nodes": len(run["z"]),
                "bottom": float(run["z"][-1]),
                "seconds": run["seconds"],
                "lunations": run["lunations"],
            }
        row["diff_ours_refined"] = difference(
            o["refined"].surface_temperature, o["default"].surface_temperature
        )
        row["diff_heat1d_refined_default"] = difference(
            h["refined"]["surface"], h["default"]["surface"]
        )
        row["ours_convergence"] = {
            "cycles": o["default"].n_cycles,
            "residual": o["default"].residual,
            "drift": o["default"].drift_estimate,
            "nodes": len(o["default"].grid),
            "bottom": float(o["default"].grid[-1]),
            "seconds": o["seconds"],
        }
        depth_rows = {}
        for j, z in enumerate(DEPTHS):
            ours_z = o["default"].depth_temperature[:, j]
            ours_ref_z = o["refined"].depth_temperature[:, j]
            entry = {
                "ours_mean": float(ours_z.mean()),
                "ours_refined_mean": float(ours_ref_z.mean()),
            }
            for key, run in h.items():
                hz = heat1d_depth(run, z)
                entry[f"heat1d_{key}_mean"] = float(hz.mean())
                entry[f"heat1d_{key}_max_abs"] = float(np.max(np.abs(hz - ours_z)))
            depth_rows[f"{z:.2f}"] = entry
        row["depths"] = depth_rows
        rows.append(row)
    return rows


def diviner_check(tri: dict) -> dict:
    """This solver against Table A2 and the Figure A2 Diviner points (and the Apollo diurnal means)."""
    out: dict = {"equator": [], "fig_a2": [], "apollo": []}

    # Equatorial curve, 1 AU: Eq. A8 albedo at several H, and the constant-albedo run Table A2 states.
    equator_runs = [
        ("Eq. A8 albedo, H = 0.06 m (default)", 0.06, True),
        ("constant A0 = 0.12, H = 0.06 m", 0.06, False),
        ("Eq. A8 albedo, H = 0.03 m", 0.03, True),
        ("Eq. A8 albedo, H = 0.068 m (paper's global mean)", 0.068, True),
        ("Eq. A8 albedo, H = 0.07 m (heat1d's default)", 0.07, True),
        ("Eq. A8 albedo, H = 0.10 m", 0.10, True),
    ]
    curves = {}
    for label, H, angle in equator_runs:
        flux, dt = absorbed_flux(0.0, coeffs=ALBEDO_COEFFS if angle else (0.0, 0.0))
        res, _ = run_ours(flux, dt, RegolithParams(H=H))
        metrics = curve_metrics(res.surface_temperature, flux)
        lt = np.array([p[0] for p in FIG_A2_DIVINER[0.0]])
        Td = np.array([p[1] for p in FIG_A2_DIVINER[0.0]])
        resid = at_local_time(res.surface_temperature, lt) - Td
        out["equator"].append(
            {
                "label": label,
                "H": H,
                "angle_albedo": angle,
                **metrics,
                "fig_a2_mean_resid": float(resid.mean()),
                "fig_a2_rmse": float(np.sqrt(np.mean(resid**2))),
            }
        )
        curves[label] = res.surface_temperature

    # Figure A2 points at 0, 30, 60 degrees against the default-parameter triangulation runs.
    for lat, pts in FIG_A2_DIVINER.items():
        name = f"lat{int(lat):02d}"
        surface = tri["ours"][name]["default"].surface_temperature
        lt = np.array([p[0] for p in pts])
        Td = np.array([p[1] for p in pts])
        model = at_local_time(surface, lt)
        out["fig_a2"].append(
            {
                "lat": lat,
                "local_time": lt.tolist(),
                "diviner": Td.tolist(),
                "model": model.tolist(),
                "mean_resid": float((model - Td).mean()),
                "max_abs_resid": float(np.max(np.abs(model - Td))),
            }
        )

    # Diurnal means at 0, 20 and 26 deg (Figure A1 model; Apollo 17 and 15 measurements, Table A2)
    # under each forcing setting in APOLLO_SETTINGS, with Table A1 regolith parameters. The column is
    # deepened to 20 deep skin depths (about 1.5 m) so 1.30 m lies inside it. The deeper column relaxes
    # more slowly than the 0.98-per-cycle ratio the solver will extrapolate along, so spin-up runs
    # unaccelerated and needs more than the default 200 cycles.
    observed = {
        lat: (name, T_surf, depth, T_depth)
        for name, lat, T_surf, depth, T_depth in APOLLO
    }
    for label, A0, coeffs, distance in APOLLO_SETTINGS:
        for lat, (fig_surface, fig_1m) in FIG_A1_MODEL.items():
            flux, dt = absorbed_flux(lat, A0=A0, distance_au=distance, coeffs=coeffs)
            res, _ = run_ours(
                flux,
                dt,
                RegolithParams(),
                grid=GridSettings(bottom_skin_depths=20.0),
                depths=(0.83, 1.0, 1.30),
                spinup=SpinupSettings(max_cycles=2000),
            )
            means = res.depth_temperature.mean(axis=0)
            entry = {
                "setting": label,
                "lat": lat,
                "A0": A0,
                "coeffs": list(coeffs),
                "distance_au": distance,
                "surface_mean": float(res.surface_temperature.mean()),
                "mean_0.83m": float(means[0]),
                "mean_1m": float(means[1]),
                "mean_1.30m": float(means[2]),
                "fig_a1_surface": fig_surface,
                "fig_a1_1m": fig_1m,
                "bottom": float(res.grid[-1]),
                "cycles": res.n_cycles,
            }
            if lat in observed:
                site, T_surf, depth, T_depth = observed[lat]
                entry.update(
                    site=site,
                    surface_obs=T_surf,
                    depth=depth,
                    depth_obs=T_depth,
                    depth_mean=float(means[(0.83, 1.0, 1.30).index(depth)]),
                )
            out["apollo"].append(entry)
    out["curves"] = curves
    return out


# ---------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------


def markdown_tables(rows: list[dict], div: dict) -> str:
    """The results tables as Markdown, for docs/methodology.md."""
    lines = []
    lines.append(
        "**Surface temperature, K** (this solver at its defaults; heat1d at its default grid"
    )
    lines.append(
        "m = 10, n = 4 and refined m = 40, n = 20; differences heat1d minus this solver)\n"
    )
    lines.append(
        "| Case | This solver: noon / pre-dawn / midnight | heat1d default | heat1d refined "
        "| RMSE, max: default | RMSE, max: refined |"
    )
    lines.append("|---|---|---|---|---|---|")

    def trio(m):
        return f"{m['noon_max']:.1f} / {m['predawn_min']:.1f} / {m['midnight']:.1f}"

    for r in rows:
        d0, d1 = r["diff_default"], r["diff_refined"]
        lines.append(
            f"| {r['label']} | {trio(r['ours'])} | {trio(r['heat1d_default'])} "
            f"| {trio(r['heat1d_refined'])} | {d0['rmse']:.2f}, {d0['max_abs']:.2f} "
            f"| {d1['rmse']:.2f}, {d1['max_abs']:.2f} |"
        )

    lines.append(
        "\n**Diurnal mean temperature at depth, K** (in brackets: largest difference from"
    )
    lines.append("this solver over the cycle)\n")
    lines.append(
        "| Case | 10 cm: this solver | heat1d default | heat1d refined "
        "| 50 cm: this solver | heat1d default | heat1d refined |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        cells = []
        for z in ("0.10", "0.50"):
            e = r["depths"][z]
            cells += [
                f"{e['ours_mean']:.1f}",
                f"{e['heat1d_default_mean']:.1f} ({e['heat1d_default_max_abs']:.1f})",
                f"{e['heat1d_refined_mean']:.1f} ({e['heat1d_refined_max_abs']:.1f})",
            ]
        lines.append(f"| {r['label']} | " + " | ".join(cells) + " |")

    lines.append(
        "\n**Each solver's own discretization error** (default grid against its refined grid,"
    )
    lines.append("surface; and at 50 cm, diurnal mean)\n")
    lines.append(
        "| Case | This solver: surface RMSE, max | 50 cm mean shift "
        "| heat1d: surface RMSE, max | 50 cm mean shift |"
    )
    lines.append("|---|---|---|---|---|")
    for r in rows:
        e = r["depths"]["0.50"]
        a, b = r["diff_ours_refined"], r["diff_heat1d_refined_default"]
        lines.append(
            f"| {r['label']} | {a['rmse']:.2f}, {a['max_abs']:.2f} "
            f"| {e['ours_refined_mean'] - e['ours_mean']:+.2f} "
            f"| {b['rmse']:.2f}, {b['max_abs']:.2f} "
            f"| {e['heat1d_refined_mean'] - e['heat1d_default_mean']:+.2f} |"
        )

    eq = rows[0]
    lines.append(
        "\n**heat1d grid convergence, equator, default parameters** (this solver: "
    )
    lines.append(
        f"{eq['ours']['predawn_min']:.2f} K pre-dawn, {eq['ours']['midnight']:.2f} K midnight, "
        f"{eq['depths']['0.50']['ours_mean']:.2f} K mean at 50 cm)\n"
    )
    lines.append(
        "| heat1d run | Nodes | Pre-dawn min | Midnight | 50 cm mean | Surface RMSE, max vs this solver |"
    )
    lines.append("|---|---|---|---|---|---|")
    for key, name in (
        ("default", "m = 10, n = 4 (default)"),
        ("mid", "m = 20, n = 10"),
        ("refined", "m = 40, n = 20"),
        ("default_patched", "m = 10, n = 4, midpoint conductivity (diagnostic)"),
    ):
        m = eq[f"heat1d_{key}"]
        c = eq[f"heat1d_{key}_convergence"]
        d = eq[f"diff_{key}"]
        lines.append(
            f"| {name} | {c['nodes']} | {m['predawn_min']:.2f} | {m['midnight']:.2f} "
            f"| {eq['depths']['0.50'][f'heat1d_{key}_mean']:.2f} | {d['rmse']:.2f}, {d['max_abs']:.2f} |"
        )

    lines.append(
        "\n**Diviner check, equator, 1 AU** (Table A2: 385 K noon, 101 K midnight, 95 K pre-dawn;"
    )
    lines.append(
        "Figure A2 residual: this solver minus the nine digitized 0° points)\n"
    )
    lines.append(
        "| Run | Noon max | Midnight | Pre-dawn min | Fig. A2 mean residual | Fig. A2 RMSE |"
    )
    lines.append("|---|---|---|---|---|---|")
    for e in div["equator"]:
        lines.append(
            f"| {e['label']} | {e['noon_max']:.1f} | {e['midnight']:.1f} | {e['predawn_min']:.1f} "
            f"| {e['fig_a2_mean_resid']:+.2f} | {e['fig_a2_rmse']:.2f} |"
        )

    lines.append(
        "\n**Figure A2 nighttime points, default parameters** (this solver minus Diviner)\n"
    )
    lines.append("| Latitude | Mean residual | Largest residual |")
    lines.append("|---|---|---|")
    for e in div["fig_a2"]:
        lines.append(
            f"| {e['lat']:g}° | {e['mean_resid']:+.2f} | {e['max_abs_resid']:.2f} |"
        )

    lines.append(
        "\n**Diurnal means: this solver, Figure A1's model curves (approx.) and the Apollo"
    )
    lines.append("measurements (Table A2, ±5 K)**, K\n")
    lines.append(
        "| Forcing | Latitude | Surface: this solver / Fig. A1 / Apollo "
        "| 1 m: this solver / Fig. A1 | Probe depth: this solver / Apollo |"
    )
    lines.append("|---|---|---|---|---|")
    for e in div["apollo"]:
        obs = f" / {e['surface_obs']:.0f}" if "site" in e else ""
        probe = (
            f"{e['depth']:.2f} m: {e['depth_mean']:.1f} / {e['depth_obs']:.0f}"
            if "site" in e
            else ""
        )
        where = f"{e['lat']:g}°" + (f" ({e['site']})" if "site" in e else "")
        lines.append(
            f"| {e['setting']} | {where} | {e['surface_mean']:.1f} / {e['fig_a1_surface']:.1f}{obs} "
            f"| {e['mean_1m']:.1f} / {e['fig_a1_1m']:.1f} | {probe} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------------

# Categorical slots 1-3 of the reference palette (validated for any pairing of the first three).
C_OURS, C_H1D, C_H1D_REF = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


def _clock_axis(ax):
    """x axis in hours after noon, labelled as the 24 h clock."""
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["12", "18", "00", "06", "12"])


def figure_overlays(tri: dict, names: list[str], path: Path, title: str) -> None:
    """Top row: both solvers' surface curves; bottom row: heat1d minus this solver."""
    import matplotlib.pyplot as plt

    x = hours_after_noon()
    fig, axes = plt.subplots(
        2, len(names), figsize=(2.6 * len(names), 4.6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1.2]},
    )  # fmt: skip
    by_name = {c.name: c for c in CASES}
    for j, name in enumerate(names):
        o = tri["ours"][name]["default"].surface_temperature
        h0 = tri["heat1d"][name]["default"]["surface"]
        h1 = tri["heat1d"][name]["refined"]["surface"]
        top, bot = axes[0, j], axes[1, j]
        top.plot(x, h0, color=C_H1D, lw=1.6, label="heat1d, default grid")
        top.plot(x, h1, color=C_H1D_REF, lw=1.6, label="heat1d, refined grid")
        top.plot(x, o, color=C_OURS, lw=1.2, ls="--", label="this solver")
        top.set_title(by_name[name].label, fontsize=8.5, color=INK)
        bot.axhline(0, color=INK_MUTED, lw=0.6)
        bot.plot(x, h0 - o, color=C_H1D, lw=1.4)
        bot.plot(x, h1 - o, color=C_H1D_REF, lw=1.4)
        for ax in (top, bot):
            _style(ax)
            _clock_axis(ax)
        bot.set_xlabel("Local time (h)", fontsize=8, color=INK_MUTED)
        if j == 0:
            top.set_ylabel("Surface temperature (K)", fontsize=8, color=INK_MUTED)
            bot.set_ylabel("heat1d − this solver (K)", fontsize=8, color=INK_MUTED)
    lim = max(abs(a) for ax in axes[1] for a in ax.get_ylim())
    for ax in axes[1]:
        ax.set_ylim(-lim, lim)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title, y=0.995, fontsize=10, color=INK)
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3,
        frameon=False, fontsize=8.5,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def figure_convergence(rows: list[dict], path: Path) -> None:
    """heat1d's equatorial pre-dawn minimum and 50 cm mean against its grid, with this solver's values."""
    import matplotlib.pyplot as plt

    eq = rows[0]
    keys = ("default", "mid", "refined")
    nodes = [eq[f"heat1d_{k}_convergence"]["nodes"] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    panels = (
        ("Pre-dawn surface minimum (K)", lambda k: eq[f"heat1d_{k}"]["predawn_min"],
         eq["ours"]["predawn_min"]),
        ("Diurnal mean at 50 cm (K)", lambda k: eq["depths"]["0.50"][f"heat1d_{k}_mean"],
         eq["depths"]["0.50"]["ours_mean"]),
    )  # fmt: skip
    for ax, (ylabel, get, ours_value) in zip(axes, panels, strict=True):
        ax.plot(nodes, [get(k) for k in keys], color=C_H1D, lw=1.6, marker="o", ms=6,
                label="heat1d (m, n) = (10, 4), (20, 10), (40, 20)")  # fmt: skip
        ax.plot(nodes[0], get("default_patched"), ls="none", marker="D", ms=6, color=INK_MUTED,
                label="heat1d default grid, midpoint conductivity (diagnostic)")  # fmt: skip
        ax.axhline(
            ours_value, color=C_OURS, lw=1.4, ls="--", label="this solver (39 nodes)"
        )
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(nodes)
        ax.set_xticklabels([str(n) for n in nodes])
        ax.set_xlabel("heat1d grid nodes", fontsize=8, color=INK_MUTED)
        ax.set_ylabel(ylabel, fontsize=8, color=INK_MUTED)
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Equator, default parameters: heat1d under grid refinement", y=0.995,
                 fontsize=10, color=INK)  # fmt: skip
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=1,
               frameon=False, fontsize=8)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.74))
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def figure_diviner(tri: dict, div: dict, path: Path) -> None:
    """Left: equatorial day with Table A2 values. Right: nights at 0/30/60° with Figure A2 points."""
    import matplotlib.pyplot as plt

    x = hours_after_noon()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    default = div["curves"]["Eq. A8 albedo, H = 0.06 m (default)"]
    const = div["curves"]["constant A0 = 0.12, H = 0.06 m"]
    ax1.plot(x, default, color=C_OURS, lw=1.6, label="this solver, Eq. A8 albedo")
    ax1.plot(
        x, const, color=C_OURS, lw=1.2, ls=":", label="this solver, constant A0 = 0.12"
    )
    # Table A2 values: noon at 12:00, midnight at 00:00; the pre-dawn minimum has no stated time,
    # so it is drawn as a short bar over the last hour before sunrise.
    ax1.plot([0], [TABLE_A2_EQUATOR["noon_max"]], ls="none", marker="o", ms=7, color=INK,
             label="Table A2 (Diviner, Vasavada et al. 2012)")  # fmt: skip
    ax1.plot(
        [12], [TABLE_A2_EQUATOR["midnight"]], ls="none", marker="o", ms=7, color=INK
    )
    ax1.plot([17, 18], [TABLE_A2_EQUATOR["predawn_min"]] * 2, color=INK, lw=3)
    ax1.set_ylabel("Surface temperature (K)", fontsize=8, color=INK_MUTED)
    ax1.set_title("Equator, 1 AU, default parameters", fontsize=9, color=INK)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper right")
    _clock_axis(ax1)

    # Night, 18:00 to 06:00 (6 to 18 hours after noon).
    markers = {0.0: "o", 30.0: "s", 60.0: "^"}
    for e in div["fig_a2"]:
        lat = e["lat"]
        surface = tri["ours"][f"lat{int(lat):02d}"]["default"].surface_temperature
        ax2.plot(x, surface, color=C_OURS, lw=1.4)
        xs = (np.array(e["local_time"]) - 12.0) % 24.0
        ax2.plot(xs, e["diviner"], ls="none", marker=markers[lat], ms=6, color=INK, mfc="white",
                 mew=1.2)  # fmt: skip
        ax2.annotate(f"{lat:g}°", (17.2, at_local_time(surface, [5.2])[0]), fontsize=8,
                     color=INK_MUTED, xytext=(4, 0), textcoords="offset points")  # fmt: skip
    ax2.plot([], [], color=C_OURS, lw=1.4, label="this solver")
    ax2.plot([], [], ls="none", marker="o", ms=6, color=INK, mfc="white",
             label="Diviner, Fig. A2 (digitized, approx.)")  # fmt: skip
    ax2.set_xlim(6, 18)
    ax2.set_ylim(78, 125)
    ax2.set_xticks([6, 8, 10, 12, 14, 16, 18])
    ax2.set_xticklabels(["18", "20", "22", "00", "02", "04", "06"])
    ax2.set_title("Night at 0°, 30°, 60° (markers ○ □ △)", fontsize=9, color=INK)
    ax2.legend(frameon=False, fontsize=7.5, loc="upper right")
    for ax in (ax1, ax2):
        _style(ax)
        ax.set_xlabel("Local time (h)", fontsize=8, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if k != "curves"}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.floating | float):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def write_results(
    path: Path, tri: dict, rows: list[dict], div: dict, meta: dict
) -> None:
    """One `.npz`: every surface and depth curve, plus a JSON summary under `summary_json`."""
    arrays = {"hours_after_noon": hours_after_noon(), "depths": np.array(DEPTHS)}
    for name, o in tri["ours"].items():
        arrays[f"{name}/flux"] = o["flux"]
        for key in ("default", "refined"):
            arrays[f"{name}/ours_{key}/surface"] = o[key].surface_temperature
            arrays[f"{name}/ours_{key}/depths"] = o[key].depth_temperature
        for key, run in tri["heat1d"][name].items():
            arrays[f"{name}/heat1d_{key}/surface"] = run["surface"]
            arrays[f"{name}/heat1d_{key}/nodes"] = run["nodes"]
            arrays[f"{name}/heat1d_{key}/z"] = run["z"]
    for label, curve in div["curves"].items():
        arrays[f"diviner/{label}"] = curve
    summary = {
        "meta": meta,
        "cases": rows,
        "diviner": div,
        "flux_check": tri["flux_check"],
    }
    arrays["summary_json"] = np.array(json.dumps(_jsonable(summary), indent=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Triangulate the solver against heat1d; Diviner check."
    )
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="results .npz (outside the repo)"
    )
    ap.add_argument(
        "--figures",
        type=Path,
        default=DEFAULT_FIGURES,
        help="directory for PNG figures",
    )
    ap.add_argument(
        "--workers", type=int, default=12, help="processes for the heat1d runs"
    )
    ap.add_argument("--years", type=int, default=10, help="heat1d equilibration, years")
    ap.add_argument(
        "--out-cycles", type=int, default=4, help="heat1d output lunations (>= 3)"
    )
    args = ap.parse_args(argv)

    out = args.out.resolve()
    if out.is_relative_to(REPO_ROOT):
        ap.error(
            f"--out must be outside the repo ({REPO_ROOT}); results data stays out of git"
        )
    if args.out_cycles < 3:
        ap.error(
            "--out-cycles must be at least 3 (the drift estimate needs two changes)"
        )

    t0 = time.perf_counter()
    print("Running cases ...", file=sys.stderr)
    tri = run_triangulation(args.workers, args.years, args.out_cycles)
    rows = summarise(tri)
    print("Diviner check ...", file=sys.stderr)
    div = diviner_check(tri)

    # Cost per lunation: this solver's spin-up cycles, heat1d's simulated lunations (equilibration,
    # warm-up and output), both at default grids, over the eight cases.
    ours_per = np.mean(
        [
            r["ours_convergence"]["seconds"] / r["ours_convergence"]["cycles"]
            for r in rows
        ]
    )
    h1d_per = np.mean(
        [
            r["heat1d_default_convergence"]["seconds"]
            / r["heat1d_default_convergence"]["lunations"]
            for r in rows
        ]
    )
    meta = {
        "heat1d": HEAT1D_SOURCE,
        "n_samples": N_SAMPLES,
        "dt": tri["ours"]["lat00"]["dt"],
        "heat1d_grids": HEAT1D_GRIDS,
        "heat1d_years": args.years,
        "heat1d_out_cycles": args.out_cycles,
        "ours_refined": {
            "layers_per_skin_depth": OURS_REFINED.layers_per_skin_depth,
            "growth": OURS_REFINED.growth,
            "max_step": OURS_REFINED.max_step,
        },
        "seconds_per_lunation": {
            "ours": float(ours_per),
            "heat1d_default": float(h1d_per),
        },
        "wall_seconds": time.perf_counter() - t0,
    }

    write_results(out, tri, rows, div, meta)
    args.figures.mkdir(parents=True, exist_ok=True)
    figure_overlays(
        tri, ["lat00", "lat30", "lat60", "lat75"], args.figures / "triangulation_latitudes.png",
        "Flat surface, default parameters, 1 AU: heat1d against this solver",
    )  # fmt: skip
    figure_overlays(
        tri, ["eq_H0.03", "eq_H0.10", "eq_A0.08", "eq_A0.16"],
        args.figures / "triangulation_equator_params.png",
        "Equator, 1 AU: H and albedo variations",
    )  # fmt: skip
    figure_convergence(rows, args.figures / "triangulation_heat1d_convergence.png")
    figure_diviner(tri, div, args.figures / "diviner_check.png")

    print(markdown_tables(rows, div))
    print(
        f"\nheat1d flux vs Eqs. A8-A9: max |difference| {tri['flux_check']:.2e} W m^-2"
        f"\nCost per lunation: this solver {ours_per * 1e3:.1f} ms, heat1d (default grid) "
        f"{h1d_per * 1e3:.0f} ms"
        f"\nheat1d convergence (last-cycle change, projected drift, K):"
    )
    for r in rows:
        for key in ("default", "refined"):
            c = r[f"heat1d_{key}_convergence"]
            print(f"  {r['case']:9s} {key:8s} {c['last_change']:.4f} {c['drift']:.4f}")
    print(
        f"\nResults: {out}\nFigures: {args.figures}\nWall time: {meta['wall_seconds']:.0f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
