"""Triangulation of the solver against heat1d (Hayne's own model) as regression guards.

`scripts/triangulate_heat1d.py` runs the full comparison and documents it
(docs/methodology.md, "Solver triangulation and Diviner check"). These tests keep
two pieces of it from drifting:

* the shared forcing: heat1d's absorbed-flux series for the circularised Moon is
  Hayne et al. Eqs. A8-A9 at zero declination, evaluated independently here;
* one equatorial standard-model case (Table A1, H = 0.06 m), this solver
  against heat1d's explicit solver at heat1d's default grid. The tolerances are
  set from the agreement observed on 2026-11-23 with these exact settings, plus
  a margin. They are regression bounds, not accuracy claims: at its default
  grid heat1d carries a first-order discretization error (about +5 K at 50 cm,
  -0.8 K in the pre-dawn minimum) that shrinks toward this solver's answer as
  heat1d's grid is refined.

heat1d is a dev dependency (pinned in pyproject.toml); the tests skip without it.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

heat1d = pytest.importorskip("heat1d")

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "triangulate_heat1d.py"


def _load_script():
    """Import scripts/triangulate_heat1d.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("triangulate_heat1d", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered before running it: dataclasses look their module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tri = _load_script()


@pytest.mark.parametrize(
    "lat, kwargs",
    [
        (0.0, {}),
        (30.0, {}),
        (60.0, {}),
        (75.0, {}),
        (0.0, {"A0": 0.16}),
        (0.0, {"coeffs": (0.0, 0.0)}),
        (26.0, {"distance_au": 1.017}),
        (20.0, {"A0": 0.06, "coeffs": (0.03, 0.125)}),
    ],
)
def test_heat1d_flux_is_eqs_a8_a9(lat, kwargs):
    """Both solvers' forcing equals Eqs. A8-A9 to rounding (measured: 2e-7 W m^-2)."""
    flux, dt = tri.absorbed_flux(lat, **kwargs)
    assert flux.shape == (tri.N_SAMPLES,)
    assert dt * tri.N_SAMPLES == pytest.approx(29.53059 * 86400.0)
    np.testing.assert_allclose(flux, tri.paper_flux(lat, **kwargs), rtol=0, atol=1e-6)


@pytest.mark.slow
def test_equator_default_against_heat1d():
    """Equator, standard model (H = 0.06 m), 1 AU: this solver against heat1d's default grid.

    Observed on 2026-11-23 with heat1d equilibrated for 5 years (heat1d minus this
    solver): surface RMSE 0.72 K, max 3.26 K (at sunrise); noon maximum +0.10 K;
    pre-dawn minimum -0.74 K; midnight -0.86 K; diurnal mean at 50 cm +5.54 K; last
    heat1d cycle-to-cycle change 0.013 K. Each bound below is the
    observed value plus a margin; the 50 cm band is two-sided so a change in this
    solver's deep temperature in either direction fails.
    """
    flux, dt = tri.absorbed_flux(0.0)
    params = tri.RegolithParams.standard_model()
    ours, _ = tri.run_ours(flux, dt, params)
    h = tri.run_heat1d(
        {
            "lat": 0.0,
            "H": params.H,
            "A0": 0.12,
            "grid": "default",
            "years": 5,
            "out_cycles": 3,
            "patched": False,
        }
    )
    assert h["changes"][-1] < 0.05  # heat1d close to its own periodic state

    diff = tri.difference(h["surface"], ours.surface_temperature)
    assert diff["rmse"] < 0.9
    assert diff["max_abs"] < 4.0

    m_ours = tri.curve_metrics(ours.surface_temperature, flux)
    m_h = tri.curve_metrics(h["surface"], flux)
    assert abs(m_h["noon_max"] - m_ours["noon_max"]) < 0.3
    assert -1.2 < m_h["predawn_min"] - m_ours["predawn_min"] < -0.3
    assert -1.3 < m_h["midnight"] - m_ours["midnight"] < -0.4

    j = tri.DEPTHS.index(0.50)
    deep_diff = tri.heat1d_depth(h, 0.50).mean() - ours.depth_temperature[:, j].mean()
    assert 4.0 < deep_diff < 6.5
