"""Synthetic forcings for solver tests. Test-only; not the project's forcing.

The project's forcing (`forcing.py`, a later deliverable) turns Penumbra
Luna's terrain-shadowed illumination into absorbed flux. Until that exists,
and to keep tests independent of it, these helpers build the simplest
realistic forcing: a flat surface on the equator at equinox, lit by a
clipped-cosine Sun over one synodic month.

Constants and formulas are from Hayne et al. (2017), Appendix A and Table A1
(arXiv:1711.00977 numbering).
"""

from __future__ import annotations

import numpy as np

#: Solar constant, W m^-2 (Table A1; Kopp and Lean 2011).
SOLAR_CONSTANT = 1361.0

#: Synodic month, s (Table A1; 29.5306 d).
SYNODIC_MONTH = 2.55024e6

#: Bond albedo at normal incidence and its angle-dependence constants (Eq. A8, Table A1).
ALBEDO_A0 = 0.12
ALBEDO_A = 0.06
ALBEDO_B = 0.25


def albedo(incidence: np.ndarray) -> np.ndarray:
    """Bond albedo at solar incidence angle ``incidence`` (radians), Eq. A8."""
    x = np.asarray(incidence, dtype=np.float64)
    return (
        ALBEDO_A0
        + ALBEDO_A * (x / (np.pi / 4)) ** 3
        + ALBEDO_B * (x / (np.pi / 2)) ** 8
    )


def equatorial_absorbed_flux(
    n_samples: int,
    period: float = SYNODIC_MONTH,
    distance_au: float = 1.0,
    angle_dependent_albedo: bool = True,
) -> tuple[np.ndarray, float]:
    """Absorbed flux on a flat equatorial surface at equinox over one lunar day.

    Sample ``k`` is at time ``k * dt`` with ``dt = period / n_samples``; time 0
    is local noon, so the hour angle is ``2 pi t / period``. Incident flux is
    ``S / r^2 * max(cos h, 0)`` (Eq. A9 with latitude and declination zero);
    absorbed flux is ``(1 - A(theta)) * incident`` with the albedo of Eq. A8, or
    a constant ``A0`` if ``angle_dependent_albedo`` is false.

    Returns:
        (absorbed flux in W m^-2, shape ``(n_samples,)``; the step ``dt`` in s)
    """
    dt = period / n_samples
    h = 2.0 * np.pi * np.arange(n_samples) * dt / period
    cos_i = np.cos(h)
    lit = cos_i > 0.0
    incident = np.where(lit, SOLAR_CONSTANT / distance_au**2 * cos_i, 0.0)
    if angle_dependent_albedo:
        theta = np.arccos(np.clip(cos_i, 0.0, 1.0))
        A = albedo(theta)
    else:
        A = np.full(n_samples, ALBEDO_A0)
    return (1.0 - A) * incident, dt
