"""Regolith thermophysical properties from Hayne et al. (2017).

This module is the single place where the regolith's physical properties are
defined: density as a function of depth, contact (phonon) conductivity as a
function of depth, total conductivity as a function of depth and temperature,
and specific heat capacity as a function of temperature. The solver
(`lunar_thermal.solver`) evaluates the same formulas inside its compiled
kernels; `tests/test_solver.py` checks that the two agree.

Source
------
Hayne, P. O., Bandfield, J. L., Siegler, M. A., Vasavada, A. R., Ghent, R. R.,
Williams, J.-P., Greenhagen, B. T., Aharonson, O., Elder, C. M., Lucey, P. G.,
& Paige, D. A. (2017). Global regolith thermophysical properties of the Moon
from the Diviner Lunar Radiometer Experiment. Journal of Geophysical Research:
Planets, 122, 2371-2400. https://doi.org/10.1002/2017JE005387

Equation and table numbers cited below are those of the arXiv preprint of that
paper (arXiv:1711.00977, the version openly available), Appendix A. The
published version was not accessible when this was written; if its numbering
differs, the citations here need updating.

Units are SI throughout: metres, kelvin, kilograms, seconds, watts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Stefan-Boltzmann constant, W m^-2 K^-4 (CODATA 2018 exact value).
STEFAN_BOLTZMANN = 5.670374419e-8

#: Reference temperature in the radiative-conductivity term, K (Eq. A4).
RADIATIVE_REFERENCE_TEMPERATURE = 350.0


@dataclass(frozen=True)
class RegolithParams:
    """Regolith parameters for one column, defaulting to Hayne et al. (2017).

    Every default is the "standard model" value in Table A1 of Hayne et al.
    (2017), the parameter table for the thermal model in their Appendix A.

    Attributes:
        H: Scale height of the density and conductivity profiles, m (Eq. A2;
            Table A1, "Scale factor H: 0 to > 0.2 m (avg. = 0.06 m)"). The
            paper's main text gives a global mean of 0.068 m (Section 4,
            Figure 9) and the abstract says "~7 cm"; Table A1's 0.06 m is
            used here because it is the value the paper's own standard model
            runs use (Figures A2, A3). This is an open question for the
            project, not a settled choice.
        rho_s: Density at the surface, kg m^-3 (Eq. A2; Table A1, 1100).
        rho_d: Density at depth (z >> H), kg m^-3 (Eq. A2; Table A1, 1800).
        k_s: Contact conductivity at the surface, W m^-1 K^-1 (Eq. A5; Table
            A1, 7.4e-4).
        k_d: Contact conductivity at depth, W m^-1 K^-1 (Eq. A5; Table A1,
            3.4e-3).
        chi: Radiative conductivity parameter, dimensionless (Eq. A4; Table
            A1, 2.7).
        cp_coeffs: Coefficients c0..c4 of the heat-capacity polynomial, in
            J kg^-1 K^-(1+i) (Eq. A6; Table A1). c0 = -3.6125, c1 = 2.7431,
            c2 = 2.3616e-3, c3 = -1.2340e-5, c4 = 8.9093e-9.
        emissivity: Infrared emissivity, dimensionless (Eq. A7; Table A1,
            0.95).
        geothermal_flux: Interior heat flow at the bottom of the column,
            W m^-2 (Eq. A12; Table A1, 0.018, from Langseth et al. 1976).

    The dataclass is frozen so a parameter set can be shared between runs and
    used as a dictionary key without being changed by accident. Tests build
    constant-property columns by setting ``rho_s == rho_d``, ``k_s == k_d``,
    ``chi = 0`` and ``cp_coeffs = (c, 0, 0, 0, 0)``.
    """

    H: float = 0.06
    rho_s: float = 1100.0
    rho_d: float = 1800.0
    k_s: float = 7.4e-4
    k_d: float = 3.4e-3
    chi: float = 2.7
    cp_coeffs: tuple[float, ...] = (
        -3.6125,
        2.7431,
        2.3616e-3,
        -1.2340e-5,
        8.9093e-9,
    )
    emissivity: float = 0.95
    geothermal_flux: float = 0.018

    def __post_init__(self) -> None:
        # Catch parameter sets the formulas can't handle, at construction time
        # rather than as NaNs deep inside a solver run.
        if self.H <= 0.0:
            raise ValueError(f"H must be positive, got {self.H}")
        if self.rho_s <= 0.0 or self.rho_d <= 0.0:
            raise ValueError("densities must be positive")
        if self.k_s <= 0.0 or self.k_d <= 0.0:
            raise ValueError("contact conductivities must be positive")
        if self.rho_s == self.rho_d and self.k_s != self.k_d:
            # Eq. A5 ties conductivity to density; with no density contrast it
            # has no way to vary, so a conductivity contrast is contradictory.
            raise ValueError(
                "rho_s == rho_d requires k_s == k_d (Eq. A5 is undefined otherwise)"
            )
        if self.chi < 0.0:
            raise ValueError(f"chi must be non-negative, got {self.chi}")
        if not 0.0 < self.emissivity <= 1.0:
            raise ValueError(f"emissivity must be in (0, 1], got {self.emissivity}")
        if len(self.cp_coeffs) == 0:
            raise ValueError("cp_coeffs must hold at least one coefficient")
        # Store as a tuple of floats so the dataclass stays hashable even if a
        # list was passed.
        object.__setattr__(self, "cp_coeffs", tuple(float(c) for c in self.cp_coeffs))


def density(z: ArrayLike, params: RegolithParams) -> NDArray[np.float64]:
    """Bulk density at depth ``z``, kg m^-3.

    Eq. A2 of Hayne et al. (2017) (also Eq. 1 of the main text):
    ``rho(z) = rho_d - (rho_d - rho_s) * exp(-z / H)``.

    The exponential profile is what the Diviner nighttime cooling curves
    support: fluffy material at the surface compacting towards a deep value
    over a scale height H of a few centimetres.
    """
    z = np.asarray(z, dtype=np.float64)
    return params.rho_d - (params.rho_d - params.rho_s) * np.exp(-z / params.H)


def contact_conductivity(z: ArrayLike, params: RegolithParams) -> NDArray[np.float64]:
    """Contact (solid, phonon) conductivity at depth ``z``, W m^-1 K^-1.

    Eq. A5 of Hayne et al. (2017):
    ``Kc = Kd - (Kd - Ks) * (rho_d - rho) / (rho_d - rho_s)``.

    Contact conductivity is taken as linear in density (after Fountain and
    West, 1970), so it follows the same exponential profile as density:
    ``Kc(0) = Ks`` and ``Kc -> Kd`` at depth. When ``rho_s == rho_d`` the
    ratio is undefined and the profile is flat; this function then returns
    ``Kd`` everywhere, which is the constant-property limit tests rely on.
    """
    z = np.asarray(z, dtype=np.float64)
    if params.rho_d == params.rho_s:
        # Uniform density: Eq. A5 is 0/0, and RegolithParams has already
        # required k_s == k_d in this case, so conductivity is uniform too.
        return np.full_like(z, params.k_d)
    rho = density(z, params)
    return params.k_d - (params.k_d - params.k_s) * (params.rho_d - rho) / (
        params.rho_d - params.rho_s
    )


def conductivity(
    z: ArrayLike, T: ArrayLike, params: RegolithParams
) -> NDArray[np.float64]:
    """Total thermal conductivity at depth ``z`` and temperature ``T``, W m^-1 K^-1.

    Eq. A4 of Hayne et al. (2017) (Eq. A3 with ``B = Kc * chi / 350^3``):
    ``K = Kc(z) * (1 + chi * (T / 350 K)^3)``.

    The ``T^3`` term is radiative transfer across the pores between grains,
    which grows quickly with temperature. Hayne et al. scale it with the local
    contact conductivity ``Kc`` rather than with a constant, which is why the
    radiative part also increases with depth. ``z`` and ``T`` broadcast
    against each other.
    """
    T = np.asarray(T, dtype=np.float64)
    kc = contact_conductivity(z, params)
    return kc * (1.0 + params.chi * (T / RADIATIVE_REFERENCE_TEMPERATURE) ** 3)


def heat_capacity(T: ArrayLike, params: RegolithParams) -> NDArray[np.float64]:
    """Specific heat capacity at temperature ``T``, J kg^-1 K^-1.

    Eq. A6 of Hayne et al. (2017), with the four-degree polynomial and
    coefficients of Table A1 (fit to Hemingway et al. 1981 and Ledlow et al.
    1992): ``cp = c0 + c1 T + c2 T^2 + c3 T^3 + c4 T^4``.

    The paper states the fit is valid from below 90 K to above 400 K and turns
    negative below about 1.3 K. Below about 90 K (polar night) the values are
    an extrapolation; the paper notes that low-temperature heat capacity is
    poorly constrained.
    """
    T = np.asarray(T, dtype=np.float64)
    # Horner's rule: fewer operations and better rounding than summing powers.
    result = np.zeros_like(T)
    for c in reversed(params.cp_coeffs):
        result = result * T + c
    return result


def specific_enthalpy(T: ArrayLike, params: RegolithParams) -> NDArray[np.float64]:
    """Heat content per unit mass relative to 0 K, J kg^-1.

    The integral of `heat_capacity` from 0 to ``T``, done exactly on the
    polynomial: ``h(T) = sum_i c_i T^(i+1) / (i+1)``. This is not in Hayne et
    al. (2017); it follows directly from Eq. A6.

    The solver steps this quantity rather than temperature (``rho dh/dt =
    -div(flux)``), so the energy it stores is exactly the energy that crossed
    the boundaries, even though heat capacity changes with temperature. Only
    differences in ``h`` carry meaning; the 0 K reference is a convention.
    """
    T = np.asarray(T, dtype=np.float64)
    result = np.zeros_like(T)
    for i in reversed(range(len(params.cp_coeffs))):
        result = result * T + params.cp_coeffs[i] / (i + 1)
    return result * T


def thermal_diffusivity(
    z: ArrayLike, T: ArrayLike, params: RegolithParams
) -> NDArray[np.float64]:
    """Thermal diffusivity ``K / (rho cp)`` at depth ``z`` and temperature ``T``, m^2 s^-1.

    Used in the skin depth (Eq. A31 of Hayne et al. 2017), which sets the
    solver grid.
    """
    return conductivity(z, T, params) / (density(z, params) * heat_capacity(T, params))


def skin_depth(
    z: ArrayLike, T: ArrayLike, period: float, params: RegolithParams
) -> NDArray[np.float64]:
    """Thermal skin depth for a periodic forcing of ``period`` seconds, m.

    Eq. A31 of Hayne et al. (2017): ``zs = sqrt(kappa P / pi)``. A surface
    temperature wave with this period decays by a factor ``e`` over one skin
    depth. Evaluated with the properties at depth ``z`` and temperature ``T``,
    since both change the diffusivity.
    """
    kappa = thermal_diffusivity(z, T, params)
    return np.sqrt(kappa * period / np.pi)
