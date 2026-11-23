"""Tests for lunar_thermal.properties against Hayne et al. (2017).

Reference values are computed by hand from the published equations and the
Table A1 parameters of Hayne et al. (2017), Appendix A (arXiv:1711.00977
numbering). The arithmetic is written out next to each value so it can be
checked without running code.
"""

import numpy as np
import pytest

from lunar_thermal.properties import (
    RegolithParams,
    conductivity,
    contact_conductivity,
    density,
    heat_capacity,
    skin_depth,
    specific_enthalpy,
)

P = RegolithParams()


# --- Defaults match Table A1 ------------------------------------------------


def test_defaults_match_table_a1():
    """Every default is the Table A1 standard-model value."""
    assert P.H == 0.06  # "avg. = 0.06 m"
    assert P.rho_s == 1100.0
    assert P.rho_d == 1800.0
    assert P.k_s == 7.4e-4
    assert P.k_d == 3.4e-3
    assert P.chi == 2.7
    assert P.cp_coeffs == (-3.6125, 2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9)
    assert P.emissivity == 0.95
    assert P.geothermal_flux == 0.018


# --- Density, Eq. A2 --------------------------------------------------------


def test_density_is_surface_value_at_zero():
    """rho(0) = rho_d - (rho_d - rho_s) * exp(0) = rho_s exactly."""
    assert density(0.0, P) == pytest.approx(P.rho_s, rel=0, abs=1e-12)


def test_density_tends_to_deep_value():
    """At z = 20 H, exp(-20) = 2.1e-9, so rho is within 1e-8 relative of rho_d."""
    assert density(20 * P.H, P) == pytest.approx(P.rho_d, rel=1e-8)


def test_density_at_one_scale_height():
    """rho(H) = 1800 - 700 * exp(-1) = 1800 - 257.5156088 = 1542.4843912 kg m^-3."""
    assert density(P.H, P) == pytest.approx(1542.4843912, rel=1e-9)


def test_density_increases_with_depth():
    z = np.linspace(0, 1, 201)
    assert np.all(np.diff(density(z, P)) > 0)


# --- Contact conductivity, Eq. A5 -------------------------------------------


def test_contact_conductivity_bounds():
    """Kc(0) = Ks and Kc -> Kd at depth (Eq. A5 with rho = rho_s, rho_d)."""
    assert contact_conductivity(0.0, P) == pytest.approx(P.k_s, rel=1e-12)
    assert contact_conductivity(20 * P.H, P) == pytest.approx(P.k_d, rel=1e-8)


# --- Total conductivity, Eq. A4 ---------------------------------------------


@pytest.mark.parametrize(
    "z, T, expected",
    [
        # Surface, T = 350 K: Ks * (1 + chi) = 7.4e-4 * 3.7 = 2.738e-3.
        (0.0, 350.0, 2.738e-3),
        # Surface, T = 200 K: (200/350)^3 = 64/343 = 0.18658892;
        # 1 + 2.7 * 0.18658892 = 1.50379009; * 7.4e-4 = 1.11280466e-3.
        (0.0, 200.0, 1.11280466e-3),
        # Deep (Kc = Kd), T = 100 K: (100/350)^3 = 8/343 = 0.02332362;
        # 1 + 2.7 * 0.02332362 = 1.06297376; * 3.4e-3 = 3.61411079e-3.
        (100.0, 100.0, 3.61411079e-3),
    ],
)
def test_conductivity_hand_values(z, T, expected):
    """K = Kc(z) * (1 + chi * (T / 350)^3), Hayne et al. (2017) Eq. A4, Table A1."""
    assert conductivity(z, T, P) == pytest.approx(expected, rel=1e-7)


def test_conductivity_broadcasts():
    z = np.array([0.0, 0.1, 0.5])[:, None]
    T = np.array([100.0, 250.0, 400.0])[None, :]
    assert conductivity(z, T, P).shape == (3, 3)


# --- Heat capacity, Eq. A6 --------------------------------------------------


@pytest.mark.parametrize(
    "T, expected",
    [
        # c0 + c1 T + c2 T^2 + c3 T^3 + c4 T^4 with Table A1 coefficients.
        # T = 100: -3.6125 + 274.31 + 23.616 - 12.34 + 0.89093 = 282.86443
        (100.0, 282.86443),
        # T = 250: -3.6125 + 685.775 + 147.6 - 192.8125 + 34.8019531 = 671.7519531
        (250.0, 671.7519531),
        # T = 400: -3.6125 + 1097.24 + 377.856 - 789.76 + 228.07808 = 909.80158
        (400.0, 909.80158),
    ],
)
def test_heat_capacity_hand_values(T, expected):
    """Hayne et al. (2017) Eq. A6 with the Table A1 coefficients."""
    assert heat_capacity(T, P) == pytest.approx(expected, rel=1e-8)


def test_heat_capacity_positive_20_to_420_K():
    """The fit turns negative only below ~1.3 K (Hayne et al. 2017, Appendix A1)."""
    T = np.linspace(20.0, 420.0, 4001)
    assert np.all(heat_capacity(T, P) > 0)


def test_enthalpy_is_integral_of_heat_capacity():
    """h(0) = 0 and dh/dT = cp (central difference; tolerance 1e-8 relative)."""
    assert specific_enthalpy(0.0, P) == 0.0
    T = np.linspace(30.0, 400.0, 38)
    step = 1e-3
    dh = (specific_enthalpy(T + step, P) - specific_enthalpy(T - step, P)) / (2 * step)
    np.testing.assert_allclose(dh, heat_capacity(T, P), rtol=1e-8)


# --- Skin depth, Eq. A31 ----------------------------------------------------


def test_skin_depth_constant_properties():
    """zs = sqrt(kappa P / pi) with kappa = K / (rho c).

    K = 1e-3, rho = 1500, c = 600: kappa = 1.1111e-9 m^2 s^-1.
    P = 2.55024e6 s: zs = sqrt(1.1111e-9 * 2.55024e6 / pi) = sqrt(9.01962e-4) = 0.0300327 m.
    """
    p = RegolithParams(
        rho_s=1500, rho_d=1500, k_s=1e-3, k_d=1e-3, chi=0.0, cp_coeffs=(600.0,)
    )
    assert skin_depth(0.3, 250.0, 2.55024e6, p) == pytest.approx(0.0300327, rel=1e-5)


# --- Parameter validation ---------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"H": 0.0},
        {"rho_s": -1.0},
        {"k_d": 0.0},
        {"chi": -0.1},
        {"emissivity": 1.5},
        {"cp_coeffs": ()},
        {"rho_s": 1500.0, "rho_d": 1500.0},  # uniform density but Ks != Kd
    ],
)
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        RegolithParams(**kwargs)


def test_cp_coeffs_list_becomes_tuple():
    """A list is accepted and stored as a tuple, so params stay hashable."""
    p = RegolithParams(cp_coeffs=[600.0])
    assert p.cp_coeffs == (600.0,)
    hash(p)
