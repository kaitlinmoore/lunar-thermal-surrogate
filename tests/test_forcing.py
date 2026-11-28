"""Tests for the absorbed-flux forcing (`lunar_thermal.forcing`).

Two kinds:

* Pure tests (run everywhere, including CI): Eq. A8, the incidence and
  azimuth/aspect conventions, the flux formula's zeros, sample times, the
  engine's slope and aspect read through this module on synthetic tilted
  planes in each map projection (including the matched east-west baseline
  near the north pole), and the refusal at the poles on the global map.
* `data` tests (need SPICE kernels and DEMs under LSS_DATA_DIR; skipped
  without them): the Sun distance over a year, a flat surface under a zero
  horizon against an independent SPICE evaluation, the aspect convention with
  real ephemerides, zeros and the solver contract on real terrain, the visible
  fraction against `compute_visibility`, the Sun-only call against a full
  one, and the slope against the engine's own slope statistics, at the two
  demonstration points (85 S 90 E and 5 N 60 W).

Every location used here is checked to lie at least 50 km from all 23
held-out sites (`test_locations_are_away_from_the_sites`); no test reads a
site's terrain or illumination. Each test states its tolerance.
"""

from __future__ import annotations

import math
import warnings
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import synthetic_forcing

from lunar_thermal import forcing as fm
from lunar_thermal.forcing import (
    DEFAULT_SAMPLES,
    SYNODIC_MONTH_S,
    Facet,
    HorizonGapWarning,
    absorbed_flux,
    albedo_eq_a8,
    compute_forcing,
    cos_incidence,
    sample_times,
)

# The illumination demonstration points (95 km and 1134 km from the nearest site),
# and a southern mid-latitude point used only with a synthetic zero horizon.
POLAR = (-85.0, 90.0)
EQUATORIAL = (5.0, -60.0)
MIDLAT_SOUTH = (-45.0, 0.0)
START = datetime(2027, 1, 1, tzinfo=UTC)
ALL_POINTS = (POLAR, EQUATORIAL, MIDLAT_SOUTH)


def _have_data() -> bool:
    try:
        from lss.datafiles import get

        ids = ("naif0012", "pck00010", "de421_spk", "moon_pa_de421", "moon_fk")
        ids += ("moon_assoc_me", "ldem_80s_80m", "ldem_60s_240m", "ldem_128_img")
        return all(get(i).path.is_file() for i in ids)
    except (KeyError, OSError, ValueError):
        return False


def _arc_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    h = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2 * 1737.4 * math.asin(min(1.0, math.sqrt(h)))


def test_locations_are_away_from_the_sites():
    """Every test point is >= 50 km from each of the 23 held-out sites (exact)."""
    from lss.registry.sites import load_sites

    sites = load_sites()
    assert len(sites) == 23
    for lat, lon in ALL_POINTS:
        d = min(_arc_km(lat, lon, s.location.lat, s.location.lon) for s in sites)
        assert d >= 50.0, (lat, lon, d)


# --------------------------------------------------------------------------- Eq. A8


@pytest.mark.parametrize(
    ("deg", "expected_minus_a0"),
    [
        (0.0, 0.0),
        # a (1)^3 + b (1/2)^8
        (45.0, 0.06 + 0.25 / 256.0),
        # a (4/3)^3 + b (2/3)^8
        (60.0, 0.06 * (4.0 / 3.0) ** 3 + 0.25 * (2.0 / 3.0) ** 8),
        # a 2^3 + b
        (90.0, 0.06 * 8.0 + 0.25),
    ],
)
def test_eq_a8_values(deg, expected_minus_a0):
    """Eq. A8 at 0, 45, 60, 90 deg against closed forms and tests/synthetic_forcing.py.

    Tolerance 1e-12 (rounding).
    """
    a0 = 0.12
    got = float(albedo_eq_a8(math.radians(deg), a0))
    assert got == pytest.approx(a0 + expected_minus_a0, abs=1e-12)
    independent = float(synthetic_forcing.albedo(math.radians(deg)))  # A0 = 0.12 there
    assert got == pytest.approx(independent, abs=1e-12)


def test_eq_a8_is_clipped_beyond_grazing():
    """Beyond 90 deg the albedo stays at its 90 deg value (exact)."""
    assert albedo_eq_a8(math.radians(120.0), 0.1) == albedo_eq_a8(math.pi / 2, 0.1)
    assert albedo_eq_a8(-0.3, 0.1) == albedo_eq_a8(0.0, 0.1)


# --------------------------------------------------------------------------- conventions


def test_flat_cos_incidence_is_sin_elevation():
    """On flat ground cos i = sin(el) at any azimuth and aspect. Tolerance 1e-15."""
    az = np.linspace(0.0, 359.0, 37)
    el = np.linspace(-30.0, 90.0, 37)
    np.testing.assert_allclose(
        cos_incidence(az, el, 0.0, 123.0), np.sin(np.radians(el)), atol=1e-15
    )


def test_aspect_convention_on_a_southern_point():
    """Southern hemisphere: the noon Sun is due north (az 0). A slope facing north
    (aspect 0, equator-facing) tilts toward it: incidence = zenith angle - slope;
    facing south (aspect 180, pole-facing) away: zenith angle + slope. Tolerance
    1e-12."""
    el, s = 40.0, 15.0  # zenith angle 50 deg
    north = float(cos_incidence(0.0, el, s, 0.0))
    south = float(cos_incidence(0.0, el, s, 180.0))
    assert north == pytest.approx(math.cos(math.radians(50.0 - s)), abs=1e-12)
    assert south == pytest.approx(math.cos(math.radians(50.0 + s)), abs=1e-12)
    assert north > south


def test_azimuth_is_clockwise_from_north():
    """A morning Sun in the east (az 90) lights an east-facing slope (aspect 90)
    more than a west-facing one (aspect 270); a slope facing the Sun's azimuth
    at elevation 90 - s is lit at normal incidence. Tolerance 1e-12."""
    east = float(cos_incidence(90.0, 10.0, 30.0, 90.0))
    west = float(cos_incidence(90.0, 10.0, 30.0, 270.0))
    assert east > west
    assert float(cos_incidence(200.0, 60.0, 30.0, 200.0)) == pytest.approx(
        1.0, abs=1e-12
    )


def test_flat_facet_and_missing_slope_conventions():
    """The engine's NaN aspect on a flat pixel becomes aspect 0; NaN slope, or NaN
    aspect on a sloped pixel (a polar pixel centred on the pole), raise ValueError;
    anything else passes through unchanged (exact)."""
    assert fm._conventions(0.0, math.nan, "here") == (0.0, 0.0)
    assert fm._conventions(3.5, 271.25, "here") == (3.5, 271.25)
    with pytest.raises(ValueError, match="no slope at here"):
        fm._conventions(math.nan, math.nan, "here")
    with pytest.raises(ValueError, match="undefined at the pole"):
        fm._conventions(2.0, math.nan, "here")


# --------------------------------------------------------------------------- flux formula


def test_flux_is_zero_without_sun_or_behind_the_facet():
    """Flux is exactly 0 where the visible fraction is 0 or cos i <= 0, and equals
    S/r^2 f cos i (1 - A) elsewhere. Tolerance 1e-12 W m^-2."""
    r = np.array([1.0, 1.0, 0.99, 1.01, 1.0])
    f = np.array([0.0, 1.0, 1.0, 0.5, 1.0])
    c = np.array([0.8, -0.2, 0.0, 0.6, 1.0])
    flux, alb = absorbed_flux(r, f, c, 0.12)
    assert flux[0] == 0.0 and flux[1] == 0.0 and flux[2] == 0.0
    assert not np.signbit(flux).any()  # no -0.0
    expected = 1361.0 / r**2 * f * np.maximum(c, 0) * (1 - alb)
    np.testing.assert_allclose(flux, expected, atol=1e-12)
    assert flux[4] == pytest.approx(1361.0 * (1 - 0.12), abs=1e-12)


def test_sample_times_meet_the_solver_contract():
    """Default: 709 samples over a synodic month, uniform, no duplicated endpoint,
    step at most an hour. Tolerance 1e-9 s."""
    t, dt = sample_times(DEFAULT_SAMPLES, SYNODIC_MONTH_S)
    assert DEFAULT_SAMPLES == 709
    assert t.size == 709 and t[0] == 0.0
    assert dt == pytest.approx(SYNODIC_MONTH_S / 709, abs=1e-9)
    assert dt <= 3600.0
    np.testing.assert_allclose(np.diff(t), dt, atol=1e-6)
    assert t[-1] + dt == pytest.approx(SYNODIC_MONTH_S, abs=1e-6)
    assert t[-1] < SYNODIC_MONTH_S - 0.5 * dt


def test_bad_arguments_raise():
    """Naive time, non-positive height, slope without aspect, A0 too high (exact)."""
    with pytest.raises(ValueError, match="timezone"):
        compute_forcing(0.0, 0.0, 1.0, datetime(2027, 1, 1), a0=0.12)
    with pytest.raises(ValueError, match="height"):
        compute_forcing(0.0, 0.0, 0.0, START, a0=0.12)
    with pytest.raises(ValueError, match="both"):
        compute_forcing(0.0, 0.0, 1.0, START, a0=0.12, slope_deg=5.0)
    with pytest.raises(ValueError, match="albedo"):
        compute_forcing(0.0, 0.0, 1.0, START, a0=0.3)
    with pytest.raises(ValueError, match="n_samples"):
        sample_times(1, 100.0)


# --------------------------------------------------------------------------- slope geometry


def _plane(e, n, slope_deg, aspect_deg):
    """Heights of a plane facing `aspect_deg` (downhill) with `slope_deg`, at local
    east/north offsets (m)."""
    t = math.tan(math.radians(slope_deg))
    a = math.radians(aspect_deg)
    return -t * (e * math.sin(a) + n * math.cos(a))


def _local_en(lat0, lon0, lat, lon):
    """East/north offsets (m) of (lat, lon) from (lat0, lon0) in the tangent plane
    at (lat0, lon0), from 3D positions on the 1737.4 km sphere (independent of the
    projection formulas under test)."""
    from lss.ephem import enu_basis

    east, north, up = enu_basis(lat0, lon0)
    p = 1_737_400.0 * enu_basis(lat, lon)[2]
    p0 = 1_737_400.0 * up
    d = p - p0
    return float(d @ east), float(d @ north)


@pytest.mark.parametrize("pole", ["south", "north"])
@pytest.mark.parametrize("lon", [0.0, 37.0, 90.0, 200.0, 315.0])
@pytest.mark.parametrize("aspect", [0.0, 90.0, 225.0])
def test_polar_stereographic_facet_on_a_tilted_plane(pole, lon, aspect):
    """A known plane sampled on a 3x3 polar stereographic window (80 m pixels) at
    85 deg latitude, defined in the tangent frame of the centre pixel's centre,
    recovers its slope and aspect through the engine's `slope_aspect`, whose slope
    equals its `slope_deg` exactly. Tolerance 0.01 deg; baseline two pixels over
    the scale factor (within 0.5%)."""
    from lss.terrain.core import latlon_from_ps, ps_from_latlon

    lat = -85.0 if pole == "south" else 85.0
    res, slope = 80.0, 12.0
    x_c, y_c = (float(v) for v in ps_from_latlon(lat, lon, pole))
    lat_c, lon_c = (float(v) for v in latlon_from_ps(x_c, y_c, pole))
    z3 = np.empty((3, 3))
    for i, dy in enumerate((res, 0.0, -res)):  # row 0 is +y (up-map)
        for j, dx in enumerate((-res, 0.0, res)):
            la, lo = (float(v) for v in latlon_from_ps(x_c + dx, y_c + dy, pole))
            z3[i, j] = _plane(*_local_en(lat_c, lon_c, la, lo), slope, aspect)
    s, a, engine, base = fm._ps_facet(z3, x_c, y_c, res, pole)
    assert s == pytest.approx(slope, abs=0.01)
    assert engine == s
    assert base[0] == base[1] == pytest.approx(2.0 * res, rel=0.005)
    assert (a - aspect + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=0.01)


def _lonlat_patch(lat, lon, slope, aspect, res_deg=1.0 / 128.0):
    """A LonLatGrid of three rows centred on (lat, lon), wide enough for the
    engine's matched east-west offset, holding a plane defined in the tangent
    frame at (lat, lon)."""
    from lss.terrain.core import LonLatGrid
    from lss.terrain.slope import ew_offset_px

    half = int(ew_offset_px(lat)) + 2
    lats = lat + res_deg * np.array([1.0, 0.0, -1.0])  # rows north to south
    lons = lon + res_deg * np.arange(-half, half + 1)
    z = np.array(
        [
            [_plane(*_local_en(lat, lon, la, lo), slope, aspect) for lo in lons]
            for la in lats
        ]
    )
    g = LonLatGrid(z, lon - (half + 0.5) * res_deg, lat + 1.5 * res_deg, res_deg, 236.9)
    return g, half


@pytest.mark.parametrize("lat", [-60.0, 5.0, 75.0, 85.0, 89.0])
@pytest.mark.parametrize("aspect", [0.0, 135.0, 270.0])
def test_lonlat_facet_on_a_tilted_plane(lat, aspect):
    """A known plane on a latitude/longitude grid (1/128 deg pixels) recovers its
    slope and aspect through the engine's `slope_aspect_lonlat`, whose slope equals
    `slope_deg_lonlat` exactly. Tolerance 0.01 deg (slope) and 0.05 deg (aspect,
    the engine's own test tolerance on these grids). The east-west baseline is the
    engine's matched one: 2 n pixels, within 2/3 to 4/3 of the north-south one
    (1e-9 relative)."""
    from lss.terrain.slope import ew_offset_px

    g, _ = _lonlat_patch(lat, 123.0, 8.0, aspect)
    s, a, engine, (b_ew, b_ns) = fm._lonlat_grid_facet(g, lat, 123.0, "test grid")
    assert s == pytest.approx(8.0, abs=0.01)
    assert engine == s
    assert (a - aspect + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=0.05)
    step = math.radians(1.0 / 128.0) * 1_737_400.0
    n = int(ew_offset_px(lat))
    assert b_ns == pytest.approx(2.0 * step, rel=1e-9)
    assert b_ew == pytest.approx(2.0 * n * step * math.cos(math.radians(lat)), rel=1e-9)
    assert 2.0 / 3.0 - 1e-9 <= b_ew / b_ns <= 4.0 / 3.0 + 1e-9


def test_lonlat_facet_flat_and_too_narrow():
    """A flat patch has slope 0 and aspect 0 (after the conventions); a grid without
    the matched east-west neighbours gives no slope (exact)."""
    from lss.terrain.core import LonLatGrid

    g, half = _lonlat_patch(85.0, 10.0, 0.0, 0.0)
    s, a, _, _ = fm._lonlat_grid_facet(g, 85.0, 10.0, "flat")
    assert s == 0.0 and math.isnan(a)
    assert fm._conventions(s, a, "flat") == (0.0, 0.0)
    g, half = _lonlat_patch(85.0, 10.0, 5.0, 90.0)  # n = 11 at 85 deg
    narrow = LonLatGrid(
        g.array[:, half - 3 : half + 4],
        10.0 - 3.5 * g.res_deg,
        g.lat0,
        g.res_deg,
        g.res,
    )
    s, _, _, _ = fm._lonlat_grid_facet(narrow, 85.0, 10.0, "narrow")
    assert math.isnan(s)
    with pytest.raises(ValueError, match="no slope"):
        fm._conventions(s, math.nan, "narrow")


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("lon", [0.0, 123.4, -90.0])
def test_global_map_refuses_within_0_0081_deg_of_a_pole(sign, lon):
    """On an image of LDEM_128's geometry (flat, synthetic: no data needed), the
    facet is refused within 0.0081 deg (about 246 m) of either pole: the pole row
    (1/128 deg), where the engine has no slope, plus about 10 m where the window
    would span every longitude. It is given (flat: slope 0) at 0.0082 deg and
    0.01 deg, where the engine's east-west offset is about 7,000 and 5,700
    pixels (exact)."""
    from lss.terrain.global_layer import LDEM_128

    img = np.broadcast_to(np.int16(0), (LDEM_128.lines, LDEM_128.samples))
    for d in (0.0, 0.004, 0.0078, 0.0081):
        with pytest.raises(ValueError, match="too close to the pole"):
            fm._lonlat_facet(img, sign * (90.0 - d), lon, 236.901, "flat image")
    for d in (0.0082, 0.01):
        s, a, _, _ = fm._lonlat_facet(
            img, sign * (90.0 - d), lon, 236.901, "flat image"
        )
        assert s == 0.0 and math.isnan(a)


# --------------------------------------------------------------------------- data tests


def _zero_horizon(monkeypatch, missing: int = 0):
    """Make the engine see a horizon of 0 deg everywhere, at terrain height 0.

    Replaces the cache lookup that `lss.series.compute_visibility` uses, so the
    whole engine path (ephemeris, disc fraction, events) runs on it.
    """
    from lss.horizon import Horizon, HorizonParams

    def fake(lat, lon, unit_height_m, params=None, layers=None, **_):
        p = params or HorizonParams()
        n = p.n_azimuths
        return Horizon(
            lat=lat,
            lon=lon,
            unit_height_m=unit_height_m,
            surface_height_m=0.0,
            azimuth_deg=np.arange(n) * p.azimuth_step_deg,
            elevation_deg=np.zeros(n),
            distance_m=np.full(n, np.nan),
            missing_samples=missing,
            total_samples=1000 * n,
            params=p,
        )

    monkeypatch.setattr("lss.series.core.cached_horizon", fake)


def _spice_sun(lat, lon, observer_height_m, start, t_s):
    """Sun's sin(elevation) and distance (AU) from raw SPICE, independent of
    lss.ephem.topocentric: observer on the sphere normal, elevation from the
    dot product with that normal."""
    import spiceypy as sp

    from lss.ephem import load_kernels

    load_kernels()
    phi, lam = math.radians(lat), math.radians(lon)
    up = np.array(
        [math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)]
    )
    obs_km = (1_737_400.0 + observer_height_m) * up / 1000.0
    et0 = sp.str2et(start.strftime("%Y-%m-%dT%H:%M:%S"))
    sin_el, dist = [], []
    for t in t_s:
        state, _ = sp.spkcpo(
            "SUN",
            et0 + t,
            "MOON_ME_DE421",
            "OBSERVER",
            "LT+S",
            obs_km,
            "MOON",
            "MOON_ME_DE421",
        )
        v = np.asarray(state[:3])
        d = np.linalg.norm(v)
        sin_el.append(float(v @ up) / d)
        dist.append(d / 149_597_870.7)
    return np.array(sin_el), np.array(dist)


@pytest.mark.data
@pytest.mark.skipif(not _have_data(), reason="needs kernels and DEMs in LSS_DATA_DIR")
@pytest.mark.parametrize("point", [EQUATORIAL, MIDLAT_SOUTH])
def test_flat_zero_horizon_matches_independent_evaluation(point, monkeypatch):
    """Flat ground, zero horizon: flux = S (1/r)^2 sin(el) (1 - A(90 - el)) at every
    sample where the whole disc is up, and 0 where it is all down, with el and r
    from raw SPICE and A from tests/synthetic_forcing.py. Tolerance 1e-6 W m^-2;
    partly risen samples lie between 0 and the full-disc value."""
    _zero_horizon(monkeypatch)
    lat, lon = point
    f = compute_forcing(lat, lon, 1.0, START, a0=0.12, slope_deg=0.0, aspect_deg=0.0)
    sin_el, r = _spice_sun(lat, lon, f.observer_height_m, START, f.t_s)
    np.testing.assert_allclose(f.distance_au, r, rtol=1e-12)
    np.testing.assert_allclose(f.cos_incidence, sin_el, atol=1e-12)
    full = (
        (1361.0 / r**2) * sin_el * (1.0 - synthetic_forcing.albedo(np.arccos(sin_el)))
    )
    up = sin_el > math.sin(math.radians(0.3))  # disc radius is ~0.27 deg
    down = sin_el < -math.sin(math.radians(0.3))
    assert up.sum() > 300 and down.sum() > 300
    np.testing.assert_allclose(f.absorbed_flux[up], full[up], atol=1e-6)
    assert (f.absorbed_flux[down] == 0.0).all()
    mid = ~(up | down)
    assert (f.absorbed_flux[mid] <= np.maximum(full[mid], 0.0) + 1e-9).all()


@pytest.mark.data
@pytest.mark.skipif(not _have_data(), reason="needs kernels and DEMs in LSS_DATA_DIR")
def test_distance_stays_in_range_over_a_year():
    """Sun distance over 2027 at daily steps spans Earth's perihelion to aphelion
    widened by the Moon's orbit (0.00257 AU): within [0.980, 1.020] AU, reaching
    below 0.984 and above 1.016."""
    t = np.arange(366) * 86400.0
    r = fm.sun_distance_au(*EQUATORIAL, 0.0, START, t)
    assert 0.980 <= r.min() < 0.984
    assert 1.016 < r.max() <= 1.020


@pytest.mark.data
@pytest.mark.skipif(not _have_data(), reason="needs kernels and DEMs in LSS_DATA_DIR")
def test_equator_facing_slope_gets_more_noon_flux(monkeypatch):
    """Southern point (45 S), zero horizon: a 20 deg slope facing north (aspect 0,
    equator-facing) gets more flux at the highest Sun than one facing south
    (aspect 180), by the closed-form ratio cos(z - s) / cos(z + s) to 1e-9,
    and more over the period. The noon Sun is within 1 deg of due north."""
    _zero_horizon(monkeypatch)
    kw = {"a0": 0.12, "slope_deg": 20.0}
    north = compute_forcing(*MIDLAT_SOUTH, 1.0, START, aspect_deg=0.0, **kw)
    south = compute_forcing(*MIDLAT_SOUTH, 1.0, START, aspect_deg=180.0, **kw)
    k = int(np.argmax(north.sun_elevation_deg))
    az = north.sun_azimuth_deg[k]
    assert min(az, 360.0 - az) < 1.0 + 360.0 * north.dt_s / SYNODIC_MONTH_S
    assert north.absorbed_flux[k] > south.absorbed_flux[k]
    assert north.absorbed_flux.mean() > south.absorbed_flux.mean()
    assert north.cos_incidence[k] == pytest.approx(
        float(cos_incidence(az, north.sun_elevation_deg[k], 20.0, 0.0)), abs=1e-12
    )


@pytest.mark.data
@pytest.mark.skipif(not _have_data(), reason="needs kernels and DEMs in LSS_DATA_DIR")
def test_horizon_gaps_are_reported(monkeypatch):
    """A horizon with missing samples warns and reports the count (exact)."""
    _zero_horizon(monkeypatch, missing=7)
    with pytest.warns(HorizonGapWarning, match="7 of"):
        f = compute_forcing(
            *EQUATORIAL, 1.0, START, a0=0.12, slope_deg=0.0, aspect_deg=0.0
        )
    assert f.horizon_missing_samples == 7


@pytest.mark.data
@pytest.mark.skipif(not _have_data(), reason="needs kernels and DEMs in LSS_DATA_DIR")
def test_dates_outside_kernel_coverage_raise():
    """Windows that start before 1900 or end after the orientation kernel's end
    (2051-01-01 TDB) raise OutOfCoverage (exact)."""
    from lss.ephem import OutOfCoverage

    with pytest.raises(OutOfCoverage, match="coverage"):
        compute_forcing(*EQUATORIAL, 1.0, datetime(1899, 6, 1, tzinfo=UTC), a0=0.12)
    with pytest.raises(OutOfCoverage, match="coverage"):
        compute_forcing(*EQUATORIAL, 1.0, datetime(2050, 12, 20, tzinfo=UTC), a0=0.12)


@pytest.fixture(scope="module", params=[POLAR, EQUATORIAL], ids=["85S90E", "5N60W"])
def demo(request):
    """compute_forcing at a demonstration point on real terrain (1 m, A0 0.12)."""
    if not _have_data():
        pytest.skip("needs kernels and DEMs in LSS_DATA_DIR")
    lat, lon = request.param
    with warnings.catch_warnings():
        warnings.simplefilter("error", HorizonGapWarning)
        return compute_forcing(lat, lon, 1.0, START, a0=0.12)


@pytest.mark.data
def test_visible_fraction_matches_compute_visibility(demo):
    """The forcing's Sun samples are compute_visibility's Sun output on the same
    grid, without its endpoint: fraction, azimuth and elevation exactly equal."""
    from lss.series import SeriesParams, compute_visibility

    end = demo.window_start + timedelta(seconds=demo.period_s)
    res = compute_visibility(
        demo.lat,
        demo.lon,
        1.0,
        demo.window_start,
        end,
        SeriesParams(grid_step_s=demo.dt_s),
    )
    g = res.sun.grid
    assert g.t_s.size == demo.n_samples + 1
    np.testing.assert_array_equal(demo.visible_fraction, g.fraction[:-1])
    np.testing.assert_array_equal(demo.sun_azimuth_deg, g.azimuth_deg[:-1])
    np.testing.assert_array_equal(demo.sun_elevation_deg, g.elevation_deg[:-1])
    assert demo.horizon_missing_samples == res.horizon.missing_samples == 0


@pytest.mark.data
def test_sun_only_call_matches_a_full_call(demo, monkeypatch):
    """compute_forcing asks the engine for the Sun alone (bodies=("SUN",)). Forced
    through a full call instead (both bodies), every per-sample array and the
    facet are identical (exact)."""
    import lss.series

    seen = {}

    def full(*args, **kwargs):
        seen["bodies"] = kwargs.pop("bodies", None)
        res = lss.series.compute_visibility(*args, **kwargs)
        assert res.sun is not None and res.earth is not None
        return res

    monkeypatch.setattr(fm, "compute_visibility", full)
    f = compute_forcing(demo.lat, demo.lon, 1.0, demo.window_start, a0=demo.a0)
    assert seen["bodies"] == ("SUN",)
    for name in (
        "t_s",
        "absorbed_flux",
        "visible_fraction",
        "sun_azimuth_deg",
        "sun_elevation_deg",
        "cos_incidence",
        "albedo",
        "distance_au",
    ):
        np.testing.assert_array_equal(
            getattr(f, name), getattr(demo, name), err_msg=name
        )
    assert f.facet == demo.facet
    assert f.horizon_missing_samples == demo.horizon_missing_samples


@pytest.mark.data
def test_distance_matches_the_engine_disc_radius(demo):
    """r from topocentric agrees with R_sun / sin(apparent radius) from the engine's
    series. Tolerance 1e-9 relative."""
    from lss.ephem import body_radius_km
    from lss.series import SeriesParams, compute_visibility

    end = demo.window_start + timedelta(seconds=demo.period_s)
    g = compute_visibility(
        demo.lat,
        demo.lon,
        1.0,
        demo.window_start,
        end,
        SeriesParams(grid_step_s=demo.dt_s),
    ).sun.grid
    r = body_radius_km("SUN") / np.sin(np.radians(g.radius_deg[:-1])) / fm.AU_KM
    np.testing.assert_allclose(demo.distance_au, r, rtol=1e-9)


@pytest.mark.data
def test_real_terrain_zeros_and_solver_contract(demo):
    """On real terrain: flux 0 exactly where the Sun is hidden or behind the facet,
    positive where both are favourable; output finite, non-negative, uniform,
    709 samples, no duplicated endpoint; solve_periodic accepts it and converges."""
    from lunar_thermal.solver import solve_periodic

    hidden = demo.visible_fraction == 0.0
    behind = demo.cos_incidence <= 0.0
    assert (demo.absorbed_flux[hidden | behind] == 0.0).all()
    assert (demo.absorbed_flux[~hidden & ~behind] > 0.0).all()
    assert hidden.any() and (~hidden).any()
    f = demo.absorbed_flux
    assert f.shape == (709,) and np.isfinite(f).all() and (f >= 0.0).all()
    np.testing.assert_allclose(np.diff(demo.t_s), demo.dt_s, atol=1e-6)
    assert demo.t_s[-1] + demo.dt_s == pytest.approx(demo.period_s, abs=1e-6)
    res = solve_periodic(f, demo.dt_s)
    assert res.converged


@pytest.mark.data
def test_slope_matches_the_engine_slope_statistics(demo):
    """The facet's slope equals the engine's slope at the point (lss.terrain.slope:
    slope_stats on the polar map, global_slope_stats on LDEM_128). Tolerance 1e-9
    deg. Source and baseline are recorded."""
    fc: Facet = demo.facet
    assert fc.source == "terrain"
    if demo.lat < -80.0:
        from lss.registry.terrain import POLAR_80M
        from lss.terrain.core import ps_from_latlon
        from lss.terrain.slope import slope_stats

        assert fc.layer == POLAR_80M.label and fc.resolution_m == 80.0
        x, y = (float(v) for v in ps_from_latlon(demo.lat, demo.lon, "south"))
        ref = slope_stats(POLAR_80M.dem().path, x, y, 200.0).at_point_deg
    else:
        from lss.terrain.global_layer import global_slope_stats

        assert fc.file_id == "ldem_128_img"
        ref = global_slope_stats(demo.lat, demo.lon, 1000.0).at_point_deg
    assert fc.slope_deg == pytest.approx(ref, abs=1e-9)
    assert fc.engine_slope_deg == fc.slope_deg
    assert all(b > 0 for b in fc.baseline_m)
    assert demo.terrain_layers[0]["role"] == "near_field"
    assert demo.terrain_layers[0]["label"] == fc.layer
