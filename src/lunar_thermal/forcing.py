"""Absorbed solar flux at a real point on the Moon: the solver's forcing.

For a point, an observer height and a UTC window start, `compute_forcing`
returns one period of absorbed solar flux on the local sloped surface, sampled
uniformly, ready for `lunar_thermal.solver.solve_periodic`. Every training
sample, and later every held-out site, is forced by it.

Physics (Hayne et al. 2017, JGR Planets 122, 2371-2400,
doi:10.1002/2017JE005387, Eqs. A8-A11 and Table A1)::

    F(t) = S (1 AU / r)^2  f  max(cos i, 0)  (1 - A(i))

* ``S = 1361 W m^-2``, the solar constant at 1 AU (Table A1).
* ``r``: the Sun's distance from the observer, from SPICE through the
  vendored engine (`lss.ephem.topocentric`). It differs from the Sun-Moon
  centre distance by at most the lunar radius, 1.2e-5 of 1 AU.
* ``f``: the visible, limb-darkened fraction of the solar disc over the real
  terrain horizon, from the engine (`lss.series.compute_visibility`, for the
  Sun alone: ``bodies=("SUN",)``, identical to the Sun of a full call). This
  module does not compute visibility.
* ``i``: the angle between the Sun's direction (disc centre) and the normal
  of the local ground facet, from the point's slope and aspect.
* ``A(i) = A0 + a (i / (pi/4))^3 + b (i / (pi/2))^8`` (Eq. A8), evaluated at
  the local incidence ``i`` (on a slope, not the Sun's zenith angle, as heat1d
  does), with ``a = 0.06``, ``b = 0.25`` (Table A1).

Two things can shadow the ground, and both are applied: the terrain horizon
(``f``: distant or nearby hills block the Sun) and the facet itself
(``max(cos i, 0)``: the Sun is behind the tilted ground even though it is
above the horizon).

Conventions (the engine's, `lss.ephem.core`; tested in `tests/test_forcing.py`):

* Latitude is planetocentric, longitude east, both in degrees.
* The local frame is east-north-up from the normal of the 1737.4 km reference
  sphere. Sun elevation is measured from that frame's horizontal plane, not
  from the sloped ground.
* Azimuth (Sun's, and the facet's aspect) is degrees clockwise from north,
  0-360.
* Slope is the angle between the ground facet and the horizontal plane.
  Aspect is the azimuth the facet faces: the downhill direction. On a
  southern-hemisphere point an equator-facing slope has aspect 0 (north).

Slope and aspect come from the engine (`lss.terrain.slope`, ``SLOPE_VERSION``
2), on the near-field terrain layer that its resolver
(`lss.registry.terrain.resolve`) picks for the point: `slope_aspect` on the
polar stereographic maps and `slope_aspect_lonlat` on the global map. Both
take central differences over the pixel containing the point. On a polar map
the baseline is two pixels in each direction (160 m on the 80 m map, 480 m
on the 240 m map). On the 237 m global map it is 474 m north-south and a
matched east-west baseline of ``2 n`` pixels, ``n = round(1 / cos(lat))``
(``n = 1`` below 48.19 deg), so slopes don't inflate near the poles. This
module only reads the pixels and applies two conventions: aspect 0 on a flat
facet (the engine gives NaN there), and a `ValueError` where the engine gives
no slope (on the global map, within 0.0081 deg, 246 m, of either pole).

Sampling: ``n_samples`` uniform samples over one period, ``t_k = k T / n``
for ``k = 0 .. n - 1`` (no duplicated endpoint). The engine evaluates the Sun
at exactly those instants (its internal grid is set to the sample step), so
each sample is an instantaneous value, not an average over the step. The
default is one mean synodic month (29.530589 d) at 709 samples, a step of
3598.65 s, the step closest to one hour that divides the period evenly.

Horizons are reused through the engine's cache (`lss.horizon.cached_horizon`,
in ``LSS_DATA_DIR/cache/horizon/``): the first window at a point and observer
height computes the horizon; later windows at the same point load it.

Written with AI assistance (Claude Code). The illumination itself is the
vendored Penumbra Luna engine in `src/lss`, unchanged.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lss.datafiles import get as datafile
from lss.ephem import MOON_RADIUS_M, OutOfCoverage, coverage_utc, topocentric, utc_to_et
from lss.horizon import HorizonParams
from lss.horizon.cache import cache_path, key_spec
from lss.registry.terrain import LAYERS, GlobalLayerSpec, LayerSpec, resolve
from lss.series import SeriesParams, compute_visibility
from lss.terrain.core import LonLatGrid, ps_from_latlon
from lss.terrain.global_layer import LDEM_128, window
from lss.terrain.slope import (
    ew_offset_px,
    scale_factor,
    slope_aspect,
    slope_aspect_lonlat,
    slope_deg,
    slope_deg_lonlat,
)

#: Solar constant at 1 AU, W m^-2 (Hayne et al. 2017 Table A1; Kopp and Lean 2011).
SOLAR_CONSTANT = 1361.0

#: Eq. A8 angle-dependence constants a and b (Hayne et al. 2017 Table A1).
ALBEDO_A = 0.06
ALBEDO_B = 0.25

#: Astronomical unit, km (IAU 2012 Resolution B2).
AU_KM = 149_597_870.7

#: Mean synodic month, days and seconds (as in scripts/illumination_point.py,
#: tests/synthetic_forcing.py and heat1d).
SYNODIC_MONTH_DAYS = 29.530589
SYNODIC_MONTH_S = SYNODIC_MONTH_DAYS * 86400.0

#: Default samples per period: the whole number of steps closest to one hour
#: in a synodic month (709, so dt = 3598.65 s; the solver asks for at least
#: hourly sampling, and 708 would give 3603.7 s).
DEFAULT_SAMPLES = round(SYNODIC_MONTH_S / 3600.0)

#: Where the vendored engine came from (src/lss/VENDORED.md).
ENGINE_SOURCE = (
    "github.com/kaitlinmoore/nasa-lunar-site-scout@1e45c30 (src/lss, unchanged)"
)


class HorizonGapWarning(UserWarning):
    """The horizon has ray samples no terrain layer covered (count in the result)."""


# --------------------------------------------------------------------------- physics


def albedo_eq_a8(
    incidence_rad: ArrayLike,
    a0: float,
    a: float = ALBEDO_A,
    b: float = ALBEDO_B,
) -> NDArray[np.float64]:
    """Bond albedo at solar incidence angle ``incidence_rad`` (Hayne et al. Eq. A8).

    ``A(i) = A0 + a (i / (pi/4))^3 + b (i / (pi/2))^8``. Incidence is clipped
    to [0, pi/2]: beyond grazing the flux is zero anyway, and the polynomial
    would otherwise exceed 1.
    """
    i = np.clip(np.asarray(incidence_rad, dtype=np.float64), 0.0, np.pi / 2.0)
    return a0 + a * (i / (np.pi / 4.0)) ** 3 + b * (i / (np.pi / 2.0)) ** 8


def cos_incidence(
    sun_azimuth_deg: ArrayLike,
    sun_elevation_deg: ArrayLike,
    slope_deg: float,
    aspect_deg: float,
) -> NDArray[np.float64]:
    """Cosine of the angle between the Sun and a facet's normal.

    The Sun's unit vector in east-north-up is (cos el sin az, cos el cos az,
    sin el); the facet normal of a slope s facing azimuth a (downhill) is
    (sin s sin a, sin s cos a, cos s). Their dot product is
    ``cos s sin el + sin s cos el cos(az - a)``. Negative values mean the Sun
    is behind the facet. Azimuths clockwise from north, degrees.
    """
    az = np.radians(np.asarray(sun_azimuth_deg, dtype=np.float64))
    el = np.radians(np.asarray(sun_elevation_deg, dtype=np.float64))
    s, a = math.radians(slope_deg), math.radians(aspect_deg)
    return math.cos(s) * np.sin(el) + math.sin(s) * np.cos(el) * np.cos(az - a)


def absorbed_flux(
    distance_au: ArrayLike,
    visible_fraction: ArrayLike,
    cos_i: ArrayLike,
    a0: float,
    a: float = ALBEDO_A,
    b: float = ALBEDO_B,
    solar_constant: float = SOLAR_CONSTANT,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Absorbed flux ``S (1/r)^2 f max(cos i, 0) (1 - A(i))``, W m^-2.

    Returns (absorbed flux, albedo). The albedo is Eq. A8 at the local
    incidence ``i = arccos(cos i)``, and is returned for every sample,
    including those where the flux is zero.
    """
    r = np.asarray(distance_au, dtype=np.float64)
    f = np.asarray(visible_fraction, dtype=np.float64)
    c = np.clip(np.asarray(cos_i, dtype=np.float64), -1.0, 1.0)
    alb = albedo_eq_a8(np.arccos(c), a0, a, b)
    flux = solar_constant / r**2 * f * np.maximum(c, 0.0) * (1.0 - alb)
    # max(c, 0) leaves -0.0 where c < 0; make every unlit sample a plain zero.
    return np.where(flux > 0.0, flux, 0.0), alb


def sample_times(n_samples: int, period_s: float) -> tuple[NDArray[np.float64], float]:
    """Uniform sample times ``k T / n``, ``k = 0 .. n - 1``, and the step ``T / n``."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if not (math.isfinite(period_s) and period_s > 0.0):
        raise ValueError("period_s must be positive")
    dt = period_s / n_samples
    return np.arange(n_samples) * dt, dt


# --------------------------------------------------------------------------- facet


@dataclass(frozen=True)
class Facet:
    """The ground facet the flux falls on, and where it came from.

    Attributes:
        slope_deg: Angle between the facet and the horizontal plane.
        aspect_deg: Azimuth the facet faces (downhill), clockwise from north;
            0 when the slope is 0.
        source: ``"terrain"`` (the engine's slope and aspect on the
            near-field DEM) or ``"supplied"`` (given by the caller).
        layer: Near-field layer label, e.g. ``"80 m polar map"``; empty when
            supplied.
        file_id: The layer's data-file id (`lss.datafiles`); empty when
            supplied.
        resolution_m: The layer's pixel size, m (north-south on the global
            map); NaN when supplied.
        baseline_m: Ground distance the engine's differences span at the
            point's pixel, (east-west, north-south) in m. Two pixels each way
            on a polar map; on the global map 2 R dphi north-south and
            2 n R cos(lat) dlambda east-west, with the engine's matched
            offset n. NaN when supplied.
        engine_slope_deg: The engine's slope-only function
            (`lss.terrain.slope.slope_deg` or `slope_deg_lonlat`) on the same
            pixel; equal to ``slope_deg`` exactly (checked on every call).
            NaN when supplied.
    """

    slope_deg: float
    aspect_deg: float
    source: str
    layer: str = ""
    file_id: str = ""
    resolution_m: float = math.nan
    baseline_m: tuple[float, float] = (math.nan, math.nan)
    engine_slope_deg: float = math.nan


def _conventions(slope: float, aspect: float, where: str) -> tuple[float, float]:
    """Apply this module's conventions to the engine's slope and aspect.

    The engine gives NaN aspect on a flat pixel; here a flat facet has
    aspect 0. NaN slope (no slope from the engine) or NaN aspect on a sloped
    pixel (a polar map pixel centred on the pole) raises ValueError.
    """
    if math.isnan(slope):
        raise ValueError(
            f"no slope at {where}: the engine's differences need neighbours "
            "outside the terrain read (lss.terrain.slope)"
        )
    if math.isnan(aspect):
        if slope == 0.0:
            return 0.0, 0.0
        raise ValueError(
            f"no aspect at {where}: east and north are undefined at the pole "
            "(lss.terrain.slope)"
        )
    return slope, aspect


def _ps_facet(
    z3: NDArray[np.float64], x_c: float, y_c: float, res: float, pole: str
) -> tuple[float, float, float, tuple[float, float]]:
    """Engine facet on a 3x3 polar stereographic window centred on the point's pixel.

    ``z3`` rows run down the map (row 0 is the largest y), columns along +x,
    as the GeoTIFF stores them; ``(x_c, y_c)`` is the centre pixel's centre.
    Slope and aspect are `lss.terrain.slope.slope_aspect` (east and north at
    the pixel's own position); the engine slope is `slope_deg` with the
    scale factor at that position. Returns (slope, aspect, engine slope,
    baseline) with the engine's NaN conventions; `_conventions` applies this
    module's.
    """
    s, a = slope_aspect(z3, res, np.array([[x_c]]), np.array([[y_c]]), pole)
    k = float(scale_factor(x_c, y_c))
    engine = float(slope_deg(z3, res, k)[0, 0])
    ground = 2.0 * res / k  # the engine's baseline: two pixels, map step / k
    return float(s[0, 0]), float(a[0, 0]), engine, (ground, ground)


def _lonlat_facet(
    image: ArrayLike, lat: float, lon: float, radius_m: float, label: str
) -> tuple[float, float, float, tuple[float, float]]:
    """Engine facet at (lat, lon) on a simple-cylindrical image (LDEM_128 geometry).

    Reads a window around the point with `lss.terrain.global_layer.window`,
    widened east and west by the engine's matched offset
    (`lss.terrain.slope.ew_offset_px`) at the most poleward latitude the
    point's pixel row can have, then hands it to `_lonlat_grid_facet`.
    ``image`` is anything `window` reads: the memory-mapped LDEM_128 image,
    or a synthetic array of the same shape in tests.

    Raises:
        ValueError: The window would span every longitude: the engine
            refuses such windows, since east and north are undefined at the
            pole. With a one-pixel radius that is within 0.0081 deg (246 m)
            of either pole: the pole row (1/128 deg), where the engine has no
            slope, plus about 10 m. Also as `_lonlat_grid_facet`.
    """
    half_px = 0.5 / LDEM_128.ppd
    n = int(min(ew_offset_px(min(abs(lat) + half_px, 90.0)), LDEM_128.samples))
    g = window(image, lat, lon, radius_m, LDEM_128, extra_cols=n - 1)
    if g.wrap:
        raise ValueError(
            f"no slope at ({lat}, {lon}) on {label}: too close to the pole, where "
            "the window spans every longitude; the engine refuses there, since east "
            "and north are undefined at the pole (lss.terrain.slope)"
        )
    return _lonlat_grid_facet(g, lat, lon, label)


def _lonlat_grid_facet(
    g: LonLatGrid, lat: float, lon: float, label: str
) -> tuple[float, float, float, tuple[float, float]]:
    """Engine facet of the pixel of ``g`` (a `LonLatGrid`) that contains (lat, lon).

    Slope and aspect are `lss.terrain.slope.slope_aspect_lonlat` on the three
    rows around the pixel, and the engine slope is `slope_deg_lonlat` on the
    same rows. Returns (slope, aspect, engine slope, baseline) with the
    engine's NaN conventions (`_conventions` applies this module's).

    Raises:
        ValueError: The pixel has no neighbour row in ``g``, or no data within
            the engine's east-west offset.
    """
    where = f"({lat}, {lon}) on {label}"
    r = math.floor((g.lat0 - lat) / g.res_deg)
    c = math.floor(((lon - g.lon0) % 360.0) / g.res_deg)
    z = np.asarray(g.array, dtype=np.float64)
    if not (1 <= r < z.shape[0] - 1 and 1 <= c < z.shape[1] - 1):
        raise ValueError(
            f"no slope at {where}: the pixel has no neighbour on every side"
        )
    lats = g.lat0 - (np.arange(r - 1, r + 2) + 0.5) * g.res_deg
    n = int(ew_offset_px(lats[1]))
    z3 = z[r - 1 : r + 2, :]
    if np.isnan(z3[:, max(c - n, 0) : c + n + 1]).any():
        raise ValueError(f"no terrain data around {where}")
    s, a = slope_aspect_lonlat(z3, g.res_deg, lats)
    engine = float(slope_deg_lonlat(z3, g.res_deg, lats)[0, c - 1])
    step = math.radians(g.res_deg) * MOON_RADIUS_M
    baseline = (2.0 * n * step * math.cos(math.radians(lats[1])), 2.0 * step)
    return float(s[0, c - 1]), float(a[0, c - 1]), engine, baseline


def terrain_facet(lat: float, lon: float, spec: LayerSpec) -> Facet:
    """Slope and aspect at (lat, lon) from terrain layer ``spec``, by the engine.

    The engine's slope at the point: the pixel containing the point, by
    central differences with its neighbours (`lss.terrain.slope`,
    ``SLOPE_VERSION`` 2). Reads only what that needs: a 3x3 block of a polar
    GeoTIFF, or a small window of the memory-mapped LDEM_128 image, widened
    for the matched east-west offset.

    On a polar map the baseline is two pixels each way. On the global map it
    is 474 m north-south and, east-west, ``2 n`` pixels with
    ``n = round(1 / cos(lat))`` (1 below 48.19 deg), so it stays between 2/3
    and 4/3 of the north-south one up to the poles.

    Raises:
        ValueError: The pixel or a neighbour has no data or lies on the
            map's edge, or the engine gives no slope there: on the global map
            within 0.0081 deg (246 m) of either pole (see
            `_lonlat_facet`; docs/methodology.md, "Forcing"). With the
            default layers only the north pole is affected: no polar map
            covers it, while the south pole lies on the 80 m map, whose
            pixel centres are never on the pole, so the engine gives a slope
            and aspect there.

    On a polar map the engine turns the map gradient into east and north at
    the pixel's centre, so the aspect is in that pixel's frame. It differs
    from the point's own frame (in which the Sun's azimuth is computed) by
    the angle the two subtend at the pole: 0.015 deg at 85 S 90 E, about
    3 deg within 1 km of the pole on the 80 m map.
    """
    if isinstance(spec, GlobalLayerSpec):
        from lss.terrain.global_layer import open_image

        img = open_image(datafile(spec.file_id).path, datafile(spec.label_file_id).path)
        # One pixel around the point; `window` adds its own 2-pixel margin.
        slope, aspect, engine, baseline = _lonlat_facet(
            img, lat, lon, float(spec.resolution_m), spec.label
        )
    else:
        import rasterio
        from rasterio.windows import Window

        dem = spec.dem()
        m = dem.meta  # header only; the raster is not loaded
        x, y = (float(v) for v in ps_from_latlon(lat, lon, spec.pole))
        c = math.floor((x - m["x0"]) / m["res"])
        r = math.floor((m["y0"] - y) / m["res"])
        if not (1 <= r < m["height"] - 1 and 1 <= c < m["width"] - 1):
            raise ValueError(f"({lat}, {lon}) is not inside {spec.label}'s interior")
        with rasterio.open(dem.path) as d:
            z3 = d.read(1, window=Window(c - 1, r - 1, 3, 3)).astype(np.float64)
        if m["nodata"] is not None and not np.isnan(m["nodata"]):
            z3[z3 == m["nodata"]] = np.nan
        z3 = z3 * m["scale"] + m["offset"]
        if np.isnan(z3).any():
            raise ValueError(f"no terrain data around ({lat}, {lon}) on {spec.label}")
        x_c = m["x0"] + (c + 0.5) * m["res"]
        y_c = m["y0"] - (r + 0.5) * m["res"]
        slope, aspect, engine, baseline = _ps_facet(z3, x_c, y_c, m["res"], spec.pole)
    slope, aspect = _conventions(slope, aspect, f"({lat}, {lon}) on {spec.label}")
    if slope != engine:
        raise RuntimeError(
            f"engine slope_aspect gives {slope} deg, its slope-only function {engine} deg"
        )
    return Facet(
        slope_deg=slope,
        aspect_deg=aspect,
        source="terrain",
        layer=spec.label,
        file_id=spec.file_id,
        resolution_m=float(spec.resolution_m),
        baseline_m=baseline,
        engine_slope_deg=engine,
    )


# --------------------------------------------------------------------------- forcing


@dataclass(frozen=True)
class Forcing:
    """One period of absorbed flux at a point, with everything that set it.

    Per-sample arrays all have length ``n_samples``; sample ``k`` is at
    ``window_start + t_s[k]`` seconds, with ``t_s[k] = k * dt_s``. Pass
    ``absorbed_flux`` and ``dt_s`` to `solve_periodic`.

    Attributes:
        lat, lon: Planetocentric latitude and east longitude, degrees.
        unit_height_m: Observer height above the local terrain, m.
        surface_height_m: Terrain height at the point above the 1737.4 km
            sphere (the horizon's near field), m.
        observer_height_m: Observer above the sphere: surface + unit height.
        window_start: UTC start (timezone-aware).
        period_s: Length of the period, s.
        dt_s: Sample step, ``period_s / n_samples``.
        t_s: Sample times, s since ``window_start``.
        absorbed_flux: W m^-2, finite and >= 0.
        sun_elevation_deg: Disc centre above the horizontal plane.
        sun_azimuth_deg: Clockwise from north.
        visible_fraction: Limb-darkened share of the disc above the terrain.
        cos_incidence: Cosine of the Sun's angle from the facet normal (may be
            negative: Sun behind the facet).
        albedo: Eq. A8 at the local incidence.
        distance_au: Sun distance, AU.
        facet: Slope, aspect and their source (`Facet`).
        a0, albedo_a, albedo_b, solar_constant: The flux constants used.
        terrain_layers: The horizon's terrain stack from the resolver, one
            dict per layer: label, file_id, resolution_m, role (near_field /
            far_field), min_range_m, max_range_m.
        fidelity_tier, fidelity_summary: The resolver's fidelity label.
        horizon_missing_samples: Ray samples no layer covered (0 for a fully
            supported point). A nonzero count also raises `HorizonGapWarning`.
        horizon_total_samples: All ray samples of the horizon.
        horizon_from_cache: Whether the horizon was already cached.
        timing_s: Wall-clock seconds: ``visibility`` (horizon, if not cached,
            plus the series), ``distance``, ``facet``, ``total``.
        engine: Where the vendored engine came from.
    """

    lat: float
    lon: float
    unit_height_m: float
    surface_height_m: float
    observer_height_m: float
    window_start: datetime
    period_s: float
    dt_s: float
    t_s: NDArray[np.float64]
    absorbed_flux: NDArray[np.float64]
    sun_elevation_deg: NDArray[np.float64]
    sun_azimuth_deg: NDArray[np.float64]
    visible_fraction: NDArray[np.float64]
    cos_incidence: NDArray[np.float64]
    albedo: NDArray[np.float64]
    distance_au: NDArray[np.float64]
    facet: Facet
    a0: float
    albedo_a: float
    albedo_b: float
    solar_constant: float
    terrain_layers: tuple[dict, ...]
    fidelity_tier: str | None
    fidelity_summary: str | None
    horizon_missing_samples: int
    horizon_total_samples: int
    horizon_from_cache: bool
    timing_s: dict = field(default_factory=dict)
    engine: str = ENGINE_SOURCE

    @property
    def n_samples(self) -> int:
        return int(self.t_s.size)

    @property
    def slope_deg(self) -> float:
        return self.facet.slope_deg

    @property
    def aspect_deg(self) -> float:
        return self.facet.aspect_deg

    @property
    def times(self) -> list[datetime]:
        """Sample times as UTC datetimes."""
        return [self.window_start + timedelta(seconds=float(s)) for s in self.t_s]


def sun_distance_au(
    lat: float,
    lon: float,
    observer_height_m: float,
    window_start: datetime,
    t_s: ArrayLike,
) -> NDArray[np.float64]:
    """Sun distance from the observer, AU, at ``window_start + t_s`` (engine's SPICE).

    Same time axis as the engine's series: ephemeris time
    ``et(window_start) + t_s`` (`lss.series.core.ephemeris_sampler`).
    """
    et0 = float(utc_to_et(window_start)[0])
    tp = topocentric(
        "SUN", lat, lon, observer_height_m, et0 + np.asarray(t_s, dtype=float)
    )
    return tp.distance_km / AU_KM


def compute_forcing(
    lat: float,
    lon: float,
    height_m: float,
    window_start: datetime,
    *,
    a0: float,
    n_samples: int = DEFAULT_SAMPLES,
    period_s: float = SYNODIC_MONTH_S,
    slope_deg: float | None = None,
    aspect_deg: float | None = None,
    horizon_params: HorizonParams | None = None,
    layers: tuple[LayerSpec, ...] = LAYERS,
    albedo_a: float = ALBEDO_A,
    albedo_b: float = ALBEDO_B,
    solar_constant: float = SOLAR_CONSTANT,
) -> Forcing:
    """One period of absorbed solar flux at (lat, lon) on the local sloped ground.

    Args:
        lat: Planetocentric latitude, degrees (south negative).
        lon: East longitude, degrees.
        height_m: Observer height above the local terrain, m (> 0). It sets
            the horizon's viewpoint; there is no default (open decision).
        window_start: UTC start of the period, timezone-aware.
        a0: Bond albedo at normal incidence (Eq. A8's A0).
        n_samples: Samples per period. Default: 709 over a synodic month,
            a step of 3598.65 s.
        period_s: Length of the period, s. Default: one mean synodic month.
        slope_deg, aspect_deg: Override the terrain facet (both or neither;
            for tests and controlled cases). By default both come from the
            near-field terrain layer (`terrain_facet`).
        horizon_params: The engine's horizon settings (default: its converged
            defaults). Part of the horizon cache key.
        layers: Terrain layers the resolver may choose from (default: all).
            With a custom list the engine may refuse the point and raise
            `lss.horizon.HorizonUnavailable` with the reason.
        albedo_a, albedo_b, solar_constant: Table A1 values by default.

    Returns:
        `Forcing`. Its horizon's missing-sample count is always set; when it
        is nonzero a `HorizonGapWarning` is raised too, so a gap is never
        silent. The dataset builder rejects such points.

    Raises:
        ValueError: Bad arguments (naive datetime, non-positive height or
            period, fewer than 2 samples, slope without aspect, an albedo
            above 1 at grazing incidence).
        lss.ephem.OutOfCoverage: The window is outside the SPICE kernels'
            coverage (1900 to 2050 for the DE421-era lunar orientation).
        lss.horizon.HorizonUnavailable: No terrain covers the point (custom
            ``layers`` only).
    """
    t_start = time.perf_counter()
    if window_start.tzinfo is None:
        raise ValueError("window_start must be timezone-aware (UTC)")
    window_start = window_start.astimezone(UTC)
    if not (math.isfinite(height_m) and height_m > 0.0):
        raise ValueError("height_m must be positive")
    if (slope_deg is None) != (aspect_deg is None):
        raise ValueError("give both slope_deg and aspect_deg, or neither")
    if a0 < 0.0 or a0 + albedo_a * 8.0 + albedo_b > 1.0:
        raise ValueError(
            f"A0 = {a0} gives an albedo outside [0, 1] at grazing incidence "
            f"(A(90 deg) = A0 + 8a + b = {a0 + 8 * albedo_a + albedo_b:.3f})"
        )
    t_s, dt = sample_times(n_samples, period_s)
    window_end = window_start + timedelta(seconds=period_s)
    lo, hi = coverage_utc()
    if window_start < lo or window_end > hi:
        raise OutOfCoverage(
            f"window {window_start.isoformat()} to {window_end.isoformat()} is outside "
            f"the ephemeris coverage {lo.isoformat()} to {hi.isoformat()} "
            "(DE421-era lunar orientation kernel)"
        )

    hparams = horizon_params or HorizonParams()
    stack = resolve(lat, lon, hparams.max_range_m, layers)
    # Whether the horizon is already cached (as scripts/illumination_point.py
    # checks): recorded for timing; compute_visibility loads or computes it.
    from_cache = (
        stack.supported
        and cache_path(key_spec(lat, lon, height_m, hparams, stack)).is_file()
    )

    # Visibility: the engine's own entry point, for the Sun alone (the Earth
    # is not computed; the Sun is identical to a full call's). Its internal
    # grid is set to the sample step, so its samples fall exactly on t_s plus
    # the endpoint (t = period), which is dropped.
    t0 = time.perf_counter()
    res = compute_visibility(
        lat,
        lon,
        height_m,
        window_start,
        window_end,
        SeriesParams(grid_step_s=dt),
        hparams,
        layers,
        bodies=("SUN",),
    )
    t1 = time.perf_counter()
    g = res.sun.grid
    if g.t_s.size != n_samples + 1:
        raise RuntimeError(
            f"engine grid has {g.t_s.size} samples, expected {n_samples + 1}"
        )
    if not np.allclose(g.t_s[:-1], t_s, rtol=0.0, atol=1e-6 * dt):
        raise RuntimeError("engine grid does not fall on the sample times")
    az = g.azimuth_deg[:-1]
    el = g.elevation_deg[:-1]
    frac = g.fraction[:-1]

    distance = sun_distance_au(lat, lon, res.observer_height_m, window_start, t_s)
    t2 = time.perf_counter()

    if slope_deg is None:
        facet = terrain_facet(lat, lon, stack.layers[0].spec)
    else:
        facet = Facet(float(slope_deg), float(aspect_deg) % 360.0, source="supplied")
    t3 = time.perf_counter()

    cos_i = cos_incidence(az, el, facet.slope_deg, facet.aspect_deg)
    flux, alb = absorbed_flux(
        distance, frac, cos_i, a0, albedo_a, albedo_b, solar_constant
    )

    hz = res.horizon
    if hz.missing_samples:
        warnings.warn(
            f"horizon at ({lat}, {lon}), {height_m} m has {hz.missing_samples} of "
            f"{hz.total_samples} ray samples with no terrain",
            HorizonGapWarning,
            stacklevel=2,
        )
    fid = stack.fidelity()
    return Forcing(
        lat=float(lat),
        lon=float(lon),
        unit_height_m=float(height_m),
        surface_height_m=float(hz.surface_height_m),
        observer_height_m=float(res.observer_height_m),
        window_start=window_start,
        period_s=float(period_s),
        dt_s=float(dt),
        t_s=t_s,
        absorbed_flux=flux,
        sun_elevation_deg=el,
        sun_azimuth_deg=az,
        visible_fraction=frac,
        cos_incidence=cos_i,
        albedo=alb,
        distance_au=distance,
        facet=facet,
        a0=float(a0),
        albedo_a=float(albedo_a),
        albedo_b=float(albedo_b),
        solar_constant=float(solar_constant),
        terrain_layers=tuple(
            {
                "label": sl.spec.label,
                "file_id": sl.spec.file_id,
                "resolution_m": float(sl.spec.resolution_m),
                "role": sl.role,
                "min_range_m": float(sl.min_range_m),
                "max_range_m": float(sl.max_range_m),
            }
            for sl in stack.layers
        ),
        fidelity_tier=fid.tier.value if fid else None,
        fidelity_summary=fid.summary if fid else None,
        horizon_missing_samples=int(hz.missing_samples),
        horizon_total_samples=int(hz.total_samples),
        horizon_from_cache=bool(from_cache),
        timing_s={
            "visibility": t1 - t0,
            "distance": t2 - t1,
            "facet": t3 - t2,
            "total": time.perf_counter() - t_start,
        },
    )
