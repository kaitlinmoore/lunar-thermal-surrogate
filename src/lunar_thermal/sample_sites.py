"""Where on the Moon the training and development samples come from.

This module chooses the training and development points for the surrogate's
dataset and writes them to a manifest that the dataset builder reads. It is the
project's leakage guard: the 23 held-out sites are the test set, so no training
or development point may lie within the exclusion radius of any of them.

What it does, in order:

1. **Bands.** Points are split between latitude bands in fixed shares (a
   `Band` table, data rather than code branches). The v1 table, `BANDS_V1`,
   follows the latitude regimes of the held-out sites. A different table can be
   passed in, or loaded from JSON with `bands_from_json`, without changing code.
2. **Area-uniform draws within a band.** Candidates are uniform in
   sin(latitude) and in longitude, which is uniform by area on a sphere.
3. **Exclusion.** A candidate closer than the exclusion radius (great-circle
   distance on the engine's 1737.4 km sphere) to any of the registry's sites is
   rejected and another is drawn. Every kept point records its nearest site
   and the distance to it. Rejections are counted per band.
4. **Splits by point.** Each band's points are split 85/15 into train and dev.
   All of a point's windows belong to the point, so they share its split.
5. **Per-point parameters.** H and A0 are properties of the place, so they are
   drawn once per point; four lunation-long window starts per point.
6. **Terrain source.** Each point's terrain stack comes from the vendored
   resolver (`lss.registry.terrain.resolve`): which DEM layers its horizon will
   use, over which ray ranges. This only labels the point; horizons are the
   dataset builder's job.
7. **Manifest.** One CSV row per point (`MANIFEST_FIELDS`), plus a JSON file of
   the configuration. `write_manifest` re-checks the exclusion against the
   registry before it writes anything.

A separate, analysis-only sampler (`sample_near_sites`) places points in rings
at set distances around each site, for the dataset builder's study of horizon
similarity against distance, which sets the exclusion radius. Those points are
a different type (`AnalysisPoint`), carry ``analysis_only=True``, have no
train/dev split, and `write_manifest` refuses them.

Randomness: every draw comes from one integer seed. Each band gets its own
random streams for locations, splits, parameters and windows, keyed by the
band's name and the purpose (`_rng`). So the same seed and configuration always
give the same manifest, and changing one band's count leaves every other band's
points unchanged.

The held-out sites are only read for their coordinates. Nothing here computes a
horizon or a temperature anywhere.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lss.registry.sites import load_sites
from lss.registry.terrain import TerrainStack, resolve
from lss.terrain.core import MOON_RADIUS_M

#: Version of this sampler and its manifest schema. Recorded on every row. Bump
#: it when a change would give a different manifest for the same seed and
#: configuration, or when the schema changes.
SAMPLER_VERSION = "1"

#: Synodic month, s (29.530589 d): one lunation, the v1 window length. The same
#: value as `scripts/illumination_point.py` and the test forcing.
SYNODIC_MONTH_S = 29.530589 * 86400.0

#: The part of the SPICE kernels' coverage that every window must end inside,
#: UTC. The kernels cover 1899-12-31T23:59:18Z to 2050-12-31T23:58:50Z
#: (`lss.ephem.core.coverage_utc`, measured from the kernels: 1900-01-01 to
#: 2051-01-01 TDB). These bounds sit just inside that span, so they can be
#: checked without the kernel files; `tests/test_sampling.py` checks them
#: against the kernels when the data is present.
KERNEL_COVERAGE_UTC = (
    datetime(1900, 1, 1, tzinfo=UTC),
    datetime(2050, 12, 31, tzinfo=UTC),
)

#: The registry's dataset id for the PGDA product 78 5 m site maps. Those tiles
#: lie on held-out sites, so a training or development point must never resolve
#: to one. A 50 km radius already keeps every point far from them; the manifest
#: writer checks it anyway, in case the radius is made much smaller.
SITE_TILE_DATASET_ID = "pgda_78_site_dems"

SPLITS = ("train", "dev")
ANALYSIS_SPLIT = "analysis"


# --- Bands ----------------------------------------------------------------------


@dataclass(frozen=True)
class Band:
    """One latitude band of the sampling design.

    Attributes:
        name: Short identifier, used in point ids and as the key for the band's
            random streams. Must be unique within a table.
        lat_min_deg, lat_max_deg: Latitude range, degrees (planetocentric,
            south negative). Points fall in [lat_min_deg, lat_max_deg].
        share: Fraction of all points drawn from this band. A table's shares
            sum to 1.
        expected_terrain: Which DEM layers points in this band are expected to
            resolve to. Descriptive only: every point records its actual stack
            from the resolver.
        regime: Which held-out sites (if any) this band stands for. Descriptive.
    """

    name: str
    lat_min_deg: float
    lat_max_deg: float
    share: float
    expected_terrain: str = ""
    regime: str = ""


#: v1 band table (settled 2026-11-23). By area, about 10% of the mid-latitude
#: band (about 3% of all points) lies poleward of 60 degrees.
BANDS_V1: tuple[Band, ...] = (
    Band(
        "south_polar",
        -90.0,
        -80.0,
        0.50,
        "80 m polar map near field, 240 m polar map far field",
        "the 17 south polar sites (PGDA tiles and Artemis III candidates, 83.6-89.8 S)",
    ),
    Band(
        "south_subpolar",
        -80.0,
        -74.0,
        0.10,
        "240 m polar map (80 m near field in the 80 m map's corners, to about 75.8 S)",
        "the transition between the polar maps and the mid-latitudes; no site",
    ),
    Band(
        "mid",
        -74.0,
        74.0,
        0.30,
        "LOLA LDEM_128 (237 m); 240 m polar map near field with an LDEM_128 far "
        "field where 400 km rays leave the 240 m map (about 60-74 S, and its "
        "corners to about 48.5 S)",
        "the six Apollo sites (9.0 S to 26.1 N), near and far side",
    ),
    Band(
        "north_polar",
        74.0,
        90.0,
        0.10,
        "LOLA LDEM_128 (237 m)",
        "high northern latitudes; no site",
    ),
)


def validate_bands(bands: Sequence[Band]) -> None:
    """Raise ValueError unless the table is usable.

    Names unique; latitudes inside [-90, 90] with min < max; bands don't
    overlap (they may leave gaps); every share positive; shares sum to 1.
    """
    if not bands:
        raise ValueError("the band table is empty")
    names = [b.name for b in bands]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate band names: {names}")
    for b in bands:
        if not -90.0 <= b.lat_min_deg < b.lat_max_deg <= 90.0:
            raise ValueError(
                f"band {b.name}: latitudes must satisfy -90 <= min < max <= 90"
            )
        if not b.share > 0.0:
            raise ValueError(f"band {b.name}: share must be positive")
    ordered = sorted(bands, key=lambda b: b.lat_min_deg)
    for lo, hi in itertools.pairwise(ordered):
        if hi.lat_min_deg < lo.lat_max_deg:
            raise ValueError(f"bands {lo.name} and {hi.name} overlap")
    total = math.fsum(b.share for b in bands)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"band shares sum to {total}, not 1")


def bands_from_json(path: str | Path) -> tuple[Band, ...]:
    """Load a band table from JSON: a list of objects with `Band`'s fields.

    This is how the band table changes without a code change, for example::

        [{"name": "south_polar", "lat_min_deg": -90, "lat_max_deg": -80,
          "share": 0.5, "expected_terrain": "...", "regime": "..."}, ...]
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    bands = tuple(Band(**entry) for entry in raw)
    validate_bands(bands)
    return bands


def band_counts(n_points: int, bands: Sequence[Band]) -> dict[str, int]:
    """Points per band: each band's share of `n_points`, rounded so the counts
    sum to `n_points` exactly (largest remainder; ties go to the band listed
    first).

    The counts are fixed by the table rather than drawn, so the band shares in
    a manifest match the table to within one point.
    """
    validate_bands(bands)
    exact = [b.share * n_points for b in bands]
    counts = [math.floor(x) for x in exact]
    order = sorted(range(len(bands)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[: n_points - sum(counts)]:
        counts[i] += 1
    return {b.name: c for b, c in zip(bands, counts, strict=True)}


# --- Held-out sites and distances on the sphere ----------------------------------


@dataclass(frozen=True)
class HeldOutSite:
    """One held-out site: id and location, read from the vendored registry."""

    id: str
    lat_deg: float
    lon_deg: float


def held_out_sites() -> tuple[HeldOutSite, ...]:
    """The held-out sites, read from the registry (`lss/registry/sites.yaml`).

    The registry is the only source: coordinates are never copied by hand. It
    holds the project's 23 sites; whatever it holds is what is excluded.
    """
    return tuple(
        HeldOutSite(s.id, float(s.location.lat), float(s.location.lon))
        for s in load_sites()
    )


def great_circle_distance_m(
    lat1_deg: ArrayLike,
    lon1_deg: ArrayLike,
    lat2_deg: ArrayLike,
    lon2_deg: ArrayLike,
    radius_m: float = MOON_RADIUS_M,
) -> NDArray[np.float64]:
    """Great-circle distance between points on a sphere, m (broadcasts).

    Haversine formula, which stays accurate for short distances (a few metres)
    where the plain spherical law of cosines loses precision. The default
    radius is the illumination engine's reference sphere, 1737.4 km.
    """
    p1, p2 = np.radians(lat1_deg), np.radians(lat2_deg)
    dlat = p2 - p1
    dlon = np.radians(np.asarray(lon2_deg, float) - np.asarray(lon1_deg, float))
    h = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_m * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def nearest_site(
    lat_deg: ArrayLike,
    lon_deg: ArrayLike,
    sites: Sequence[HeldOutSite],
    radius_m: float = MOON_RADIUS_M,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Distance to the nearest site, m, and that site's index in `sites`,
    for each point."""
    lat = np.atleast_1d(np.asarray(lat_deg, float))
    lon = np.atleast_1d(np.asarray(lon_deg, float))
    slat = np.array([s.lat_deg for s in sites])
    slon = np.array([s.lon_deg for s in sites])
    d = great_circle_distance_m(lat[:, None], lon[:, None], slat, slon, radius_m)
    idx = np.argmin(d, axis=1)
    return d[np.arange(len(lat)), idx], idx


def destination(
    lat_deg: float,
    lon_deg: float,
    bearing_deg: float,
    distance_m: float,
    radius_m: float = MOON_RADIUS_M,
) -> tuple[float, float]:
    """The point `distance_m` along a great circle from (lat, lon), starting at
    `bearing_deg` (clockwise from north). Longitude in [-180, 180)."""
    p1, l1 = math.radians(lat_deg), math.radians(lon_deg)
    b, d = math.radians(bearing_deg), distance_m / radius_m
    sin_p2 = math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b)
    p2 = math.asin(max(-1.0, min(1.0, sin_p2)))
    l2 = l1 + math.atan2(
        math.sin(b) * math.sin(d) * math.cos(p1),
        math.cos(d) - math.sin(p1) * sin_p2,
    )
    return math.degrees(p2), _wrap_lon(math.degrees(l2))


def _wrap_lon(lon_deg: float) -> float:
    """Longitude in [-180, 180)."""
    return (lon_deg + 180.0) % 360.0 - 180.0


# --- Random streams and area-uniform draws ---------------------------------------


def _rng(seed: int, purpose: str, band: str = "") -> np.random.Generator:
    """An independent random stream for one purpose within one band.

    The key is a CRC-32 of "purpose:band" (stable across runs and machines,
    unlike Python's `hash`), mixed into the seed by NumPy's SeedSequence. So
    each band's locations, splits, parameters and windows come from their own
    streams, and changing one band's count or order never moves another's.
    """
    key = zlib.crc32(f"{purpose}:{band}".encode())
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(key,)))


def draw_area_uniform(
    rng: np.random.Generator, n: int, lat_min_deg: float, lat_max_deg: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """`n` points uniform by area between two latitudes: (lat, lon) in degrees.

    On a sphere the area between latitude phi and phi + dphi is proportional to
    cos(phi) dphi = d(sin phi). So sin(lat) uniform on [sin(min), sin(max)]
    and lon uniform on [-180, 180) is uniform by area. Drawing lat itself
    uniformly would crowd points towards the poles.

    Draws are taken from one (n, 2) block, row by row, so drawing in several
    batches gives the same points as one batch of the same total size.
    """
    u = rng.random((n, 2))
    s0, s1 = math.sin(math.radians(lat_min_deg)), math.sin(math.radians(lat_max_deg))
    lat = np.degrees(np.arcsin(np.clip(s0 + u[:, 0] * (s1 - s0), -1.0, 1.0)))
    lon = -180.0 + 360.0 * u[:, 1]
    return lat, lon


# --- Configuration and manifest rows ---------------------------------------------


@dataclass(frozen=True)
class SamplingConfig:
    """Everything that determines a manifest, besides the registry's sites.

    Attributes:
        n_points: Training plus development points, all bands.
        seed: The integer every random draw derives from.
        bands: The band table.
        exclusion_radius_m: A candidate closer than this to any held-out site
            is rejected. An instrument: 50 km is the starting value, to be set
            from the dataset builder's horizon-similarity study.
        dev_fraction: Share of each band's points assigned to dev.
        h_range_m: H (regolith scale height) is uniform on this range, m.
        albedo_range: A0 (normal bolometric albedo) is uniform on this range.
        n_windows: Window starts per point.
        window_start_min, window_start_end: Window starts are uniform over
            whole seconds in [min, end). The default end, 2046-01-01T00:00Z,
            makes 2045-12-31 the last start date.
        window_length_s: Length of each window, s (one lunation in v1). Used
            to check that every window ends inside the kernels' coverage.
    """

    n_points: int = 6000
    seed: int = 0
    bands: tuple[Band, ...] = BANDS_V1
    exclusion_radius_m: float = 50_000.0
    dev_fraction: float = 0.15
    h_range_m: tuple[float, float] = (0.02, 0.12)
    albedo_range: tuple[float, float] = (0.05, 0.25)
    n_windows: int = 4
    window_start_min: datetime = datetime(2025, 1, 1, tzinfo=UTC)
    window_start_end: datetime = datetime(2046, 1, 1, tzinfo=UTC)
    window_length_s: float = SYNODIC_MONTH_S

    def validate(self) -> None:
        """Raise ValueError if the configuration can't give a valid manifest."""
        validate_bands(self.bands)
        if self.n_points < 1:
            raise ValueError("n_points must be at least 1")
        if not self.exclusion_radius_m >= 0.0:
            raise ValueError("exclusion_radius_m must be non-negative")
        if not 0.0 <= self.dev_fraction <= 1.0:
            raise ValueError("dev_fraction must be in [0, 1]")
        for name, (lo, hi) in (
            ("h_range_m", self.h_range_m),
            ("albedo_range", self.albedo_range),
        ):
            if not 0.0 < lo < hi:
                raise ValueError(f"{name} must satisfy 0 < low < high")
        if self.n_windows < 1:
            raise ValueError("n_windows must be at least 1")
        lo, hi = KERNEL_COVERAGE_UTC
        last_end = self.window_start_end + timedelta(seconds=self.window_length_s)
        if self.window_start_min < lo or last_end > hi:
            raise ValueError(
                f"windows must lie inside kernel coverage {lo:%Y-%m-%d} to {hi:%Y-%m-%d}; "
                f"the latest window could end at {last_end:%Y-%m-%d}"
            )
        if self.window_start_end <= self.window_start_min:
            raise ValueError("window_start_end must be after window_start_min")

    def to_json_dict(self) -> dict:
        """The configuration as plain JSON values (for the manifest's config file)."""
        d = asdict(self)
        d["bands"] = [asdict(b) for b in self.bands]
        d["window_start_min"] = _iso(self.window_start_min)
        d["window_start_end"] = _iso(self.window_start_end)
        return d


@dataclass(frozen=True)
class ManifestRow:
    """One training or development point: one row of the sampling manifest.

    `MANIFEST_FIELDS` lists the columns, in this order, with units.
    """

    point_id: str
    lat_deg: float
    lon_deg: float
    band: str
    split: str  # "train" or "dev"
    H_m: float
    albedo_A0: float
    window_starts_utc: tuple[str, ...]  # ISO 8601, "YYYY-MM-DDTHH:MM:SSZ", ascending
    nearest_site_id: str
    nearest_site_distance_m: float
    exclusion_radius_m: float
    terrain_summary: str
    terrain_layers: str  # JSON list of the resolver's layers (see `_terrain_columns`)
    seed: int
    sampler_version: str
    analysis_only: bool = False

    def window_start_datetimes(self) -> tuple[datetime, ...]:
        """The window starts as timezone-aware UTC datetimes."""
        return tuple(_parse_iso(s) for s in self.window_starts_utc)


@dataclass(frozen=True)
class AnalysisPoint:
    """A near-site point for the horizon-similarity-against-distance study.

    Never a training or development point: its split is "analysis", it carries
    ``analysis_only=True``, and `write_manifest` refuses it. It has no H, A0 or
    windows, because the study needs only horizons.
    """

    point_id: str
    site_id: str  # the site whose ring this point is on
    ring_distance_m: float  # distance from that site, m
    bearing_deg: float  # from that site, clockwise from north
    lat_deg: float
    lon_deg: float
    nearest_site_id: str  # may differ from site_id where two sites are close
    nearest_site_distance_m: float
    terrain_summary: str
    terrain_layers: str
    seed: int
    sampler_version: str
    split: str = ANALYSIS_SPLIT
    analysis_only: bool = True


#: Manifest columns, in order: (name, type and units). This is the schema the
#: dataset builder reads; `read_manifest` checks a file's header against it.
MANIFEST_SCHEMA: tuple[tuple[str, str], ...] = (
    ("point_id", "str: '<band>-<5-digit index>', unique"),
    ("lat_deg", "float: planetocentric latitude, degrees, south negative"),
    ("lon_deg", "float: east longitude, degrees, [-180, 180)"),
    ("band", "str: band name from the band table"),
    ("split", "str: 'train' or 'dev'"),
    ("H_m", "float: regolith scale height H, m"),
    ("albedo_A0", "float: normal bolometric albedo A0"),
    ("window_starts_utc", "str: window starts, ISO 8601 UTC, ';'-separated, ascending"),
    ("nearest_site_id", "str: registry id of the nearest held-out site"),
    ("nearest_site_distance_m", "float: great-circle distance to it, m"),
    ("exclusion_radius_m", "float: the exclusion radius the point was drawn with, m"),
    ("terrain_summary", "str: the resolver's fidelity summary (near and far field)"),
    (
        "terrain_layers",
        (
            "str: JSON list of {dataset_id, label, role, resolution_m, min_range_m, "
            "max_range_m}, finest first"
        ),
    ),
    ("seed", "int: the seed the manifest was drawn from"),
    ("sampler_version", "str: SAMPLER_VERSION"),
    ("analysis_only", "bool: always 'false' in this manifest"),
)
MANIFEST_FIELDS: tuple[str, ...] = tuple(name for name, _ in MANIFEST_SCHEMA)

#: Columns of the analysis-only manifest, in order.
ANALYSIS_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(AnalysisPoint))


def _iso(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _terrain_columns(stack: TerrainStack) -> tuple[str, str]:
    """(summary, layers JSON) for a resolved terrain stack.

    The summary is the resolver's own fidelity text, e.g. "80 m polar map near
    field, 240 m polar map far field". The JSON keeps the whole stack: each
    layer's dataset, role and the ray range it is used over, so a point whose
    rays leave the 240 m map shows exactly where LDEM_128 takes over.
    """
    if not stack.supported:
        return f"unsupported: {stack.reason or 'no layer covers the point'}", "[]"
    layers = [
        {
            "dataset_id": sl.spec.dataset_id,
            "label": sl.spec.label,
            "role": sl.role,
            "resolution_m": sl.spec.resolution_m,
            "min_range_m": sl.min_range_m,
            "max_range_m": sl.max_range_m,
        }
        for sl in stack.layers
    ]
    fidelity = stack.fidelity()
    summary = fidelity.summary if fidelity is not None else ""
    return summary, json.dumps(layers, separators=(",", ":"))


def terrain_dataset_ids(terrain_layers: str) -> tuple[str, ...]:
    """The dataset ids in a row's `terrain_layers` JSON, finest first."""
    return tuple(layer["dataset_id"] for layer in json.loads(terrain_layers))


# --- Training and development sampling --------------------------------------------


@dataclass(frozen=True)
class BandStats:
    """What it took to fill one band: kept points, candidates drawn, and
    candidates rejected by the exclusion radius."""

    band: str
    n_points: int
    n_drawn: int
    n_rejected: int


@dataclass(frozen=True)
class SampleResult:
    """A training/development draw: the manifest rows and per-band statistics."""

    config: SamplingConfig
    rows: tuple[ManifestRow, ...]
    band_stats: tuple[BandStats, ...]
    site_ids: tuple[str, ...] = field(default=())


def _sample_band_locations(
    band: Band,
    n: int,
    sites: Sequence[HeldOutSite],
    radius_m: float,
    seed: int,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.intp], int
]:
    """Rejection sampling for one band.

    Draws area-uniform candidates in order and keeps the first `n` that are at
    least `radius_m` from every site. Returns the kept latitudes, longitudes,
    nearest-site distances and site indices, and how many candidates were drawn
    in all (kept plus rejected, up to and including the last kept one).

    The kept points are the first `n` acceptable candidates of the band's
    stream, whatever the batch size, so the result depends only on the seed.
    Accepted points are uniform by area over the band minus the exclusion
    circles: rejecting candidates that fall in a region leaves the rest
    uniform over what remains.
    """
    rng = _rng(seed, "location", band.name)
    lats, lons, dists, idxs = [], [], [], []
    kept = drawn = 0
    max_drawn = 1000 * n + 100_000  # a band almost wholly inside the circles
    while kept < n:
        need = n - kept
        batch = max(256, 2 * need)
        lat, lon = draw_area_uniform(rng, batch, band.lat_min_deg, band.lat_max_deg)
        d, i = nearest_site(lat, lon, sites)
        ok = np.flatnonzero(d >= radius_m)  # closer than the radius: rejected
        take = ok[:need]
        drawn += int(take[-1]) + 1 if len(take) == need else batch
        lats.append(lat[take])
        lons.append(lon[take])
        dists.append(d[take])
        idxs.append(i[take])
        kept += len(take)
        if kept < n and drawn > max_drawn:
            raise RuntimeError(
                f"band {band.name}: only {kept} of {n} points found after {drawn} draws; "
                "the exclusion circles cover almost all of it"
            )
    return (
        np.concatenate(lats),
        np.concatenate(lons),
        np.concatenate(dists),
        np.concatenate(idxs),
        drawn,
    )


def _round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def sample_training_points(
    config: SamplingConfig | None = None,
    sites: Sequence[HeldOutSite] | None = None,
) -> SampleResult:
    """Draw the training and development points.

    Args:
        config: The sampling configuration (default: `SamplingConfig()`, the v1
            design with seed 0).
        sites: The sites to exclude around. Default and normal use: the
            registry's held-out sites (`held_out_sites`). Tests pass their own.

    Returns:
        The manifest rows, band by band in table order, and per-band counts of
        candidates drawn and rejected.
    """
    config = config or SamplingConfig()
    config.validate()
    sites = tuple(sites) if sites is not None else held_out_sites()
    counts = band_counts(config.n_points, config.bands)
    span_s = int((config.window_start_end - config.window_start_min).total_seconds())

    rows: list[ManifestRow] = []
    stats: list[BandStats] = []
    for band in config.bands:
        n = counts[band.name]
        if n == 0:
            stats.append(BandStats(band.name, 0, 0, 0))
            continue
        lat, lon, dist, idx, drawn = _sample_band_locations(
            band, n, sites, config.exclusion_radius_m, config.seed
        )
        stats.append(BandStats(band.name, n, drawn, drawn - n))

        # Splits: a fixed number of dev points per band (stratified), chosen by
        # a seeded permutation. A point's windows are fields of its row, so
        # they can't end up in a different split from the point.
        n_dev = _round_half_up(config.dev_fraction * n)
        dev = np.zeros(n, bool)
        dev[_rng(config.seed, "split", band.name).permutation(n)[:n_dev]] = True

        # Place properties: one H and one A0 per point.
        prng = _rng(config.seed, "params", band.name)
        h = prng.uniform(*config.h_range_m, n)
        a0 = prng.uniform(*config.albedo_range, n)

        # Window starts: whole seconds, uniform over [start_min, start_end).
        offsets = _rng(config.seed, "windows", band.name).integers(
            0, span_s, (n, config.n_windows)
        )
        offsets.sort(axis=1)

        for k in range(n):
            stack = resolve(float(lat[k]), float(lon[k]))
            summary, layers = _terrain_columns(stack)
            starts = tuple(
                _iso(config.window_start_min + timedelta(seconds=int(s)))
                for s in offsets[k]
            )
            rows.append(
                ManifestRow(
                    point_id=f"{band.name}-{k:05d}",
                    lat_deg=float(lat[k]),
                    lon_deg=float(lon[k]),
                    band=band.name,
                    split="dev" if dev[k] else "train",
                    H_m=float(h[k]),
                    albedo_A0=float(a0[k]),
                    window_starts_utc=starts,
                    nearest_site_id=sites[int(idx[k])].id,
                    nearest_site_distance_m=float(dist[k]),
                    exclusion_radius_m=float(config.exclusion_radius_m),
                    terrain_summary=summary,
                    terrain_layers=layers,
                    seed=config.seed,
                    sampler_version=SAMPLER_VERSION,
                )
            )
    return SampleResult(config, tuple(rows), tuple(stats), tuple(s.id for s in sites))


# --- Analysis-only near-site sampler ----------------------------------------------

#: Default ring distances for the near-site study, m.
DEFAULT_RINGS_M: tuple[float, ...] = (5e3, 10e3, 20e3, 50e3, 100e3, 200e3)


def sample_near_sites(
    rings_m: Sequence[float] = DEFAULT_RINGS_M,
    bearings_per_ring: int = 8,
    seed: int = 0,
    sites: Sequence[HeldOutSite] | None = None,
) -> tuple[AnalysisPoint, ...]:
    """Points at set distances around each held-out site, for analysis only.

    Around each site, `bearings_per_ring` points on each ring, equally spaced
    in bearing. Each site gets one random rotation (from the seed), shared by
    all its rings, so the points at one bearing form a ray outward from the
    site: horizon similarity can be compared along the ray as well as against
    distance.

    These points lie inside the exclusion radius by design. They are a
    separate type from `ManifestRow`, have split "analysis" and
    ``analysis_only=True``, and `write_manifest` refuses them. No point is
    placed at a site itself (every ring distance must be positive).
    """
    if not rings_m or min(rings_m) <= 0.0:
        raise ValueError("ring distances must be positive")
    if bearings_per_ring < 1:
        raise ValueError("bearings_per_ring must be at least 1")
    sites = tuple(sites) if sites is not None else held_out_sites()
    out: list[AnalysisPoint] = []
    for site in sites:
        rotation = float(_rng(seed, "analysis_rotation", site.id).uniform(0.0, 360.0))
        step = 360.0 / bearings_per_ring
        for ring in rings_m:
            for j in range(bearings_per_ring):
                bearing = (rotation + j * step) % 360.0
                lat, lon = destination(site.lat_deg, site.lon_deg, bearing, ring)
                d, i = nearest_site(lat, lon, sites)
                summary, layers = _terrain_columns(resolve(lat, lon))
                out.append(
                    AnalysisPoint(
                        point_id=f"analysis-{site.id}-{ring / 1000:g}km-{j:02d}",
                        site_id=site.id,
                        ring_distance_m=float(ring),
                        bearing_deg=bearing,
                        lat_deg=lat,
                        lon_deg=lon,
                        nearest_site_id=sites[int(i[0])].id,
                        nearest_site_distance_m=float(d[0]),
                        terrain_summary=summary,
                        terrain_layers=layers,
                        seed=seed,
                        sampler_version=SAMPLER_VERSION,
                    )
                )
    return tuple(out)


# --- Manifest writing and reading ------------------------------------------------


def _format(value: object) -> str:
    """CSV cell text: floats as the shortest string that reads back exactly."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, tuple):
        return ";".join(value)
    return str(value)


def check_rows(
    rows: Iterable[object], sites: Sequence[HeldOutSite] | None = None
) -> tuple[ManifestRow, ...]:
    """The leakage checks `write_manifest` runs; raises on the first failure.

    Every row must be a `ManifestRow` (an `AnalysisPoint` is refused by type),
    not analysis-only, in train or dev, with a unique id. Its distance to every
    held-out site is recomputed from the registry, not taken from the row, and
    must be at least the row's exclusion radius. Its terrain stack must not use
    a 5 m site map (`SITE_TILE_DATASET_ID`), which lies on a held-out site.
    """
    sites = tuple(sites) if sites is not None else held_out_sites()
    rows = tuple(rows)
    seen: set[str] = set()
    for r in rows:
        if not isinstance(r, ManifestRow):
            raise TypeError(f"not a training/development row: {type(r).__name__}")
        if r.analysis_only or r.split not in SPLITS:
            raise ValueError(f"{r.point_id}: analysis-only or split {r.split!r}")
        if r.point_id in seen:
            raise ValueError(f"duplicate point id {r.point_id}")
        seen.add(r.point_id)
        if SITE_TILE_DATASET_ID in terrain_dataset_ids(r.terrain_layers):
            raise ValueError(
                f"{r.point_id}: terrain stack uses a held-out site's 5 m map"
            )
    if rows:
        d, i = nearest_site([r.lat_deg for r in rows], [r.lon_deg for r in rows], sites)
        radius = np.array([r.exclusion_radius_m for r in rows])
        bad = np.flatnonzero(d < radius)
        if len(bad):
            r = rows[int(bad[0])]
            raise ValueError(
                f"{r.point_id} is {d[bad[0]]:.1f} m from {sites[int(i[bad[0]])].id}, "
                f"inside the {r.exclusion_radius_m:.0f} m exclusion radius"
                f" ({len(bad)} rows in all)"
            )
    return rows


def write_manifest(
    rows: Iterable[ManifestRow],
    path: str | Path,
    config: SamplingConfig | None = None,
    sites: Sequence[HeldOutSite] | None = None,
) -> Path:
    """Write the training/development manifest as CSV, after `check_rows`.

    Nothing is written if any row fails. With `config`, the configuration is
    also written next to the CSV as `<name>.config.json`, with the registry
    site ids the exclusion used. Returns the CSV path.
    """
    sites = tuple(sites) if sites is not None else held_out_sites()
    rows = check_rows(rows, sites)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(MANIFEST_FIELDS)
        for r in rows:
            w.writerow(_format(getattr(r, name)) for name in MANIFEST_FIELDS)
    if config is not None:
        meta = {
            "sampler_version": SAMPLER_VERSION,
            "config": config.to_json_dict(),
            "excluded_site_ids": [s.id for s in sites],
            "schema": dict(MANIFEST_SCHEMA),
        }
        path.with_suffix(".config.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
    return path


def read_manifest(path: str | Path) -> tuple[ManifestRow, ...]:
    """Read a manifest written by `write_manifest` back into rows.

    Raises ValueError if the header isn't `MANIFEST_FIELDS`.
    """
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = tuple(next(reader))
        if header != MANIFEST_FIELDS:
            raise ValueError(f"unexpected manifest header: {header}")
        out = []
        for cells in reader:
            v = dict(zip(header, cells, strict=True))
            out.append(
                ManifestRow(
                    point_id=v["point_id"],
                    lat_deg=float(v["lat_deg"]),
                    lon_deg=float(v["lon_deg"]),
                    band=v["band"],
                    split=v["split"],
                    H_m=float(v["H_m"]),
                    albedo_A0=float(v["albedo_A0"]),
                    window_starts_utc=tuple(v["window_starts_utc"].split(";")),
                    nearest_site_id=v["nearest_site_id"],
                    nearest_site_distance_m=float(v["nearest_site_distance_m"]),
                    exclusion_radius_m=float(v["exclusion_radius_m"]),
                    terrain_summary=v["terrain_summary"],
                    terrain_layers=v["terrain_layers"],
                    seed=int(v["seed"]),
                    sampler_version=v["sampler_version"],
                    analysis_only=v["analysis_only"] == "true",
                )
            )
    return tuple(out)


def write_analysis_manifest(points: Iterable[AnalysisPoint], path: str | Path) -> Path:
    """Write the analysis-only points as CSV (`ANALYSIS_FIELDS`).

    Accepts only `AnalysisPoint`s, so a training row can't be mixed in either.
    """
    points = tuple(points)
    for p in points:
        if not isinstance(p, AnalysisPoint) or not p.analysis_only:
            raise TypeError(f"not an analysis-only point: {type(p).__name__}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(ANALYSIS_FIELDS)
        for p in points:
            w.writerow(_format(getattr(p, name)) for name in ANALYSIS_FIELDS)
    return path


# --- Exclusion area ---------------------------------------------------------------


def excluded_area(
    lat_min_deg: float,
    lat_max_deg: float,
    radius_m: float,
    sites: Sequence[HeldOutSite] | None = None,
    n_rows: int = 20_000,
    sphere_radius_m: float = MOON_RADIUS_M,
) -> tuple[float, float]:
    """Area between two latitudes that lies within `radius_m` of any site:
    (excluded area in m^2, excluded fraction of the band's area).

    Deterministic quadrature, exact in longitude. The band is cut into
    `n_rows` rows of equal area (equal steps in sin(latitude)). At each row's
    middle latitude phi, the circle of radius r around a site at (phi_s,
    lambda_s) covers the longitudes within dlambda of lambda_s, where, from the
    spherical law of cosines,

        cos(r / R) = sin(phi) sin(phi_s) + cos(phi) cos(phi_s) cos(dlambda).

    The union of the sites' intervals (overlapping circles count once) gives
    the row's excluded fraction. Only the rows' latitude steps are
    approximated, so the error shrinks as 1 / n_rows.
    """
    sites = tuple(sites) if sites is not None else held_out_sites()
    s0, s1 = math.sin(math.radians(lat_min_deg)), math.sin(math.radians(lat_max_deg))
    band_area = 2.0 * math.pi * sphere_radius_m**2 * (s1 - s0)
    if radius_m <= 0.0 or not sites:
        return 0.0, 0.0
    s = s0 + (np.arange(n_rows) + 0.5) * (s1 - s0) / n_rows
    phi = np.arcsin(s)[:, None]
    phs = np.radians([x.lat_deg for x in sites])[None, :]
    lam = np.array([x.lon_deg for x in sites])
    with np.errstate(divide="ignore", invalid="ignore"):
        c = (math.cos(radius_m / sphere_radius_m) - np.sin(phi) * np.sin(phs)) / (
            np.cos(phi) * np.cos(phs)
        )
    # c >= 1: the circle misses this latitude; c <= -1: it covers the whole row.
    half = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    half[~np.isfinite(c) | (c >= 1.0)] = 0.0
    half[c <= -1.0] = 180.0
    covered = np.zeros(n_rows)
    for k in np.flatnonzero(half.max(axis=1) > 0.0):
        covered[k] = _union_length(lam, half[k]) / 360.0
    frac = float(covered.mean())
    return frac * band_area, frac


def _union_length(centres_deg: NDArray, half_widths_deg: NDArray) -> float:
    """Total length, degrees, of the union of longitude intervals
    [c - h, c + h] on a circle of 360 degrees."""
    if half_widths_deg.max() >= 180.0:
        return 360.0
    intervals = []
    for c, h in zip(centres_deg, half_widths_deg, strict=True):
        if h <= 0.0:
            continue
        lo = (c - h) % 360.0
        hi = lo + 2.0 * h
        if hi > 360.0:  # wraps: split at 0
            intervals += [(lo, 360.0), (0.0, hi - 360.0)]
        else:
            intervals.append((lo, hi))
    intervals.sort()
    total, cur_lo, cur_hi = 0.0, None, None
    for lo, hi in intervals:
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None:
                total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    if cur_hi is not None:
        total += cur_hi - cur_lo
    return min(total, 360.0)


# --- Summary and command line ----------------------------------------------------


def summarize(result: SampleResult, sites: Sequence[HeldOutSite] | None = None) -> dict:
    """Per-band figures for a draw: points and share, dev count, minimum
    distance to any site, candidates rejected, and the band area the
    exclusion removes."""
    sites = tuple(sites) if sites is not None else held_out_sites()
    out = {}
    n_total = len(result.rows)
    for band, st in zip(result.config.bands, result.band_stats, strict=True):
        rows = [r for r in result.rows if r.band == band.name]
        area_m2, frac = excluded_area(
            band.lat_min_deg, band.lat_max_deg, result.config.exclusion_radius_m, sites
        )
        out[band.name] = {
            "points": st.n_points,
            "share": st.n_points / n_total if n_total else 0.0,
            "table_share": band.share,
            "dev": sum(r.split == "dev" for r in rows),
            "min_site_distance_km": (
                min(r.nearest_site_distance_m for r in rows) / 1000.0 if rows else None
            ),
            "drawn": st.n_drawn,
            "rejected": st.n_rejected,
            "rejected_fraction": st.n_rejected / st.n_drawn if st.n_drawn else 0.0,
            "excluded_area_km2": area_m2 / 1e6,
            "excluded_area_fraction": frac,
        }
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: draw a manifest, write it, and print the per-band summary."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out", type=Path, required=True, help="output folder (outside the repo)"
    )
    p.add_argument("--n-points", type=int, default=SamplingConfig.n_points)
    p.add_argument("--seed", type=int, default=SamplingConfig.seed)
    p.add_argument(
        "--radius-km", type=float, default=SamplingConfig.exclusion_radius_m / 1000.0
    )
    p.add_argument("--bands", type=Path, help="band table JSON (default: BANDS_V1)")
    p.add_argument(
        "--analysis",
        action="store_true",
        help="also write the analysis-only near-site points",
    )
    a = p.parse_args(argv)

    config = SamplingConfig(
        n_points=a.n_points,
        seed=a.seed,
        bands=bands_from_json(a.bands) if a.bands else BANDS_V1,
        exclusion_radius_m=a.radius_km * 1000.0,
    )
    sites = held_out_sites()
    result = sample_training_points(config, sites)
    path = write_manifest(result.rows, a.out / "sampling_manifest.csv", config, sites)
    print(f"wrote {len(result.rows)} points to {path}")
    print(json.dumps(summarize(result, sites), indent=2))
    if a.analysis:
        pts = sample_near_sites(seed=a.seed, sites=sites)
        apath = write_analysis_manifest(pts, a.out / "analysis_near_sites.csv")
        print(f"wrote {len(pts)} analysis-only points to {apath}")


if __name__ == "__main__":
    main()
