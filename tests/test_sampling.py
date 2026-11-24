"""Tests for lunar_thermal.sample_sites: the training-point sampler and leakage guard.

The exclusion checks recompute distances with an independent formula (the angle
between unit vectors, from atan2 of their cross and dot products) rather than
the module's haversine, so a bug in one shows up as a disagreement. Statistical
tests use fixed seeds, so they are deterministic; each states its threshold.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from lss.registry.sites import load_sites
from lss.terrain.core import MOON_RADIUS_M
from lunar_thermal.sample_sites import (
    BANDS_V1,
    KERNEL_COVERAGE_UTC,
    MANIFEST_FIELDS,
    SAMPLER_VERSION,
    SITE_TILE_DATASET_ID,
    SYNODIC_MONTH_S,
    AnalysisPoint,
    Band,
    HeldOutSite,
    ManifestRow,
    SamplingConfig,
    _rng,
    band_counts,
    bands_from_json,
    check_rows,
    destination,
    draw_area_uniform,
    excluded_area,
    great_circle_distance_m,
    held_out_sites,
    nearest_site,
    read_manifest,
    sample_near_sites,
    sample_training_points,
    terrain_dataset_ids,
    validate_bands,
    write_analysis_manifest,
    write_manifest,
)

RADIUS = 50_000.0

#: Kolmogorov-Smirnov critical value at significance 0.001, asymptotic form:
#: reject uniformity if D > KS_C / sqrt(n). (c(alpha) = sqrt(-ln(alpha / 2) / 2).)
KS_C = math.sqrt(-math.log(0.001 / 2.0) / 2.0)  # 1.949


def independent_distance_m(lat1, lon1, lat2, lon2):
    """Great-circle distance from unit vectors: R * atan2(|a x b|, a . b)."""

    def unit(lat, lon):
        p, q = np.radians(lat), np.radians(lon)
        return np.stack(
            np.broadcast_arrays(
                np.cos(p) * np.cos(q), np.cos(p) * np.sin(q), np.sin(p)
            ),
            axis=-1,
        )

    a, b = unit(lat1, lon1), unit(lat2, lon2)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    return MOON_RADIUS_M * np.arctan2(cross, np.sum(a * b, axis=-1))


def min_site_distance_m(lat, lon, sites):
    lat, lon = np.asarray(lat)[:, None], np.asarray(lon)[:, None]
    slat = np.array([s.lat_deg for s in sites])[None, :]
    slon = np.array([s.lon_deg for s in sites])[None, :]
    return independent_distance_m(lat, lon, slat, slon).min(axis=1)


def ks_uniform(u):
    """Kolmogorov-Smirnov statistic of a sample against U(0, 1)."""
    u = np.sort(np.asarray(u))
    n = len(u)
    i = np.arange(1, n + 1)
    return max(np.max(i / n - u), np.max(u - (i - 1) / n))


@pytest.fixture(scope="module")
def sites():
    return held_out_sites()


@pytest.fixture(scope="module")
def draw(sites):
    """The v1 design at its planned size: 6,000 points, seed 0."""
    return sample_training_points(SamplingConfig(n_points=6000, seed=0), sites)


# --- Site list ---------------------------------------------------------------


def test_sites_come_from_the_registry(sites):
    """All 23 held-out sites, exactly as the registry gives them."""
    reg = load_sites()
    assert len(sites) == 23
    assert len({s.id for s in sites}) == 23
    assert [(s.id, s.lat_deg, s.lon_deg) for s in sites] == [
        (s.id, s.location.lat, s.location.lon) for s in reg
    ]


def test_sites_are_never_emitted_as_samples(draw, sites):
    """No training, development or analysis point sits on a site."""
    site_ids = {s.id for s in sites}
    lat = np.array([r.lat_deg for r in draw.rows])
    lon = np.array([r.lon_deg for r in draw.rows])
    assert min_site_distance_m(lat, lon, sites).min() > 0.0
    assert not site_ids & {r.point_id for r in draw.rows}
    pts = sample_near_sites(sites=sites)
    plat = np.array([p.lat_deg for p in pts])
    plon = np.array([p.lon_deg for p in pts])
    # Not at any site. (Rings around one site can pass close to another: Site20
    # and Mons Mouton are about 0.4 km apart.)
    assert min_site_distance_m(plat, plon, sites).min() > 1.0
    with pytest.raises(ValueError):
        sample_near_sites(rings_m=(0.0, 5e3), sites=sites)


# --- Distances on the sphere ------------------------------------------------------


def test_haversine_matches_the_independent_formula():
    rng = np.random.default_rng(1)
    lat1, lat2 = rng.uniform(-90, 90, (2, 5000))
    lon1, lon2 = rng.uniform(-180, 180, (2, 5000))
    d1 = great_circle_distance_m(lat1, lon1, lat2, lon2)
    d2 = independent_distance_m(lat1, lon1, lat2, lon2)
    np.testing.assert_allclose(d1, d2, rtol=0, atol=1e-3)
    # A quarter of a great circle, and the engine's radius.
    assert great_circle_distance_m(0, 0, 0, 90) == pytest.approx(
        math.pi / 2 * 1_737_400.0, abs=1e-6
    )


@pytest.mark.parametrize("bearing", [0.0, 45.0, 137.0, 270.0])
@pytest.mark.parametrize("dist", [1.0, 5e3, 50e3, 400e3])
def test_destination_lands_at_the_distance(bearing, dist):
    for lat, lon in [(0.67416, 23.47314), (-89.761966, -157.067197), (60.0, 179.9)]:
        lat2, lon2 = destination(lat, lon, bearing, dist)
        assert -180.0 <= lon2 < 180.0
        assert great_circle_distance_m(lat, lon, lat2, lon2) == pytest.approx(
            dist, abs=1e-4
        )


# --- Exclusion -----------------------------------------------------------------


def test_no_point_within_the_radius_large_draw(sites):
    """20,000 points (seed 7): every one at least 50 km from every site, by an
    independent distance formula; the recorded nearest site and distance agree."""
    res = sample_training_points(SamplingConfig(n_points=20_000, seed=7), sites)
    lat = np.array([r.lat_deg for r in res.rows])
    lon = np.array([r.lon_deg for r in res.rows])
    d = min_site_distance_m(lat, lon, sites)
    assert d.min() >= RADIUS
    recorded = np.array([r.nearest_site_distance_m for r in res.rows])
    np.testing.assert_allclose(recorded, d, rtol=0, atol=1e-3)
    _, idx = nearest_site(lat, lon, sites)
    assert [sites[i].id for i in idx] == [r.nearest_site_id for r in res.rows]
    # The south polar band is where the sites are: it must have rejected draws.
    stats = {s.band: s for s in res.band_stats}
    assert stats["south_polar"].n_rejected > 0
    assert all(s.n_drawn == s.n_points + s.n_rejected for s in res.band_stats)


def _row_at(lat, lon, **kw):
    base = {
        "point_id": "test-00000",
        "lat_deg": lat,
        "lon_deg": lon,
        "band": "mid",
        "split": "train",
        "H_m": 0.068,
        "albedo_A0": 0.12,
        "window_starts_utc": ("2030-01-01T00:00:00Z",),
        "nearest_site_id": "apollo-11",
        "nearest_site_distance_m": 1e9,  # deliberately wrong: the writer must recompute
        "exclusion_radius_m": RADIUS,
        "terrain_summary": "",
        "terrain_layers": "[]",
        "seed": 0,
        "sampler_version": SAMPLER_VERSION,
    }
    base.update(kw)
    return ManifestRow(**base)


@pytest.mark.parametrize("bearing", [0.0, 90.0, 200.0, 315.0])
def test_exclusion_boundary(sites, bearing, tmp_path):
    """A point 1 m inside a site's circle is excluded; one 1 m outside is not.

    Apollo 11 is used because no other site is within 300 km of it, so the
    point just outside its circle is outside every circle.
    """
    a11 = next(s for s in sites if s.id == "apollo-11")
    inside = destination(a11.lat_deg, a11.lon_deg, bearing, RADIUS - 1.0)
    outside = destination(a11.lat_deg, a11.lon_deg, bearing, RADIUS + 1.0)
    d_in, _ = nearest_site(*inside, sites)
    d_out, _ = nearest_site(*outside, sites)
    assert d_in[0] < RADIUS <= d_out[0]
    assert independent_distance_m(*inside, a11.lat_deg, a11.lon_deg) < RADIUS
    assert independent_distance_m(*outside, a11.lat_deg, a11.lon_deg) > RADIUS

    # The writer recomputes the distance and refuses the inside point...
    path = tmp_path / "m.csv"
    with pytest.raises(ValueError, match="exclusion radius"):
        write_manifest([_row_at(*inside)], path, sites=sites)
    assert not path.exists()
    # ...and accepts the outside one.
    write_manifest([_row_at(*outside)], path, sites=sites)
    assert len(read_manifest(path)) == 1


def test_sampler_rejects_inside_a_constructed_circle():
    """With one site at (0, 0) and a 500 km radius over a thin equatorial band,
    many candidates fall inside; none is kept."""
    site = (HeldOutSite("test-site", 0.0, 0.0),)
    band = (Band("eq", -1.0, 1.0, 1.0),)
    cfg = SamplingConfig(n_points=2000, seed=3, bands=band, exclusion_radius_m=500e3)
    res = sample_training_points(cfg, site)
    lat = np.array([r.lat_deg for r in res.rows])
    lon = np.array([r.lon_deg for r in res.rows])
    assert min_site_distance_m(lat, lon, site).min() >= 500e3
    # The circle spans about 33 of the band's 360 degrees of longitude.
    frac = res.band_stats[0].n_rejected / res.band_stats[0].n_drawn
    assert 0.06 < frac < 0.12


# --- Bands -----------------------------------------------------------------------


def test_v1_band_table():
    validate_bands(BANDS_V1)
    assert [(b.lat_min_deg, b.lat_max_deg, b.share) for b in BANDS_V1] == [
        (-90.0, -80.0, 0.50),
        (-80.0, -74.0, 0.10),
        (-74.0, 74.0, 0.30),
        (74.0, 90.0, 0.10),
    ]


@pytest.mark.parametrize("n", [6000, 1001, 7])
def test_band_shares_match_the_table(n):
    """Counts are allocated, not drawn: within one point of share * n, summing to n."""
    counts = band_counts(n, BANDS_V1)
    assert sum(counts.values()) == n
    for b in BANDS_V1:
        assert abs(counts[b.name] - b.share * n) < 1.0


def test_band_shares_in_the_draw(draw):
    for b in BANDS_V1:
        assert sum(r.band == b.name for r in draw.rows) == round(b.share * 6000)
        rows = [r for r in draw.rows if r.band == b.name]
        assert all(b.lat_min_deg <= r.lat_deg <= b.lat_max_deg for r in rows)


def test_band_table_from_json_without_code_changes(tmp_path):
    table = [
        {"name": "south", "lat_min_deg": -90, "lat_max_deg": -60, "share": 0.7},
        {"name": "rest", "lat_min_deg": -60, "lat_max_deg": 90, "share": 0.3},
    ]
    path = tmp_path / "bands.json"
    path.write_text(json.dumps(table))
    bands = bands_from_json(path)
    res = sample_training_points(SamplingConfig(n_points=100, bands=bands))
    assert sum(r.band == "south" for r in res.rows) == 70
    assert all(r.lat_deg <= -60 for r in res.rows if r.band == "south")


@pytest.mark.parametrize(
    "bands, message",
    [
        ((Band("a", -90, 0, 0.5), Band("b", -10, 90, 0.5)), "overlap"),
        ((Band("a", -90, 0, 0.5), Band("b", 0, 90, 0.4)), "sum"),
        ((Band("a", -90, 0, 0.5), Band("a", 0, 90, 0.5)), "duplicate"),
        ((Band("a", 10, 0, 1.0),), "latitudes"),
        ((Band("a", -90, 0, 1.0), Band("b", 0, 90, 0.0)), "positive"),
    ],
)
def test_invalid_band_tables_are_refused(bands, message):
    with pytest.raises(ValueError, match=message):
        validate_bands(bands)


# --- Area-uniform sampling -------------------------------------------------------


@pytest.mark.parametrize("band", BANDS_V1, ids=lambda b: b.name)
def test_area_uniform_within_a_band(band):
    """sin(latitude), rescaled to [0, 1], is uniform within the band, and so is
    longitude: Kolmogorov-Smirnov at significance 0.001 on 20,000 draws."""
    n = 20_000
    lat, lon = draw_area_uniform(
        _rng(11, "test", band.name), n, band.lat_min_deg, band.lat_max_deg
    )
    s0, s1 = (
        math.sin(math.radians(band.lat_min_deg)),
        math.sin(math.radians(band.lat_max_deg)),
    )
    u = (np.sin(np.radians(lat)) - s0) / (s1 - s0)
    assert ks_uniform(u) < KS_C / math.sqrt(n)
    assert ks_uniform((lon + 180.0) / 360.0) < KS_C / math.sqrt(n)


def test_area_uniform_test_has_power():
    """The same test rejects latitude drawn uniformly in degrees (which crowds
    points towards the pole) in the south polar band."""
    n = 20_000
    lat = np.random.default_rng(11).uniform(-90.0, -80.0, n)
    s0, s1 = -1.0, math.sin(math.radians(-80.0))
    u = (np.sin(np.radians(lat)) - s0) / (s1 - s0)
    assert ks_uniform(u) > KS_C / math.sqrt(n)


def test_sampler_is_area_uniform_with_no_exclusion(sites):
    """The full sampler with a zero radius: sin(latitude) uniform in every band
    (KS at 0.001). With a positive radius, points are uniform over the band
    minus the circles instead, so this check needs the radius off."""
    cfg = SamplingConfig(n_points=8000, seed=5, exclusion_radius_m=0.0)
    res = sample_training_points(cfg, sites)
    for b in BANDS_V1:
        lat = np.array([r.lat_deg for r in res.rows if r.band == b.name])
        s0, s1 = (
            math.sin(math.radians(b.lat_min_deg)),
            math.sin(math.radians(b.lat_max_deg)),
        )
        u = (np.sin(np.radians(lat)) - s0) / (s1 - s0)
        assert ks_uniform(u) < KS_C / math.sqrt(len(u)), b.name


def test_batching_does_not_change_the_draw():
    """Drawing in two batches gives the same points as one batch."""
    a = draw_area_uniform(_rng(2, "x"), 1000, -90, -80)
    r = _rng(2, "x")
    b1, b2 = draw_area_uniform(r, 400, -90, -80), draw_area_uniform(r, 600, -90, -80)
    np.testing.assert_array_equal(a[0], np.concatenate([b1[0], b2[0]]))
    np.testing.assert_array_equal(a[1], np.concatenate([b1[1], b2[1]]))


# --- Splits ----------------------------------------------------------------------


def test_splits_by_point_and_stratified(draw):
    ids = [r.point_id for r in draw.rows]
    assert len(set(ids)) == len(ids)
    assert {r.split for r in draw.rows} == {"train", "dev"}
    # Each point's windows, unrolled as the dataset builder will, share its split.
    split_of: dict[str, set[str]] = {}
    for r in draw.rows:
        for _ in r.window_starts_utc:
            split_of.setdefault(r.point_id, set()).add(r.split)
    assert all(len(s) == 1 for s in split_of.values())
    # Stratified: 15% of each band, rounded half up.
    for b in BANDS_V1:
        rows = [r for r in draw.rows if r.band == b.name]
        assert sum(r.split == "dev" for r in rows) == math.floor(0.15 * len(rows) + 0.5)


# --- Reproducibility --------------------------------------------------------------


def test_same_seed_same_manifest(draw, sites, tmp_path):
    again = sample_training_points(SamplingConfig(n_points=6000, seed=0), sites)
    assert again.rows == draw.rows
    p1 = write_manifest(draw.rows, tmp_path / "a.csv", sites=sites)
    p2 = write_manifest(again.rows, tmp_path / "b.csv", sites=sites)
    assert p1.read_bytes() == p2.read_bytes()
    other = sample_training_points(SamplingConfig(n_points=6000, seed=1), sites)
    assert other.rows != draw.rows
    assert all(r.seed == 0 and r.sampler_version == SAMPLER_VERSION for r in draw.rows)


def test_bands_draw_independently(draw, sites):
    """Changing one band's latitudes leaves the other bands' points unchanged."""
    bands = tuple(
        replace(b, lat_min_deg=80.0) if b.name == "north_polar" else b for b in BANDS_V1
    )
    res = sample_training_points(
        SamplingConfig(n_points=6000, seed=0, bands=bands), sites
    )
    for name in ("south_polar", "south_subpolar", "mid"):
        assert [r for r in res.rows if r.band == name] == [
            r for r in draw.rows if r.band == name
        ]
    assert [r for r in res.rows if r.band == "north_polar"] != [
        r for r in draw.rows if r.band == "north_polar"
    ]


# --- Windows and parameters -------------------------------------------------------


def test_windows_inside_2025_2045_and_kernel_coverage(draw):
    lo, hi = KERNEL_COVERAGE_UTC
    first, last = datetime(2025, 1, 1, tzinfo=UTC), datetime(2046, 1, 1, tzinfo=UTC)
    length = timedelta(seconds=SYNODIC_MONTH_S)
    for r in draw.rows:
        starts = r.window_start_datetimes()
        assert len(starts) == 4
        assert list(starts) == sorted(starts)
        for t in starts:
            assert first <= t < last  # starts on 2025-01-01 through 2045-12-31
            assert lo <= t and t + length <= hi
    every = [t for r in draw.rows for t in r.window_start_datetimes()]
    assert min(every).year == 2025 and max(every).year == 2045


def test_windows_past_kernel_coverage_are_refused():
    cfg = SamplingConfig(window_start_end=datetime(2050, 12, 30, tzinfo=UTC))
    with pytest.raises(ValueError, match="kernel coverage"):
        cfg.validate()


@pytest.mark.data
def test_kernel_coverage_constant_is_inside_the_kernels():
    """The constant the sampler checks against lies inside the kernels' real
    coverage (needs the SPICE kernels under LSS_DATA_DIR)."""
    from lss.ephem.core import KernelsMissing, coverage_utc

    try:
        lo, hi = coverage_utc()
    except KernelsMissing:
        pytest.skip("SPICE kernels not present")
    assert lo <= KERNEL_COVERAGE_UTC[0] and KERNEL_COVERAGE_UTC[1] <= hi


def test_parameters_inside_their_ranges(draw):
    h = np.array([r.H_m for r in draw.rows])
    a0 = np.array([r.albedo_A0 for r in draw.rows])
    assert h.min() >= 0.02 and h.max() <= 0.12
    assert a0.min() >= 0.05 and a0.max() <= 0.25
    # And they span the ranges (6,000 uniform draws).
    assert h.min() < 0.021 and h.max() > 0.119
    assert a0.min() < 0.052 and a0.max() > 0.248


# --- Terrain source ---------------------------------------------------------------


def test_every_point_records_its_terrain_stack(draw):
    for r in draw.rows:
        layers = json.loads(r.terrain_layers)
        assert layers and not r.terrain_summary.startswith("unsupported")
        assert layers[0]["role"] == "near_field" and layers[0]["min_range_m"] == 0.0
        assert layers[-1]["max_range_m"] == 400_000.0
        assert SITE_TILE_DATASET_ID not in terrain_dataset_ids(r.terrain_layers)
    south = [r for r in draw.rows if r.band == "south_polar"]
    assert all(json.loads(r.terrain_layers)[0]["resolution_m"] == 80.0 for r in south)


def test_writer_refuses_a_site_tile_stack(tmp_path, sites):
    layers = json.dumps([{"dataset_id": SITE_TILE_DATASET_ID}])
    with pytest.raises(ValueError, match="5 m map"):
        write_manifest(
            [_row_at(10.0, 10.0, terrain_layers=layers)],
            tmp_path / "m.csv",
            sites=sites,
        )


# --- Analysis-only points ---------------------------------------------------------


def test_analysis_points(sites):
    pts = sample_near_sites(sites=sites)
    assert len(pts) == 23 * 6 * 8
    assert all(isinstance(p, AnalysisPoint) for p in pts)
    assert all(p.analysis_only and p.split == "analysis" for p in pts)
    by_id = {s.id: s for s in sites}
    for p in pts:
        s = by_id[p.site_id]
        assert great_circle_distance_m(s.lat_deg, s.lon_deg, p.lat_deg, p.lon_deg) == (
            pytest.approx(p.ring_distance_m, abs=1e-3)
        )
    assert sample_near_sites(sites=sites) == pts  # reproducible


def test_analysis_points_cannot_reach_a_training_manifest(draw, sites, tmp_path):
    pts = sample_near_sites(sites=sites)
    path = tmp_path / "m.csv"
    with pytest.raises(TypeError):
        write_manifest(pts, path, sites=sites)
    with pytest.raises(TypeError):  # one mixed into real rows
        write_manifest([*draw.rows[:10], pts[0]], path, sites=sites)
    assert not path.exists()
    # A training row relabelled as analysis-only, or given another split, is refused too.
    for bad in (
        replace(draw.rows[0], analysis_only=True),
        replace(draw.rows[0], split="analysis"),
    ):
        with pytest.raises(ValueError):
            check_rows([bad], sites)
    # And the analysis writer refuses training rows.
    with pytest.raises(TypeError):
        write_analysis_manifest([draw.rows[0]], tmp_path / "a.csv")
    write_analysis_manifest(pts, tmp_path / "a.csv")
    assert not ({r.point_id for r in draw.rows} & {p.point_id for p in pts})


# --- Manifest round trip ----------------------------------------------------------


def test_manifest_round_trip(draw, sites, tmp_path):
    cfg = SamplingConfig(n_points=6000, seed=0)
    path = write_manifest(draw.rows, tmp_path / "manifest.csv", cfg, sites)
    assert path.read_text().splitlines()[0].split(",") == list(MANIFEST_FIELDS)
    assert read_manifest(path) == draw.rows
    meta = json.loads(path.with_suffix(".config.json").read_text())
    assert meta["config"]["seed"] == 0
    assert meta["config"]["exclusion_radius_m"] == RADIUS
    assert meta["excluded_site_ids"] == [s.id for s in sites]
    assert list(meta["schema"]) == list(MANIFEST_FIELDS)


def test_duplicate_ids_refused(draw, sites):
    with pytest.raises(ValueError, match="duplicate"):
        check_rows([draw.rows[0], draw.rows[0]], sites)


# --- Excluded area ----------------------------------------------------------------


def cap_area(r):
    return 2.0 * math.pi * MOON_RADIUS_M**2 * (1.0 - math.cos(r / MOON_RADIUS_M))


@pytest.mark.parametrize("lat", [0.0, -60.0, -89.9])
def test_excluded_area_of_one_circle(lat):
    """One circle wholly inside the band: its area is the spherical cap's,
    2 pi R^2 (1 - cos(r / R)), including a circle that covers the pole."""
    site = (HeldOutSite("s", lat, 30.0),)
    area, _ = excluded_area(max(-90.0, lat - 5.0), lat + 5.0, RADIUS, site)
    assert area == pytest.approx(cap_area(RADIUS), rel=1e-4)


def test_excluded_area_counts_overlap_once():
    one = (HeldOutSite("a", -45.0, 10.0),)
    twice = (*one, HeldOutSite("b", -45.0, 10.0))
    assert excluded_area(-90, 0, RADIUS, twice)[0] == pytest.approx(
        excluded_area(-90, 0, RADIUS, one)[0], rel=1e-9
    )
    # A circle straddling the band edge is split between the bands.
    edge = (HeldOutSite("e", -80.0, 0.0),)
    total = sum(excluded_area(a, b, RADIUS, edge)[0] for a, b in [(-90, -80), (-80, 0)])
    assert total == pytest.approx(cap_area(RADIUS), rel=1e-3)


def test_rejection_rate_matches_excluded_area(draw, sites):
    """The south polar band's rejected fraction estimates its excluded area:
    they agree within 4 binomial standard errors."""
    st = next(s for s in draw.band_stats if s.band == "south_polar")
    _, frac = excluded_area(-90.0, -80.0, RADIUS, sites)
    se = math.sqrt(frac * (1 - frac) / st.n_drawn)
    assert abs(st.n_rejected / st.n_drawn - frac) < 4 * se
