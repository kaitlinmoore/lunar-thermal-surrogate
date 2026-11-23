"""Penumbra Luna's engine tests, run against the vendored copy in src/lss.

The test files under tests/lss/ are copied unchanged from Penumbra Luna at commit aef153a
(tests/engine/ -> tests/lss/engine/, tests/contract/ -> tests/lss/contract/), so they can be
re-copied whenever the engine is. This file is the only addition, and it is the only place that
adapts them to this repo. It does two things at collection time, by test name:

1. Marks the tests that read files under LSS_DATA_DIR with `data`, so `uv run pytest -m data`
   runs exactly those. Each one already skips itself when its files are missing (Penumbra Luna's
   own checks), so a machine without the data still passes `uv run pytest`.
2. Deselects the few tests that cannot pass in this repo for reasons outside the engine: they
   load Penumbra Luna's own scripts/ by a path relative to its repo root, need its web API, or
   assert where its pyproject.toml is. Each has its reason below. They are reported as
   "deselected" in the pytest summary, never silently dropped.

Whole files left out of the copy are listed, with reasons, in src/lss/VENDORED.md.
"""

from __future__ import annotations

import pytest

# (file name, test function name) -> needs data under LSS_DATA_DIR. Found by running the copied
# tests with LSS_DATA_DIR pointed at an empty folder: these are the ones that skip there.
DATA_TESTS: set[tuple[str, str]] = {
    # SPICE kernels
    ("test_ephem.py", "test_coverage_is_the_orientation_kernel_end"),
    ("test_ephem.py", "test_earth_radius_is_volumetric_mean"),
    ("test_ephem.py", "test_out_of_coverage_and_naive_times_rejected"),
    ("test_ephem.py", "test_pole_azimuth_is_the_limit_along_the_meridian"),
    ("test_ephem.py", "test_site01_year_is_physically_plausible"),
    ("test_ephem.py", "test_topocentric_matches_spice_azlcpo"),
    ("test_ephem.py", "test_utc_to_tdb_offset"),
    # LOLA LDEM_128 (global coarse layer)
    ("test_global_layer.py", "test_horizon_at_apollo_11_on_the_global_layer"),
    ("test_global_layer.py", "test_ldem_128_height_at_the_apollo_lunar_modules"),
    ("test_global_layer.py", "test_real_label_matches_what_the_reader_assumes"),
    ("test_slope_lonlat.py", "test_apollo_sites_on_ldem_128"),
    # PGDA product 78 site maps and product 90 polar maps
    ("test_horizon.py", "test_site01_cache"),
    ("test_horizon.py", "test_site01_kernel_matches_plain_numpy_rays"),
    ("test_slope.py", "test_reproduces_barkers_site01_slope_map"),
    ("test_slope.py", "test_site01_slope_at_the_served_point"),
    ("test_slope_lonlat.py", "test_mean_slope_agrees_with_the_polar_path_on_the_240m_map"),
    ("test_terrain.py", "test_layer_metadata_matches_files"),
    ("test_terrain.py", "test_site01_height_agrees_across_layers"),
}

# (file name, test function name) -> why it cannot pass here.
DESELECTED: dict[tuple[str, str], str] = {
    ("test_config.py", "test_default_env_file_is_repo_root"): (
        "asserts pyproject.toml sits at lss.config.REPO_ROOT; here REPO_ROOT is src/ (the package's parent)"
    ),
    ("test_global_layer.py", "test_locate_off_the_polar_maps_needs_no_dem_files"): (
        "drives Penumbra Luna's web API (lss.api, FastAPI), which is not copied"
    ),
    ("test_slope.py", "test_terrain_radii_use_the_decided_layers"): (
        "loads Penumbra Luna's scripts/precompute_terrain.py, which is not copied"
    ),
    ("test_slope.py", "test_artemis_regions_use_the_5m_map_only_where_their_point_lies_on_one"): (
        "loads Penumbra Luna's scripts/precompute_terrain.py, which is not copied"
    ),
    ("test_slope.py", "test_apollo_sites_use_the_global_map_with_the_approved_limitation"): (
        "loads Penumbra Luna's scripts/precompute_terrain.py, which is not copied"
    ),
    ("test_terrain_views.py", "test_artemis_views_use_the_5m_map_only_where_their_point_lies_on_one"): (
        "loads Penumbra Luna's scripts/terrain_views.py, which is not copied"
    ),
}


def _key(item: pytest.Item) -> tuple[str, str]:
    # originalname drops the [parameter] suffix, so one entry covers every parametrized case.
    return (item.path.name, getattr(item, "originalname", item.name))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    keep, dropped = [], []
    for item in items:
        key = _key(item)
        if key in DESELECTED:
            dropped.append(item)
            continue
        if key in DATA_TESTS:
            item.add_marker(pytest.mark.data)
        keep.append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = keep
