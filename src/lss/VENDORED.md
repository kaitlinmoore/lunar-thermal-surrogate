# Vendored: Penumbra Luna illumination engine

`src/lss/` is a copy of the illumination engine from Penumbra Luna, Kaitlin Moore's lunar landing-site tool. It is copied rather than imported, so this repo runs on its own.

| | |
|---|---|
| Source repo | https://github.com/kaitlinmoore/nasa-lunar-site-scout |
| Source commit | `9d0c648` (`9d0c6482d583e1f6601005b7ce0855b067bef82a`) |
| Copied | 2026-11-23, from the git blobs at that commit (`git cat-file blob 9d0c648:<path>`), so the bytes match the commit exactly |
| License | Apache-2.0, kept: `LICENSE` in this folder is Penumbra Luna's `LICENSE` at `9d0c648`, unchanged |

## Copy history

| Date | Source commit | What changed in `src/lss` |
|---|---|---|
| 2026-11-23 | `aef153a` | first copy |
| 2026-11-23 | `9d0c648` | re-copied `registry/terrain.py`, `horizon/core.py` and `horizon/cache.py`; every other file is identical at both commits (checked file by file against the blobs) |

Upstream commits between the two copies:

- `e43690e`: tests only. The whole and hidden disc fractions in `test_series.py::test_fraction_limits_and_half` are compared within 1e-12 instead of exactly (a GitHub-hosted runner gave 4.4e-19).
- `13f58a1`: the resolver (`registry/terrain.py`). Where 400 km rays leave the 240 m polar map (about 73°S to 60°S, and the map's corners north of 60°S), the map is used out to the distance it holds in every direction (`full_reach_m`, a closed-form cap test) and `LDEM_128` from there to 400 km. Without a global layer the point is refused, with the reason on the new `TerrainStack.reason` field, which `horizon_at` and `cached_horizon` now put in `HorizonUnavailable`.
- `9d0c648`: tests and methodology only. A new test of bodies high in the sky against exact sphere geometry (`tests/engine/test_high_elevation.py`), and Penumbra Luna methodology §3.1, §3.5 and §5.5.

## Rule: never edit `src/lss` here

Every file in this folder except `VENDORED.md` is byte-identical to the source commit. `tests/test_vendored.py` checks each one against the manifest below, so an accidental edit fails the test suite.

To change the engine, change it in Penumbra Luna first, then copy the changed files again from the new commit, update the commit and the manifest here, and log the re-copy in `docs/changelog.md`.

## How it runs here

- `lss/config.py` looks for `.env` at `REPO_ROOT`, which it takes as the folder above `lss/`. Here that is `src/`, which has no `.env`. That is fine: values already in the process environment win, and `.envrc` loads the main checkout's `.env` (with `LSS_DATA_DIR`) through direnv. Run `eval "$(direnv export bash)"` first in a shell that doesn't load direnv.
- `LSS_DATA_DIR` is this repo's own copy of Penumbra Luna's `data/` and `kernels/` (README, Data sources). The horizon cache is written to `LSS_DATA_DIR/cache/horizon/`.
- `lss/config.py`'s `FIXTURES_DIR` (`src/fixtures`) doesn't exist here. Only `contract/export.py` uses it, and that file is not copied.

## What was copied

| Path | Why |
|---|---|
| `__init__.py`, `config.py`, `datafiles.py` | package root, settings (`LSS_DATA_DIR`), and the catalog of data files the engine reads |
| `contract/__init__.py`, `contract/models.py` | pydantic models that `registry` and `series` import (`Location`, `FidelityLabel`, `Scenario`, ...). `models.py` imports nothing else from `lss` |
| `ephem/` | SPICE kernels, Sun and Earth positions, topocentric azimuth and elevation |
| `horizon/` | ray-cast terrain horizon (numba) and its cache |
| `terrain/` | DEM readers: polar stereographic maps, the global LDEM_128 window reader, slope, view helpers |
| `series/` | visibility series, events, and the limb-darkened visible disc fraction |
| `registry/` (Python files, `sites.yaml`, `datasets.yaml`) | terrain-layer resolver, site list, dataset registry |

## What was left out, and why

| Path | Why |
|---|---|
| `api/` | web API (FastAPI); not needed to compute illumination, and FastAPI is not a dependency here |
| `nl/` | natural-language question layer (calls Claude); unrelated to illumination |
| `metrics/` | power and communications summaries for the app; this project uses the raw series, not those summaries |
| `features/`, `search/`, `uncertainty/`, `thermal/` | placeholder packages at this commit (an `__init__.py` with a docstring only); nothing in the copied set imports them. `thermal/` is the app's slot for Diviner temperature lookups, unrelated to this repo's solver |
| `contract/examples.py`, `contract/export.py` | example responses and fixture export for the app's API contract; `export.py` imports `api/` |

No module in the copied set imports any left-out module (checked with grep over the import statements).

## Engine tests and the fetch script

Also copied unchanged from Penumbra Luna, outside this folder (identical at `aef153a` and `9d0c648`; the three engine test files that changed between them were re-copied from `9d0c648`):

- `scripts/fetch_data.py`: downloads the engine's data into `LSS_DATA_DIR` from the catalog in `lss/datafiles.py`, so this repo can rebuild its data on its own. `--verify` re-hashes local files against `LSS_DATA_DIR/MANIFEST.json`.
- Engine tests, Penumbra Luna `tests/engine/` → `tests/lss/engine/` and `tests/contract/` → `tests/lss/contract/`:
  - engine: `test_ephem.py`, `test_global_layer.py`, `test_horizon.py`, `test_series.py`, `test_sites_apollo.py`, `test_sites_artemis.py`, `test_slope.py`, `test_slope_lonlat.py`, `test_terrain.py`, `test_terrain_views.py`
  - contract: `conftest.py`, `test_config.py`, `test_registry.py`

`tests/lss/conftest.py` is this repo's own file. It marks the tests that read `LSS_DATA_DIR` with `data` and deselects six tests by name, with the reason for each written in the file:

- `test_config.py::test_default_env_file_is_repo_root`: asserts `pyproject.toml` sits at `REPO_ROOT`, which is `src/` here.
- `test_global_layer.py::test_locate_off_the_polar_maps_needs_no_dem_files`: drives the web API.
- Three tests in `test_slope.py` and one in `test_terrain_views.py`: load Penumbra Luna's own `scripts/`.

Engine test files left out:

| Penumbra Luna file | Why |
|---|---|
| `tests/engine/test_datafiles.py` | loads `scripts/fetch_data.py` from `REPO_ROOT` at import time; `REPO_ROOT` is `src/` here, so the whole file fails to collect |
| `tests/engine/test_horizon_lonlat.py` | imports `tests.engine.test_horizon`, a package path that only exists in Penumbra Luna's layout |
| `tests/engine/test_sites_mazarico_text.py` | compares Penumbra Luna's served results (its `cache/`, not copied) with published text; those tests can only skip here |
| `tests/engine/test_metrics.py`, `test_elevation_band.py`, `test_synthetic_e2e.py`, `test_high_elevation.py` | import `lss.metrics` (not copied); `test_high_elevation.py` uses `lss.metrics.core.azimuth_arc` for its zenith-pass tests |
| `tests/engine/test_regression_c1.py`, `test_regression_c3.py`, `test_validation_c1.py`, `test_validation_c3.py`, `test_validation_horizons.py`, `test_sites.py` | import `lss.contract.examples` or `lss.metrics`, or load Penumbra Luna's `scripts/`, or compare against its served results |
| `tests/engine/test_horizon_convergence_script.py`, `test_series_checks_script.py`, `test_iau_names.py` | test Penumbra Luna's own scripts, which are not copied |
| `tests/contract/test_fixtures.py` | tests the app's API contract fixtures (`contract/examples.py`, `contract/export.py`, `fixtures/`) |
| `tests/api/`, `tests/nl/` | the web API and the natural-language layer |

## Manifest

SHA-256 of every copied file (paths relative to `src/lss/`). `tests/test_vendored.py` reads this block.

```text
c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4  LICENSE
0350ccf6bcbfb7811b3ddef6f4ef5867535e83d236829068338eaf38ad2bde65  __init__.py
f765c47bcce521927b6e56e93a0cf1002ad8de7b34501d854bf7f8156934feaa  config.py
39c2517200381a0089b948ed8194ff3b7fb2b8129cacfb1d936fd603125763b0  contract/__init__.py
f178c37c672c6a3f327f99d917298a6a1debff7c11f87f4341a8db6f05d9bbcb  contract/models.py
ca10806b1e45f330fbe3fb1c0b58d1580329a284895f2b54feaed2f9eebeb023  datafiles.py
609e5c7adda21fc8c9e39d5c90625a832101518776501ef1db2a022e9a8f2a3f  ephem/__init__.py
f660cabc0efc855dfe1e96b4a00305a4ffb509698fce4066b5dd6d62a5b44bc1  ephem/core.py
b6dc2cddc922041e461697f976ae99da1f8cb9b096322fcea5adef6bff8578d2  horizon/__init__.py
a31cd57c58de66ff7c4a579b27b711de1bda595d6695269cd6e3227c3f6d2c77  horizon/cache.py
ba1aef79cbf37c001f48a7374c3cc4f5a069559d976caefb4958e0b4a47cf461  horizon/core.py
1bc8174e60df4037312c31d633ddba7a5030a5dc1815bec99bef85b51f646ddd  registry/__init__.py
0873e6d0ab2c2a6ff9bc3c2ef588a4e134e8f95592cb3b5c0446249c2ec3f07f  registry/datasets.yaml
6b3bf0659880557fa820205bcf238cd8316caf87d9f2a484db17b7d9dedbeb9c  registry/loader.py
abf4aad6197c3e76042047c3ad261da3d827f309a5a7cf126ea534039915b839  registry/rois.py
f6fb50fa498738c4c85476b9b5933e734acdca1321a3e574d928f31a69d59041  registry/sites.py
b816e2762c2b857ba7e7e653a2177d7b4202424dc5df052707e96f1565c422b2  registry/sites.yaml
719df26dab2884ab933dec1b4b18f6f9d2b76119031b85971a8dc557a1618ee8  registry/terrain.py
a19ee0f2212be519dd817412cd00aaccf9f0128f0afc8b25138277a02c2bb737  series/__init__.py
65548693a1b6f96fb29b73a5d83629cf8c3f9da0edc246caf43e45d1bdfc4eda  series/core.py
dd809c36698482c535c075e0472075936d6121686e676b67fdf11c1fc2ab6b90  series/disc.py
3ed362d23b6819002cce88df4f97068532f5581cfab89a28cb75238c692991df  terrain/__init__.py
cddddf9d32f641839f73ad1fa237f2d064f0d26018fe96c57a15e0d78567f0b4  terrain/core.py
11083501c6f63e1a17191107a25ad3d477c5b2ae261e66c1552c82bebbde96b2  terrain/global_layer.py
8b91de4709930bd68359355dc01c81fffef8b689203937180ea1c428927519ff  terrain/slope.py
a265913fb053070b99ad624887886129296ce7060a4b5f89e62e2541af72dda0  terrain/views.py
```
