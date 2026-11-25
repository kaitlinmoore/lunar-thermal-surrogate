# Vendored: Penumbra Luna illumination engine

`src/lss/` is a copy of the illumination engine from Penumbra Luna, Kaitlin Moore's lunar landing-site tool. It is copied rather than imported, so this repo runs on its own.

| | |
|---|---|
| Source repo | https://github.com/kaitlinmoore/nasa-lunar-site-scout |
| Source commit | `1e45c30` (`1e45c30b62081d6153225f54142648a99ac95552`) |
| Copied | 2026-11-25, from the git blobs at that commit (`git cat-file blob 1e45c30:<path>`), so the bytes match the commit exactly |
| License | Apache-2.0, kept: `LICENSE` in this folder is Penumbra Luna's `LICENSE` at `1e45c30`, unchanged (identical since the first copy) |

## Copy history

| Date | Source commit | What changed in `src/lss` |
|---|---|---|
| 2026-11-23 | `aef153a` | first copy |
| 2026-11-23 | `9d0c648` | re-copied `registry/terrain.py`, `horizon/core.py` and `horizon/cache.py`; every other file is identical at both commits (checked file by file against the blobs) |
| 2026-11-25 | `1e45c30` | re-copied 18 files. Code changed in four: `terrain/slope.py`, `terrain/global_layer.py`, `series/core.py`, `series/__init__.py`. The other 14 changed in wording only: comments, docstrings, pydantic `Field` descriptions (`contract/models.py`) and `DataFile` notes (`datafiles.py`); checked by comparing each file's syntax tree with docstrings removed (and those two kinds of string masked), and the two YAML files parsed. `__init__.py`, `config.py`, `ephem/__init__.py`, `horizon/__init__.py`, `registry/__init__.py`, `registry/rois.py`, `terrain/__init__.py` and `LICENSE` are identical at both commits |

Upstream commits from `9d0c648` to `1e45c30` that touch the copied files or tests (none between `9d0c648` and `8089a50` do; `1e45c30` itself changes only the web API, which is not copied):

- `4d10e36`: matched east-west baseline for slope on latitude/longitude grids: the difference spans n = round(1 / cos(lat)) columns each side (`ew_offset_px`; n = 1 below 48.19°, where slopes are bit for bit those of before), so slopes near either pole no longer inflate from the 0.5 m height steps; no slope at the poles. `window(..., extra_cols=...)` and `image_slope_stats` in `terrain/global_layer.py`. Penumbra Luna methodology §5.6.1.
- `0d2314d`: tests only (docstrings of the full-ray-range and high-elevation tests).
- `4c86638`: slope with aspect (downhill azimuth, clockwise from north) and east/north gradients on both map types: `slope_aspect`, `slope_aspect_lonlat`, `gradient_ps`, `gradient_lonlat`, `aspect_deg`; the slopes equal `slope_deg` and `slope_deg_lonlat` exactly. Methodology §5.6.2.
- `dbb7526`: `compute_visibility(..., bodies=...)` computes the Sun or the Earth alone, identical to the same body in a full call; a body left out is `None` in the result, and `SeriesResult.require` raises where a metric needs it. `BODIES` exported from `lss.series`. Methodology §5.5.
- `1f6aeaa`: `SLOPE_VERSION = 2`.
- `a43573c`, `926198e`: wording only, in comments, docstrings and descriptions.

Upstream commits from `aef153a` to `9d0c648`:

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

Also copied unchanged from Penumbra Luna, outside this folder. Every copied file is identical to Penumbra Luna at `1e45c30`. At that copy, `test_ephem.py`, `test_global_layer.py`, `test_horizon.py`, `test_sites_apollo.py`, `test_sites_artemis.py`, `test_slope.py`, `test_slope_lonlat.py` and `test_terrain.py` were re-copied, most for wording only; `test_slope_lonlat.py` gained the matched-baseline tests, and `test_aspect.py` is new:

- `scripts/fetch_data.py`: downloads the engine's data into `LSS_DATA_DIR` from the catalog in `lss/datafiles.py`, so this repo can rebuild its data on its own. `--verify` re-hashes local files against `LSS_DATA_DIR/MANIFEST.json`.
- Engine tests, Penumbra Luna `tests/engine/` → `tests/lss/engine/` and `tests/contract/` → `tests/lss/contract/`:
  - engine: `test_aspect.py`, `test_ephem.py`, `test_global_layer.py`, `test_horizon.py`, `test_series.py`, `test_sites_apollo.py`, `test_sites_artemis.py`, `test_slope.py`, `test_slope_lonlat.py`, `test_terrain.py`, `test_terrain_views.py`
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
| `tests/engine/test_metrics.py`, `test_elevation_band.py`, `test_synthetic_e2e.py`, `test_high_elevation.py` | import `lss.metrics` (not copied); `test_high_elevation.py` uses `lss.metrics.core.azimuth_arc` for its zenith-pass tests. `test_metrics.py` also holds upstream's tests of `compute_visibility(..., bodies=...)`; here `tests/test_forcing.py` checks that the Sun-only call equals a full call at two points |
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
73be7685e75298733987fe46ddb29926af9874a9990fa3d1525bba3779c088c5  contract/__init__.py
a220b7667a84296d54669f511cdec97915328c0acaaceca4c477e6326a2739cc  contract/models.py
19a6866d4e16f01ac4e95acd2b3b779da1d57b344669233e59263a9944f07462  datafiles.py
609e5c7adda21fc8c9e39d5c90625a832101518776501ef1db2a022e9a8f2a3f  ephem/__init__.py
d8f8f4c2e5bb7de689e1ae3d05fa6c6f0828fafdd66aebe201b08e5a47c134e5  ephem/core.py
b6dc2cddc922041e461697f976ae99da1f8cb9b096322fcea5adef6bff8578d2  horizon/__init__.py
fdf2a98e6329093f9d795321b628fc1311c0cce81e4b74ac2d2fe09a4ea83830  horizon/cache.py
ea64bb1742ac8c94a9c0e2f1ca259d96e9b99d7e990eaf3b9a52fc8879d5d10b  horizon/core.py
1bc8174e60df4037312c31d633ddba7a5030a5dc1815bec99bef85b51f646ddd  registry/__init__.py
0b80dfce70d4eb8ad0534ec6eb456820d58ba3beea09b43662b33f5b3fadff19  registry/datasets.yaml
38004a39bb0ad160887ed332d8a52462111d905ae6cb53017727c44fc818a119  registry/loader.py
abf4aad6197c3e76042047c3ad261da3d827f309a5a7cf126ea534039915b839  registry/rois.py
19baeb94bddba8f909c6b480499054ce11d622a7d4b52871af7ea796019adea2  registry/sites.py
900fee9a0d1f92e9d7c68563e68e3d6a37048d3c639aae02ff0c575ecbc013a6  registry/sites.yaml
432a28c8e5c2261e2e59780b0f1dabde9139de021331baa2372945956f0cc0cf  registry/terrain.py
f2afa7c67839dd7a95264e096e64a1c1c1d7adf2f70e313c4f2b80b9bd5ff196  series/__init__.py
85d0bd016738666898d5f9bd96c1a850584abd0fc3c77d9803de7fa7d8f5458a  series/core.py
d22357572ffd1b7dfa0cfbdd22e0547e23c9f7a64b2c14743cfc29c9c11cbaa8  series/disc.py
3ed362d23b6819002cce88df4f97068532f5581cfab89a28cb75238c692991df  terrain/__init__.py
0cc889f1fe79b93537ad19bdbb27f05a5fecfaa9c43a821b3a8f8f469d64f076  terrain/core.py
7ee9a9b58437abde3ccdd9fb0824f278463019cc4e4b06a36461b2be1c0231e2  terrain/global_layer.py
f883ce377e32cc7908770f44fb438c532372a336ed13782920bd463d83ab1289  terrain/slope.py
3339eba0c9ee05333d02eb7b9856348227b40ff0bf0eaf2ee09051d047dbf568  terrain/views.py
```
