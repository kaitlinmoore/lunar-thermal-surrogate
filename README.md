# lunar-thermal-surrogate

A 1D lunar regolith heat solver driven by Penumbra Luna's terrain-shadowed illumination, and a PhysicsNeMo Fourier neural operator (FNO) trained as its surrogate, evaluated on Penumbra Luna's 23 candidate landing sites.

Status: environment and scaffold only. Nothing below "Setup" is built yet.

## Question

_To be written._

## Data sources

Data is gitignored and never committed. It lives in the WSL filesystem at `LSS_DATA_DIR` (set in `.env`; default `$HOME/data/lunar-thermal-surrogate/lss-data`), with `data/` (terrain), `kernels/` (SPICE) and `MANIFEST.json` (URL, size, SHA-256 and retrieval time of every file). `cache/` there holds computed horizons.

**How it got there.** On 2026-11-23 it was copied once from Penumbra Luna's data folder (`C:\dev\GitHub\lss-data`; `data/`, `kernels/` and `MANIFEST.json`, not its `cache/`). That came to 44 files and 3,363,723,442 bytes (3.36 GB), most of it `LDEM_128.IMG` (2.12 GB). The copy matches the source in file count and bytes. `uv run python scripts/fetch_data.py --verify` re-hashes all 44 files against the manifest and reports every one `ok`. Every file also passes the catalog's leading-byte check (`lss.datafiles.looks_like`). To rebuild from the original sources instead, run `uv run python scripts/fetch_data.py` (the catalog is `src/lss/datafiles.py`). The access dates below are when Penumbra Luna downloaded each file, from `MANIFEST.json`.

| Source | URL | Version | Access date | License |
|---|---|---|---|---|
| NAIF SPICE generic kernels, DE421 era: `de421.bsp`, `moon_pa_de421_1900-2050.bpc`, `moon_080317.tf`, `moon_assoc_me.tf`, `naif0012.tls`, `pck00010.tpc` (Sun and Earth geometry in the Moon's mean-Earth frame) | https://naif.jpl.nasa.gov/pub/naif/generic_kernels/ | DE421 (Folkner et al. 2008); files as named. Lunar orientation covers 1900 to 2050 | 2026-11-15 | U.S. Government work, freely available |
| NASA GSFC PGDA product 90, south polar LOLA DEMs: `LDEM_80S_80MPP_ADJ` (80 m, ±304 km from the pole) and `LDEM_60S_240MPP_ADJ` (240 m, ±931 km). The horizon terrain for south-polar points | https://pgda.gsfc.nasa.gov/products/90 | as published with Barker et al. (2023), PSJ 4, 183, doi:10.3847/PSJ/acf3e1; data doi:10.60903/gsfcpgda-lola-spole | 2026-11-15 | NASA-produced data (U.S. Government work); no license stated; the PGDA Data Usage Policy asks users to cite the paper |
| LOLA global gridded DEM `LDEM_128` (`.IMG` + `.LBL`; 128 px/deg, 237 m). The horizon terrain off the polar maps | https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/img/ | LRO-L-LOLA-4-GDR-V1.0, product version 3.0 (2017-09-15); Neumann (2010), Smith et al. (2010) | 2026-11-16 | NASA PDS data (U.S. Government work) |
| NASA GSFC PGDA product 78, 5 m site DEMs (Site01, 04, 06, 07, 11, 20, 23, 42), RoI archives and the Site01 slope map. These tiles lie on held-out sites and are used by the engine only there | https://pgda.gsfc.nasa.gov/products/78 | as published with Barker et al. (2021), PSS 203, 105119, doi:10.1016/j.pss.2020.105119 | 2026-11-15 and 2026-11-16 | NASA-produced data; no license stated; PGDA asks users to cite the paper |
| PGDA product 90, clips cut by `fetch_data.py`: `LDEM_83S_10MPP_ADJ` (10 m, ±12 km or ±20 km around sites) and `LDEM_60S_60MPP_ADJ` (60 m, ±150 km around Site01). Copied with the rest; Penumbra Luna uses them for slope statistics and validation, and this repo doesn't use them yet | https://pgda.gsfc.nasa.gov/products/90 | as above | 2026-11-15 and 2026-11-16 | as above |
| PGDA product 69, average solar and Earth visibility (`AVGVISIB_85S_060M_201608`, `..._EARTH`), and the PDS LOLA `LDEM_75S_60M` clip. Copied with the rest; Penumbra Luna validation data, not used here | https://pgda.gsfc.nasa.gov/products/69 and https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/ | product 69 2016 release (Mazarico et al. 2011); LDEM_75S_60M product version 2.0 (2017) | 2026-11-15 and 2026-11-16 | NASA-produced data / NASA PDS data (U.S. Government work); cite Mazarico et al. 2011 |

Dataset details, citations and terms come from Penumbra Luna's dataset registry (`src/lss/registry/datasets.yaml`, copied with the engine).

## Vendored code

`src/lss/` is Penumbra Luna's illumination engine (Apache-2.0), copied unchanged from commit `aef153a` of https://github.com/kaitlinmoore/nasa-lunar-site-scout. It is never edited here: changes go to Penumbra Luna first and are then copied again. [`src/lss/VENDORED.md`](src/lss/VENDORED.md) records what was copied and left out, and why, plus a SHA-256 manifest that `tests/test_vendored.py` checks. `scripts/fetch_data.py` and the engine tests in `tests/lss/` come from the same commit. The repo as a whole is `MIT AND Apache-2.0`: this project's code is MIT (`LICENSE`), the vendored engine Apache-2.0 (`src/lss/LICENSE`).

Sun visibility at one point: `uv run python scripts/illumination_point.py LAT LON` (options: `--height`, `--start`, `--days`, `--out`). It writes one `.npz` outside the repo, by default to `$HOME/data/lunar-thermal-surrogate/derived/`.

## Setup

These steps take a fresh Windows machine with WSL2 (Ubuntu) to a working environment. The repo lives on the Windows drive at `C:\dev\GitHub\lunar-thermal-surrogate` (`/mnt/c/dev/GitHub/lunar-thermal-surrogate` from WSL). The Python environment lives in the WSL filesystem, because environments on `/mnt/c` are slow and would mix Windows and Linux files.

1. **GPU driver (Windows side).** Install a current NVIDIA driver on Windows. WSL2 reaches the GPU through the Windows driver. **Do not install a Linux NVIDIA driver inside WSL.** Inside WSL, `nvidia-smi` should list the GPU and report CUDA 13.0 or newer.
2. **Tools inside WSL.**
   - uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - direnv: `sudo apt install direnv`, then add `eval "$(direnv hook bash)"` to `~/.bashrc`.
   - git and gh, with git identity, `core.fileMode false` and `core.autocrlf false` set, and `gh auth login` done.
3. **Clone and enter the repo.**
   ```bash
   cd /mnt/c/dev/GitHub
   git clone https://github.com/kaitlinmoore/lunar-thermal-surrogate.git
   cd lunar-thermal-surrogate
   ```
4. **Environment variables.** `.envrc` points uv at `~/.venvs/lunar-thermal-surrogate` (or `~/.venvs/lunar-thermal-surrogate--<name>` in a worktree named `<name>`) and loads the main checkout's `.env`.
   ```bash
   cp .env.example .env    # then set LSS_DATA_DIR to an absolute path, e.g. /home/<you>/data/lunar-thermal-surrogate/lss-data
   direnv allow
   ```
   Check that `echo $UV_PROJECT_ENVIRONMENT` prints a path under `~/.venvs/`. If it doesn't, uv will create `.venv/` inside the repo instead. That is gitignored, but on the Windows drive it is slow. When running commands from a tool that doesn't load direnv, run `eval "$(direnv export bash)"` first.
5. **Python and dependencies.**
   ```bash
   uv python install 3.12
   uv sync
   ```
6. **Check the GPU.**
   ```bash
   uv run python scripts/env_report.py
   uv run pytest -m gpu
   ```
   The arch list must include `sm_120`, and both GPU tests must pass.

Worktrees go in `/mnt/c/dev/GitHub/lunar-thermal-surrogate-wt/<name>`. Run `direnv allow` once in each.

## Environment

Recorded 2026-11-23 from `scripts/env_report.py` and `nvidia-smi`. A second PhysicsNeMo project on this machine should match these versions.

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU, 16 GB, compute capability 12.0 (`sm_120`, Blackwell) |
| Windows NVIDIA driver | 621.01 (`nvidia-smi` 620.67.01 inside WSL) |
| CUDA supported by the driver | 13.5 |
| WSL | WSL 2.6.1.0, kernel 6.6.87.2-microsoft-standard-WSL2, Ubuntu 24.04.5 LTS |
| Python | 3.12.16 (via uv) |
| uv | 0.13.8 |
| PyTorch | 2.14.1+cu130, from PyTorch's cu130 index (CUDA 13.0 build; arch list `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`) |
| nvidia-physicsnemo | 2.3.0, core package with no CUDA extra |

How the CUDA stack is set up:

- **PyTorch** comes from `https://download.pytorch.org/whl/cu130`, declared in `pyproject.toml` as an explicit uv index and used only for `torch` (`[tool.uv.sources]`). The CUDA runtime libraries come as pip wheels alongside it, so WSL needs no CUDA toolkit.
- **PhysicsNeMo** is installed as the core package, without the `cu12` or `cu13` extra. Those extras add RAPIDS (cuML, pylibraft), CuPy and DALI, which a 1D FNO doesn't use. The `cu13` extra also pins `numpy<2.5`, which conflicts with the `numpy>=2.5.4` this repo needs. The lock contains no RAPIDS, CuPy or DALI packages.
- **torchvision** arrives through PhysicsNeMo's `timm` dependency and comes from PyPI (0.29.2+cu132), not the cu130 index. The FNO doesn't use it.

## How to run

_To be written as components land._

## Results

_None yet._

## Methodology

See [`docs/methodology.md`](docs/methodology.md) and [`docs/changelog.md`](docs/changelog.md).

## Limitations and scope

_To be written._ A known limit from the design: a 1D model with direct sunlight only omits scattered light and heat re-radiated by surrounding terrain, so shadowed polar sites come out too cold. Polar night minimums are not predictions.

## AI assistance

_To be written._ The environment setup and scaffold were written with AI assistance (Claude Code) and checked by running the GPU smoke tests on the target machine.
