# lunar-thermal-surrogate

A 1D lunar regolith heat solver driven by Penumbra Luna's terrain-shadowed illumination, and a PhysicsNeMo Fourier neural operator (FNO) trained as its surrogate, evaluated on Penumbra Luna's 23 candidate landing sites.

Status: environment and scaffold only. Nothing below "Setup" is built yet.

## Question

_To be written._

## Data sources

Data is gitignored and never committed.

| Source | URL | Version | Access date | License |
|---|---|---|---|---|
| _To be filled in when data is added._ | | | | |

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
