# CLAUDE.md: lunar-thermal-surrogate
Standard: Research Project Standard v2

## What this is
A 1D regolith heat solver driven by Penumbra Luna's terrain-shadowed illumination, and a PhysicsNeMo Fourier neural operator trained as its surrogate, evaluated on the 23 Penumbra Luna sites.

## Principles
- AI may write any part of this project. I decide what I write myself; don't assign work to me.
- Every component gets clear comments and docstrings, a write-up in learning/ (gitignored), and a practice notebook in practice/ where useful.
- Work is exploratory. Instruments can change freely; log each change in docs/changelog.md with its reason. Leave the held-out split unexamined during development. Freeze only before a confirmatory run (docs/frozen_config.md).
- Before a change that could affect earlier results, frozen configs or held-out data, tell me what it touches and the tradeoffs, then let me decide.
- Propose a plan and wait for my approval before large changes, restructures or deletions.
- Honest framing: separate what I implemented from library behavior, disclose AI assistance, state limits, report nulls.
- Tests for deterministic logic, against fixtures in tests/fixtures/.

## Repo conventions
- Data stays out of git. The README records data sources with URLs, versions, access dates and licenses.
- Secrets in .env; .env.example lists key names only.

## Environment
WSL2 (Ubuntu): PhysicsNeMo supports Windows only through WSL. Claude Code runs inside WSL. The repo lives at C:\dev\GitHub\lunar-thermal-surrogate (/mnt/c/dev/GitHub/lunar-thermal-surrogate from WSL); the uv environment lives in the WSL filesystem at $HOME/.venvs/lunar-thermal-surrogate, set by UV_PROJECT_ENVIRONMENT in .envrc. uv for all Python. Python version and CUDA backend: set by the environment setup step and recorded in the README.
GPU: RTX 5080 Laptop GPU (Blackwell, sm_120, 16 GB). PyTorch must be built for CUDA 12.8 or newer; `torch.cuda.get_arch_list()` must include sm_120. Never install a Linux NVIDIA driver inside WSL.
Key deps: nvidia-physicsnemo, torch, numpy, numba; heat1d for tests and triangulation only; the Penumbra Luna illumination code is copied in, not imported.

## Project-specific
- Design document: `lunar_thermal_surrogate_planning_package.md` at the repo root (local only, gitignored). Its locked decisions (section 2) stay locked unless I reopen them.
- The 23 Penumbra Luna sites are the held-out test set. Never sample training or development points within the exclusion radius of a site, and don't evaluate on the sites until a freeze.
- Polar results from a 1D direct-sunlight model omit scattered light and terrain re-radiation; never present polar night minimums as predictions.

## Working from a handoff
Tasks on this project come as handoff files in `handoffs/`, which exists only on my machine and is gitignored. State doc: `claude/orchestrator_state.md` in the claude.ai Project "Lunar Thermal Surrogate | PhysicsNeMo Extension of Penumbra Luna".
- If your prompt names or attaches a handoff, use the orchestrated-handoffs skill in worker mode. If the skill isn't available: read the handoff first, then the files it points to; do only its scope; list decisions under open questions in its Report instead of making them; work on the branch it names; fill in its Report (or put it in your final message if you can't write the file).
- Never commit, quote or link anything from `handoffs/`. Commit messages, PR descriptions and branch names never mention handoffs or task IDs.
- Don't edit other handoffs or `handoffs/INDEX.md`.
