# Changelog

Instrument and design changes, one line each: date | instrument | change | reason. Notes whether held-out data was involved.

- 2026-11-23 | environment | Python 3.12; core `nvidia-physicsnemo` 2.3.0 with PyTorch from PyTorch's cu130 index (no cu12/cu13 extra); uv environment per checkout in the WSL filesystem via `.envrc` | RTX 5080 (sm_120) needs a CUDA 12.8+ build of PyTorch; the cu13 extra would pull RAPIDS, CuPy and DALI and pins `numpy<2.5`, which conflicts with the vendored illumination code's `numpy>=2.5.4`. No held-out data involved.
- 2026-11-23 | illumination engine | vendored Penumbra Luna's `lss` engine unchanged into `src/lss/` from nasa-lunar-site-scout commit `aef153a` (Apache-2.0 kept; SHA-256 manifest in `src/lss/VENDORED.md`), with its fetch script and engine tests; copied its `data/` and `kernels/` into `LSS_DATA_DIR`; added `scripts/illumination_point.py` | forcing source for the training data, copied rather than imported so the repo runs on its own. No held-out data involved: the two demonstration points are 95 km and 1134 km from the nearest of the 23 sites (the copied engine tests do read site terrain, as in Penumbra Luna).
