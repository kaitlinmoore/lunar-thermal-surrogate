# Changelog

Instrument and design changes, one line each: date | instrument | change | reason. Notes whether held-out data was involved.

- 2026-11-23 | environment | Python 3.12; core `nvidia-physicsnemo` 2.3.0 with PyTorch from PyTorch's cu130 index (no cu12/cu13 extra); uv environment per checkout in the WSL filesystem via `.envrc` | RTX 5080 (sm_120) needs a CUDA 12.8+ build of PyTorch; the cu13 extra would pull RAPIDS, CuPy and DALI and pins `numpy<2.5`, which conflicts with the vendored illumination code's `numpy>=2.5.4`. No held-out data involved.
