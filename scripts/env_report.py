"""Print the versions that define this project's GPU environment.

Run with `uv run python scripts/env_report.py`. The output is what the README's
Environment section records, and what another PhysicsNeMo project on this
machine should match. Every probe is best-effort: a missing tool prints
"unavailable" rather than stopping the report, so the script also works as a
diagnostic on a machine where something is broken.
"""

from __future__ import annotations

import platform
import subprocess
from importlib import metadata


def _run(cmd: list[str]) -> str:
    """Return a command's stripped stdout, or "unavailable" if it fails."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return out.stdout.strip()


def _driver_info() -> tuple[str, str]:
    """Windows driver version and the CUDA version it supports, from nvidia-smi.

    Under WSL2 `nvidia-smi` reports the Windows host driver; there is no Linux
    driver inside WSL. The CUDA version is parsed from the banner because
    `--query-gpu` has no field for it.
    """
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    cuda = "unavailable"
    for line in _run(["nvidia-smi"]).splitlines():
        if "CUDA" in line and "Version" in line:
            # e.g. "| NVIDIA-SMI 620.67.01  KMD Version: 621.01  CUDA UMD Version: 13.5 |"
            tail = line.split("CUDA", 1)[1]
            cuda = tail.split(":", 1)[1].strip(" |").split()[0]
            break
    return driver.splitlines()[0] if driver != "unavailable" else driver, cuda


def main() -> None:
    import torch  # imported in main so the helpers above import without torch

    driver, driver_cuda = _driver_info()
    rows = [
        ("Python", platform.python_version()),
        ("uv", _run(["uv", "--version"])),
        ("NVIDIA driver (Windows host)", driver),
        ("CUDA supported by driver", driver_cuda),
        ("torch", torch.__version__),
        ("torch CUDA build", str(torch.version.cuda)),
        ("CUDA available", str(torch.cuda.is_available())),
        ("torch arch list", " ".join(torch.cuda.get_arch_list())),
        (
            "Device",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        ),
        (
            "Compute capability",
            "sm_{}{}".format(*torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else "none",
        ),
        ("nvidia-physicsnemo", metadata.version("nvidia-physicsnemo")),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")


if __name__ == "__main__":
    main()
