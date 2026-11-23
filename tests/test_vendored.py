"""The vendored illumination engine (`src/lss`) is unchanged from its source commit.

`src/lss/VENDORED.md` ends with a manifest: one `<sha256>  <path>` line per copied file, as written by
`sha256sum`. These tests fail if a copied file is edited, deleted or added without updating the
manifest, which is the point: `src/lss` is never edited here (changes go upstream, then are re-copied).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

LSS_DIR = Path(__file__).resolve().parent.parent / "src" / "lss"
VENDORED = LSS_DIR / "VENDORED.md"
MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  (\S+)$")


def manifest() -> dict[str, str]:
    """Path (relative to src/lss, forward slashes) -> sha256, from the manifest block in VENDORED.md."""
    out = {}
    for line in VENDORED.read_text(encoding="utf-8").splitlines():
        m = MANIFEST_LINE.match(line)
        if m:
            out[m.group(2)] = m.group(1)
    return out


def files_on_disk() -> set[str]:
    """Every file under src/lss except VENDORED.md itself and Python's bytecode caches."""
    return {
        p.relative_to(LSS_DIR).as_posix()
        for p in LSS_DIR.rglob("*")
        if p.is_file() and p.name != "VENDORED.md" and "__pycache__" not in p.parts
    }


def test_manifest_is_present():
    m = manifest()
    assert "LICENSE" in m
    assert "series/core.py" in m
    assert len(m) == len(set(m)), "duplicate paths in the manifest"


def test_no_unlisted_or_missing_files():
    listed, present = set(manifest()), files_on_disk()
    assert present - listed == set(), "files in src/lss that the manifest doesn't list"
    assert listed - present == set(), (
        "files the manifest lists that are missing from src/lss"
    )


@pytest.mark.parametrize("relpath", sorted(manifest()))
def test_file_matches_manifest(relpath):
    digest = hashlib.sha256((LSS_DIR / relpath).read_bytes()).hexdigest()
    assert digest == manifest()[relpath], (
        f"src/lss/{relpath} differs from the vendored copy"
    )
