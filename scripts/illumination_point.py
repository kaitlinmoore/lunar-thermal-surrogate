"""Sun visibility over the real terrain horizon at one point, run headless.

    uv run python scripts/illumination_point.py LAT LON [--height M] [--start UTC] [--days D] [--out PATH]

For a latitude and east longitude (planetocentric degrees), an observer height above the ground and a
UTC window (default: one mean synodic month from the start), this writes one `.npz` file holding the
Sun's azimuth, elevation and visible solar-disc fraction over the window, sampled on the engine's
internal grid (60 s by default), plus the terrain horizon profile the fraction was computed against.

Everything physical comes from the vendored Penumbra Luna engine (`src/lss`, unchanged; see
`src/lss/VENDORED.md`):

- `lss.registry.terrain.resolve` picks the terrain layers that cover the point (5 m site map, 80 m and
  240 m south-polar maps, or the 237 m global LOLA map off the polar maps);
- `lss.horizon.cached_horizon` ray-casts the horizon out to 400 km, or loads it from
  `LSS_DATA_DIR/cache/horizon/` if this point and height were computed before;
- `lss.series.compute_visibility` samples the Sun (and the Earth, which this script does not keep) with
  SPICE and cuts the limb-darkened disc with that horizon.

This script only chooses inputs, times the two steps, and writes the arrays the engine returns. It adds
no physics. Absorbed flux (albedo, incidence on sloped ground) is a separate step, not done here.

The output goes outside the repo: by default `$HOME/data/lunar-thermal-surrogate/derived/`. A path inside
the repo is refused, so generated data never lands in git or on the slow Windows drive by accident.

The script does not check the point against the 23 held-out sites. Choosing points away from them is
the caller's job (the training sampler does that with the exclusion radius).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from lss.config import get_settings
from lss.horizon import HorizonParams, cached_horizon
from lss.horizon.cache import cache_path, key_spec
from lss.registry.terrain import resolve
from lss.series import SeriesParams, compute_visibility

# Mean synodic month (new Moon to new Moon), in days. A single lunation is the forcing period the
# surrogate is trained on; at the poles it is not exactly periodic (seasons), see the planning package.
SYNODIC_MONTH_DAYS = 29.530589

# Where the vendored engine came from, recorded in every output file for provenance.
ENGINE_SOURCE = (
    "github.com/kaitlinmoore/nasa-lunar-site-scout@1e45c30 (src/lss, unchanged)"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = Path.home() / "data" / "lunar-thermal-surrogate" / "derived"


def parse_utc(text: str) -> datetime:
    """ISO 8601 time; a time without a zone is taken as UTC (the engine needs aware datetimes)."""
    t = datetime.fromisoformat(text)  # Python 3.11+ accepts a trailing "Z"
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def output_path(
    out: Path | None, lat: float, lon: float, height_m: float, start: datetime
) -> Path:
    """Resolve --out (a file, a directory, or nothing) to a file path, refusing paths inside the repo."""
    name = f"illum_{lat:+08.3f}_{lon:+08.3f}_{height_m:g}m_{start:%Y%m%dT%H%MZ}.npz"
    if out is None:
        path = DEFAULT_OUT_DIR / name
    elif out.suffix == ".npz":
        path = out
    else:
        path = out / name
    path = path.expanduser().resolve()
    if path.is_relative_to(REPO_ROOT):
        raise SystemExit(f"refusing to write inside the repo: {path}")
    return path


def run(
    lat: float,
    lon: float,
    height_m: float,
    start: datetime,
    days: float,
    grid_step_s: float,
    recompute_horizon: bool,
) -> tuple[dict[str, np.ndarray], dict]:
    """Compute the Sun's visibility series; return (arrays, metadata). No file is written here."""
    end = start + timedelta(days=days)
    hparams = (
        HorizonParams()
    )  # the engine's converged defaults (Penumbra Luna methodology section 3.2)
    sparams = SeriesParams(grid_step_s=grid_step_s)
    stack = resolve(lat, lon, hparams.max_range_m)

    cached = (
        not recompute_horizon
        and cache_path(key_spec(lat, lon, height_m, hparams, stack)).is_file()
    )
    # Horizon first, on its own, so its cost (paid once per point and height) is timed separately from
    # the series (paid once per time window). compute_visibility then loads it from the cache.
    t0 = time.perf_counter()
    hz = cached_horizon(lat, lon, height_m, hparams, recompute=recompute_horizon)
    t1 = time.perf_counter()
    res = compute_visibility(lat, lon, height_m, start, end, sparams, hparams)
    t2 = time.perf_counter()

    sun = res.sun
    g = sun.grid
    sunlit_s = sum(p.duration_s for p in sun.visible)
    fidelity = stack.fidelity()
    arrays = {
        "t_s": g.t_s,  # seconds since window_start (UTC)
        "sun_azimuth_deg": g.azimuth_deg,  # clockwise from north
        "sun_elevation_deg": g.elevation_deg,  # disc centre, above the local horizontal plane
        "sun_radius_deg": g.radius_deg,  # apparent angular radius of the disc
        "horizon_at_sun_deg": g.horizon_deg,  # terrain horizon at the disc centre's azimuth
        "sun_visible_fraction": g.fraction,  # limb-darkened share of the disc's light above the horizon
        "sunrise_s": np.asarray(
            sun.rises, dtype=float
        ),  # refined to SeriesParams.root_tol_s
        "sunset_s": np.asarray(sun.sets, dtype=float),
        "horizon_azimuth_deg": hz.azimuth_deg,
        "horizon_elevation_deg": hz.elevation_deg,
    }
    meta = {
        "lat_deg": lat,
        "lon_deg": lon,
        "unit_height_m": height_m,
        "surface_height_m": hz.surface_height_m,
        "observer_height_m": res.observer_height_m,
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "duration_days": days,
        "series_params": {
            "grid_step_s": sparams.grid_step_s,
            "root_tol_s": sparams.root_tol_s,
        },
        "horizon_params": {
            "azimuth_step_deg": hparams.azimuth_step_deg,
            "max_range_m": hparams.max_range_m,
            "sample_factor": hparams.sample_factor,
            "relative_step": hparams.relative_step,
            "min_range_m": hparams.min_range_m,
        },
        "terrain_layers": [
            {
                "label": sl.spec.label,
                "file_id": sl.spec.file_id,
                "resolution_m": sl.spec.resolution_m,
                "role": sl.role,
                "min_range_m": sl.min_range_m,
                "max_range_m": sl.max_range_m,
            }
            for sl in stack.layers
        ],
        "fidelity_tier": fidelity.tier.value if fidelity else None,
        "fidelity_summary": fidelity.summary if fidelity else None,
        "horizon_missing_samples": hz.missing_samples,
        "horizon_total_samples": hz.total_samples,
        "summary": {
            "sunlit_fraction": sunlit_s
            / res.duration_s,  # any part of the disc visible
            "mean_visible_fraction": float(
                np.trapezoid(g.fraction, g.t_s) / res.duration_s
            ),
            "sunrises": len(sun.rises),
            "sunsets": len(sun.sets),
            "sun_elevation_min_deg": float(g.elevation_deg.min()),
            "sun_elevation_max_deg": float(g.elevation_deg.max()),
            "horizon_max_deg": float(hz.elevation_deg.max()),
        },
        "timing_s": {
            "horizon": t1 - t0,
            "horizon_from_cache": cached,
            "series": t2 - t1,
        },
        "engine": ENGINE_SOURCE,
        "lss_data_dir": str(get_settings().data_dir),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return arrays, meta


def write(path: Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    """One compressed .npz per run; metadata as a JSON string under `meta`. Written atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as f:
        np.savez_compressed(
            f, **arrays, meta=np.array(json.dumps(meta, sort_keys=True))
        )
    part.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument(
        "lat", type=float, help="latitude, planetocentric degrees (south negative)"
    )
    ap.add_argument("lon", type=float, help="longitude, degrees east")
    ap.add_argument(
        "--height",
        type=float,
        default=1.0,
        help="observer height above the ground, m (default 1)",
    )
    ap.add_argument(
        "--start",
        type=parse_utc,
        default=parse_utc("2027-01-01T00:00:00Z"),
        help="UTC start (ISO)",
    )
    ap.add_argument(
        "--days",
        type=float,
        default=SYNODIC_MONTH_DAYS,
        help="window length, days (default: 1 lunation)",
    )
    ap.add_argument(
        "--step",
        type=float,
        default=60.0,
        help="sampling step, s (the engine's grid; default 60)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output .npz file or directory (default {DEFAULT_OUT_DIR})",
    )
    ap.add_argument(
        "--recompute-horizon",
        action="store_true",
        help="ignore a cached horizon (to time it cold)",
    )
    args = ap.parse_args(argv)
    if args.height <= 0:
        ap.error("--height must be positive")

    path = output_path(args.out, args.lat, args.lon, args.height, args.start)
    arrays, meta = run(
        args.lat,
        args.lon,
        args.height,
        args.start,
        args.days,
        args.step,
        args.recompute_horizon,
    )
    write(path, arrays, meta)

    s, t = meta["summary"], meta["timing_s"]
    print(
        f"point        {args.lat:+.4f}, {args.lon:+.4f} at {args.height:g} m; {meta['window_start_utc']} + {args.days:g} d"
    )
    print(
        f"terrain      {meta['fidelity_summary']} (fidelity {meta['fidelity_tier']}); "
        f"missing ray samples {meta['horizon_missing_samples']}/{meta['horizon_total_samples']}"
    )
    print(
        f"sun          sunlit {s['sunlit_fraction']:.4f}, mean visible fraction {s['mean_visible_fraction']:.4f}, "
        f"{s['sunrises']} sunrises, {s['sunsets']} sunsets, elevation {s['sun_elevation_min_deg']:.2f} to "
        f"{s['sun_elevation_max_deg']:.2f} deg; horizon max {s['horizon_max_deg']:.2f} deg"
    )
    print(
        f"time         horizon {t['horizon']:.1f} s{' (cache)' if t['horizon_from_cache'] else ''}, series {t['series']:.1f} s"
    )
    print(f"wrote        {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
