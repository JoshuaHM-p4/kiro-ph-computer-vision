"""Download the browser-side MediaPipe assets so the web demos load locally.

    .venv/bin/python -m demos.tools.vendor_web_assets
    .venv/bin/python -m demos.tools.vendor_web_assets --check

Without this, each browser fetches roughly 15 MB from jsDelivr and
storage.googleapis.com on a cold cache, which makes the first page load look like
it has hung. With it, ``landmark-stream.js`` detects
``/shared/static/vendor/vision_bundle.mjs`` and serves everything from this
machine instead: instant loads, and the demos work offline.

The files are third-party binaries, so they are git-ignored rather than committed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

TASKS_VERSION = "0.10.21"
CDN_BASE = f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{TASKS_VERSION}"
MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"

VENDOR_DIR = Path(__file__).resolve().parent.parent / "common" / "static" / "vendor"

# Relative destination -> source URL. The nosimd wasm build is included because
# FilesetResolver picks it on browsers without SIMD support.
ASSETS: dict[str, str] = {
    "vision_bundle.mjs": f"{CDN_BASE}/vision_bundle.mjs",
    "wasm/vision_wasm_internal.js": f"{CDN_BASE}/wasm/vision_wasm_internal.js",
    "wasm/vision_wasm_internal.wasm": f"{CDN_BASE}/wasm/vision_wasm_internal.wasm",
    "wasm/vision_wasm_nosimd_internal.js": f"{CDN_BASE}/wasm/vision_wasm_nosimd_internal.js",
    "wasm/vision_wasm_nosimd_internal.wasm": f"{CDN_BASE}/wasm/vision_wasm_nosimd_internal.wasm",
    "hand_landmarker.task": f"{MODEL_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task": f"{MODEL_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task": f"{MODEL_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def human(size: int) -> str:
    return f"{size / 1_048_576:.1f} MB" if size >= 1_048_576 else f"{size / 1024:.0f} kB"


def download(url: str, destination: Path, *, timeout: float = 120.0) -> int:
    """Fetch one asset to ``destination`` via a temporary file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "kiro-cv-demos"})
    with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)
    return destination.stat().st_size


def check(vendor_dir: Path = VENDOR_DIR) -> list[str]:
    """Return the relative paths that are missing or empty."""
    missing: list[str] = []
    for relative in ASSETS:
        path = vendor_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)
    return missing


def vendor(vendor_dir: Path = VENDOR_DIR, *, force: bool = False) -> int:
    """Download every missing asset. Returns the number of bytes written."""
    total = 0
    for relative, url in ASSETS.items():
        destination = vendor_dir / relative
        if destination.is_file() and destination.stat().st_size > 0 and not force:
            print(f"  have {relative} ({human(destination.stat().st_size)})")
            continue
        print(f"  get  {relative} ...", end=" ", flush=True)
        try:
            size = download(url, destination)
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"FAILED ({error})")
            raise SystemExit(
                "Could not download the browser assets. The web demos still work "
                "over the CDN as long as the browser has internet access."
            ) from error
        total += size
        print(human(size))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Vendor the browser MediaPipe assets")
    parser.add_argument("--out", type=Path, default=VENDOR_DIR, help="Destination directory")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report whether the assets are present",
    )
    args = parser.parse_args()

    if args.check:
        missing = check(args.out)
        if missing:
            print(f"{len(missing)} of {len(ASSETS)} assets missing in {args.out}:")
            for relative in missing:
                print(f"  {relative}")
            print("Run: .venv/bin/python -m demos.tools.vendor_web_assets")
            sys.exit(1)
        print(f"All {len(ASSETS)} browser assets present in {args.out}")
        return

    print(f"Vendoring browser assets into {args.out}")
    total = vendor(args.out, force=args.force)
    print(f"Done ({human(total)} downloaded). The web demos now load without the CDN.")


if __name__ == "__main__":
    main()
