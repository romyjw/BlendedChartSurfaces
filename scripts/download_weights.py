'''
Download the pre-trained DeepSDF networks that the surface configs in
configs/surfaces/ use as target shapes. They are too large for git, so they are
attached to a GitHub release; this script saves them to sdf_weights/surfaces/.
Analytic target shapes (e.g. urchin, twisted_torus, mobius) do not need them.

Usage, from the repository root:
    python scripts/download_weights.py            # all networks (~2.7 GB)
    python scripts/download_weights.py bob arm    # only some shapes
'''

import sys
import urllib.request
from pathlib import Path

RELEASE_URL = "https://github.com/romyjw/BlendedChartSurfaces/releases/download/v1.0"

SHAPES = [
    "arm", "bob", "cat-reference", "dice", "double_torus", "fandisk", "fertility",
    "fertility_no_posenc", "fruit", "gear", "igea", "lion-reference", "oloid",
]

OUT_DIR = Path(__file__).resolve().parent.parent / "sdf_weights" / "surfaces"


def download(shape):
    name = f"deep3d_{shape}_best.ckpt"
    dst = OUT_DIR / name
    if dst.exists():
        print(f"{name}: already present, skipping")
        return
    part = dst.with_name(name + ".part")
    print(f"{name}: downloading")
    with urllib.request.urlopen(f"{RELEASE_URL}/{name}") as response, open(part, "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    part.rename(dst)


def main(requested):
    unknown = [s for s in requested if s not in SHAPES]
    if unknown:
        sys.exit(f"Unknown shape(s): {', '.join(unknown)}. Available: {', '.join(SHAPES)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for shape in requested or SHAPES:
        download(shape)


if __name__ == "__main__":
    main(sys.argv[1:])
