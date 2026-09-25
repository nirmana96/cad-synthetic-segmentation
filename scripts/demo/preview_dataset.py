"""Render a small dataset with the CPU preview renderer (no Blender) and convert it with the
real COCO pipeline. Used to build the demo GIF on machines without Blender.

    python scripts/demo/preview_dataset.py --out demo_data -n 48
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import preview_render as pr  # noqa: E402

from cad2coco.coco import Category, build_coco_dataset, dataset_stats  # noqa: E402


def main():
    models = HERE.parents[1] / "examples" / "models"
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=[str(models / f) for f in ("pipe_flange.obj", "l_bracket.obj", "pipe_tee.obj")])
    ap.add_argument("--classes", nargs="+", default=["flange", "bracket", "tee"])
    ap.add_argument("-n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="demo_data")
    args = ap.parse_args()

    parts = [(pr._shaded(pr.load_part(m)), i + 1, c) for i, (m, c) in enumerate(zip(args.models, args.classes,
                                                                                         strict=True))]
    out = Path(args.out)
    pr.write_dataset_frames(parts, out / "render", args.n, np.random.default_rng(args.seed),
                            on_frame=lambda d, n: print(f"\rrendered {d}/{n}", end="", flush=True))
    print()
    build_coco_dataset(out / "render", out / "dataset", [Category(i + 1, c) for i, c in enumerate(args.classes)])
    print(dataset_stats(out / "dataset"))


if __name__ == "__main__":
    main()
