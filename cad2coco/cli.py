"""Command line interface.

    cad2coco generate bracket.obj flange.stl --classes bracket,flange -n 500 -o runs
    cad2coco preview runs/<run>/dataset --split valid -o previews
    cad2coco stats runs/<run>/dataset
    cad2coco ui
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PRESETS, GenerationConfig, apply_preset


def _cmd_generate(args: argparse.Namespace) -> int:
    from .pipeline import ModelSpec, generate

    cfg = GenerationConfig.from_yaml(args.config) if args.config else GenerationConfig()
    if args.preset:
        apply_preset(cfg, args.preset)
    if args.num_images:
        cfg.dataset.num_images = args.num_images
    if args.seed is not None:
        cfg.dataset.seed = args.seed
    if args.physics:
        cfg.scene.placement = "physics"

    models = [Path(m) for m in args.models]
    names = [n.strip() for n in args.classes.split(",")] if args.classes else [m.stem for m in models]
    if len(names) != len(models):
        print("error: --classes needs one name per model", file=sys.stderr)
        return 2
    specs = [ModelSpec(m, n) for m, n in zip(models, names, strict=True)]

    try:
        for event in generate(cfg, specs, runs_dir=args.out, run_name=args.name):
            if event.kind == "progress":
                print(f"[{event.fraction * 100:5.1f}%] {event.message}", flush=True)
            elif event.kind == "log" and args.verbose:
                print(event.message, flush=True)
            elif event.kind == "done":
                r = event.result
                print(f"\n{event.message}\nDataset: {r.dataset_dir}\nZip:     {r.zip_path}")
                print(json.dumps(r.stats, indent=2))
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    import cv2

    from .visualize import overlay_images

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = overlay_images(args.dataset, args.split, limit=args.n)
    for img, caption in items:
        name = caption.split(" ")[0]
        cv2.imwrite(str(out / f"overlay_{name}"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"wrote {len(items)} overlays to {out}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    from .coco import dataset_stats, find_dataset_root

    print(json.dumps(dataset_stats(find_dataset_root(args.dataset)), indent=2))
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    from .app import launch

    launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cad2coco", description="CAD models -> synthetic COCO segmentation data")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="render a dataset from CAD models")
    g.add_argument("models", nargs="+", help=".obj / .stl / .ply / .glb / .gltf / .fbx files")
    g.add_argument("--classes", help="comma separated class names, one per model (default: file names)")
    g.add_argument("-n", "--num-images", type=int)
    g.add_argument("-c", "--config", help="YAML config (see configs/default.yaml)")
    g.add_argument("--preset", choices=list(PRESETS))
    g.add_argument("--physics", action="store_true", help="drop parts onto the floor with rigid-body physics")
    g.add_argument("--seed", type=int)
    g.add_argument("-o", "--out", default="runs", help="runs folder (default: runs)")
    g.add_argument("--name", help="run name (default: first class name)")
    g.add_argument("-v", "--verbose", action="store_true", help="print Blender output")
    g.set_defaults(func=_cmd_generate)

    v = sub.add_parser("preview", help="draw annotations on dataset images")
    v.add_argument("dataset")
    v.add_argument("--split", default="train")
    v.add_argument("-n", type=int, default=16)
    v.add_argument("-o", "--out", default="previews")
    v.set_defaults(func=_cmd_preview)

    s = sub.add_parser("stats", help="print dataset statistics")
    s.add_argument("dataset")
    s.set_defaults(func=_cmd_stats)

    u = sub.add_parser("ui", help="launch the Gradio app")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=7860)
    u.add_argument("--share", action="store_true")
    u.set_defaults(func=_cmd_ui)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
