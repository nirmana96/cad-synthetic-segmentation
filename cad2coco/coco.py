"""Instance masks -> COCO polygon annotations, split into train / valid / test.

Output layout (the Roboflow-style layout RF-DETR reads directly):

    dataset/
      train/_annotations.coco.json + images
      valid/_annotations.coco.json + images
      test/_annotations.coco.json  + images
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ANNOTATION_FILE = "_annotations.coco.json"
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Category:
    id: int
    name: str


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------

def mask_to_polygons(mask: np.ndarray, tolerance: float = 1.0, min_part_area: float = 12.0) -> list[list[float]]:
    """Outer contours of a binary mask as COCO polygons ``[[x1, y1, x2, y2, ...], ...]``.

    One instance can yield several polygons when occlusion splits it into parts.
    COCO polygons cannot express holes, so only outer contours are kept.
    """
    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_part_area:
            continue
        if tolerance > 0:
            contour = cv2.approxPolyDP(contour, tolerance, closed=True)
        pts = contour.reshape(-1, 2).astype(float)
        if len(pts) < 3:
            continue
        polygons.append([round(v, 2) for v in pts.flatten().tolist()])
    return polygons


def mask_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    return [float(x0), float(y0), float(xs.max() - x0 + 1), float(ys.max() - y0 + 1)]


# --------------------------------------------------------------------------------------
# dataset building
# --------------------------------------------------------------------------------------

def read_frames(render_dir: Path) -> list[dict]:
    path = Path(render_dir) / "frames.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_indices(n: int, valid_ratio: float, test_ratio: float, seed: int) -> dict[str, list[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = int(round(n * test_ratio))
    n_valid = int(round(n * valid_ratio))
    # keep every split non-empty once there are enough images (RF-DETR expects all three)
    if n >= 3:
        n_test = max(n_test, 1) if test_ratio > 0 else 0
        n_valid = max(n_valid, 1) if valid_ratio > 0 else 0
    n_train = n - n_valid - n_test
    return {
        "train": sorted(idx[:n_train]),
        "valid": sorted(idx[n_train:n_train + n_valid]),
        "test": sorted(idx[n_train + n_valid:]),
    }


def build_coco_dataset(
    render_dir: str | Path,
    out_dir: str | Path,
    categories: list[Category],
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    tolerance: float = 1.0,
    min_part_area: float = 12.0,
) -> dict[str, int]:
    """Convert the render output into COCO splits. Returns ``{split: n_images}``."""
    render_dir, out_dir = Path(render_dir), Path(out_dir)
    frames = read_frames(render_dir)
    if not frames:
        raise RuntimeError(f"No rendered frames found in {render_dir}")

    cat_json = [{"id": c.id, "name": c.name, "supercategory": "part"} for c in categories]
    valid_cat_ids = {c.id for c in categories}
    splits = split_indices(len(frames), valid_ratio, test_ratio, seed)
    counts = {}

    for split, indices in splits.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        ann_id = 1
        for image_id, frame_idx in enumerate(indices, start=1):
            frame = frames[frame_idx]
            src = render_dir / frame["image"]
            file_name = src.name
            shutil.copy2(src, split_dir / file_name)
            images.append({
                "id": image_id, "file_name": file_name,
                "width": frame["width"], "height": frame["height"],
            })
            inst_map = np.load(render_dir / frame["mask"])["instances"]
            for inst in frame["instances"]:
                if inst["category_id"] not in valid_cat_ids:
                    continue
                mask = inst_map == inst["id"]
                if not mask.any():
                    continue
                polys = mask_to_polygons(mask, tolerance, min_part_area)
                if not polys:
                    continue
                annotations.append({
                    "id": ann_id, "image_id": image_id, "category_id": inst["category_id"],
                    "segmentation": polys, "area": float(mask.sum()),
                    "bbox": mask_bbox(mask), "iscrowd": 0,
                })
                ann_id += 1
        coco = {
            "info": {
                "description": "Synthetic dataset generated with CAD2COCO (BlenderProc)",
                "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": cat_json,
        }
        (split_dir / ANNOTATION_FILE).write_text(json.dumps(coco), encoding="utf-8")
        counts[split] = len(images)
    return counts


# --------------------------------------------------------------------------------------
# reading back / statistics
# --------------------------------------------------------------------------------------

def load_split(dataset_dir: str | Path, split: str) -> dict | None:
    path = Path(dataset_dir) / split / ANNOTATION_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_dataset_root(path: str | Path) -> Path:
    """Locate the folder that contains the split folders (handles zips with an extra top folder)."""
    path = Path(path)
    for candidate in [path, *sorted(p for p in path.rglob("*") if p.is_dir())]:
        if any((candidate / s / ANNOTATION_FILE).exists() for s in SPLITS):
            return candidate
    raise FileNotFoundError(f"No {ANNOTATION_FILE} found under {path}")


def dataset_stats(dataset_dir: str | Path) -> dict:
    """Per-split and per-class numbers for the summary tables."""
    per_split, per_class = {}, Counter()
    vertices, n_ann, areas = [], 0, []
    names = {}
    for split in SPLITS:
        coco = load_split(dataset_dir, split)
        if coco is None:
            continue
        names.update({c["id"]: c["name"] for c in coco["categories"]})
        per_split[split] = {"images": len(coco["images"]), "instances": len(coco["annotations"])}
        for ann in coco["annotations"]:
            per_class[names[ann["category_id"]]] += 1
            vertices.extend(len(p) // 2 for p in ann["segmentation"])
            areas.append(ann["area"])
            n_ann += 1
    n_img = sum(v["images"] for v in per_split.values())
    return {
        "per_split": per_split,
        "per_class": dict(per_class),
        "images": n_img,
        "instances": n_ann,
        "instances_per_image": n_ann / n_img if n_img else 0.0,
        "mean_polygon_vertices": float(np.mean(vertices)) if vertices else 0.0,
        "median_instance_area": float(np.median(areas)) if areas else 0.0,
    }
