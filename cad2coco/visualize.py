"""Draw COCO polygon annotations on top of their images."""

from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import numpy as np

from .coco import load_split


def category_color(cat_id: int) -> tuple[int, int, int]:
    """Stable, well-separated RGB colour per category (golden-ratio hue walk)."""
    h = (cat_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_annotations(image_rgb: np.ndarray, annotations: list[dict], names: dict[int, str],
                     alpha: float = 0.35) -> np.ndarray:
    canvas = image_rgb.copy()
    fill = image_rgb.copy()
    thickness = max(1, round(min(image_rgb.shape[:2]) / 320))
    for ann in annotations:
        color = category_color(ann["category_id"])
        for poly in ann["segmentation"]:
            pts = np.round(np.array(poly, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
            cv2.fillPoly(fill, [pts], color)
    canvas = cv2.addWeighted(fill, alpha, canvas, 1 - alpha, 0)
    for ann in annotations:
        color = category_color(ann["category_id"])
        for poly in ann["segmentation"]:
            pts = np.round(np.array(poly, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
            cv2.polylines(canvas, [pts], True, color, thickness, cv2.LINE_AA)
        x, y, _, _ = (int(v) for v in ann["bbox"])
        label = names.get(ann["category_id"], str(ann["category_id"]))
        scale = 0.4 * thickness
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        y_text = max(y, th + 4)
        cv2.rectangle(canvas, (x, y_text - th - 4), (x + tw + 4, y_text), color, -1)
        cv2.putText(canvas, label, (x + 2, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1,
                    cv2.LINE_AA)
    return canvas


def overlay_images(dataset_dir: str | Path, split: str = "train", limit: int = 12,
                   category_filter: set[int] | None = None) -> list[tuple[np.ndarray, str]]:
    """(overlay image, caption) pairs for the first ``limit`` images of a split."""
    coco = load_split(dataset_dir, split)
    if coco is None:
        return []
    names = {c["id"]: c["name"] for c in coco["categories"]}
    by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        if category_filter is None or ann["category_id"] in category_filter:
            by_image.setdefault(ann["image_id"], []).append(ann)
    out = []
    for img in coco["images"]:
        anns = by_image.get(img["id"], [])
        if category_filter is not None and not anns:
            continue
        bgr = cv2.imread(str(Path(dataset_dir) / split / img["file_name"]))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        caption = f"{img['file_name']} · {len(anns)} instance{'s' if len(anns) != 1 else ''}"
        out.append((draw_annotations(rgb, anns, names), caption))
        if len(out) >= limit:
            break
    return out
