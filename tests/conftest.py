"""Shared fixtures: a fake BlenderProc render output, so tests run without Blender."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def write_fake_render(render_dir: Path, n: int = 8, width: int = 160, height: int = 120,
                      category_ids=(1, 2), seed: int = 0) -> Path:
    """Mimic what render_scene.py writes: images/, masks/ (npz uint16) and frames.jsonl."""
    rng = np.random.default_rng(seed)
    (render_dir / "images").mkdir(parents=True, exist_ok=True)
    (render_dir / "masks").mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        rgb = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        inst = np.zeros((height, width), np.uint16)
        # instance 1: disc; instance 2: rectangle partly occluding it
        cx, cy = int(rng.integers(40, 80)), int(rng.integers(40, 80))
        cv2.circle(inst, (cx, cy), 25, 1, -1)
        cv2.rectangle(inst, (cx + 10, cy - 10), (cx + 60, cy + 30), 2, -1)
        stem = f"{i:06d}"
        cv2.imwrite(str(render_dir / "images" / f"{stem}.jpg"), rgb)
        np.savez_compressed(render_dir / "masks" / f"{stem}.npz", instances=inst)
        lines.append(json.dumps({
            "image": f"images/{stem}.jpg", "mask": f"masks/{stem}.npz", "width": width, "height": height,
            "instances": [
                {"id": 1, "category_id": category_ids[0], "visible_px": int((inst == 1).sum())},
                {"id": 2, "category_id": category_ids[-1], "visible_px": int((inst == 2).sum())},
            ],
        }))
    (render_dir / "frames.jsonl").write_text("\n".join(lines) + "\n")
    return render_dir


@pytest.fixture
def fake_render(tmp_path):
    return write_fake_render(tmp_path / "render")


@pytest.fixture
def sample_obj(tmp_path):
    """An .obj with an .mtl next to it."""
    d = tmp_path / "cad"
    d.mkdir()
    (d / "part.mtl").write_text("newmtl steel\nKd 0.6 0.6 0.6\n")
    (d / "part.obj").write_text(
        "mtllib part.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl steel\nf 1 2 3\n")
    return d / "part.obj"
