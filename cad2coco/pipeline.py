"""End-to-end run: stage models -> BlenderProc render -> COCO splits -> zip.

`generate()` is a generator of events so the Gradio UI can stream progress and logs
and the CLI can print them; closing the generator stops the Blender subprocess.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .coco import Category, build_coco_dataset, dataset_stats
from .config import GenerationConfig

RENDER_SCRIPT = Path(__file__).parent / "blender" / "render_scene.py"
MODEL_EXTENSIONS = {".obj", ".stl", ".ply", ".glb", ".gltf", ".fbx"}
PROGRESS_RE = re.compile(r"^PROGRESS (\d+)/(\d+)")


@dataclass
class ModelSpec:
    path: Path
    class_name: str


@dataclass
class RunResult:
    run_dir: Path
    dataset_dir: Path
    zip_path: Path
    counts: dict[str, int]
    stats: dict
    categories: list[Category]
    seconds: float


@dataclass
class Event:
    kind: str  # "log" | "progress" | "done"
    message: str = ""
    fraction: float = 0.0
    result: RunResult | None = field(default=None, repr=False)


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip()).strip("_") or "part"


def assign_categories(specs: list[ModelSpec]) -> list[Category]:
    """One category per distinct class name, ids from 1 (files sharing a name share a class)."""
    names: list[str] = []
    for s in specs:
        name = s.class_name.strip() or s.path.stem
        if name not in names:
            names.append(name)
    return [Category(i, n) for i, n in enumerate(names, start=1)]


def _referenced_files(model: Path) -> list[Path]:
    """.mtl libraries referenced by an .obj and the texture maps referenced by those .mtl files."""
    found: list[Path] = []
    if model.suffix.lower() != ".obj":
        return found
    for line in model.read_text(errors="ignore").splitlines():
        if line.startswith("mtllib "):
            mtl = model.parent / line.split(maxsplit=1)[1].strip()
            if mtl.exists():
                found.append(mtl)
                for mline in mtl.read_text(errors="ignore").splitlines():
                    if mline.strip().lower().startswith("map_"):
                        tex = model.parent / mline.split()[-1]
                        if tex.exists():
                            found.append(tex)
    return found


def stage_models(specs: list[ModelSpec], extra_files: list[Path], dest: Path) -> list[ModelSpec]:
    """Copy models (+ .mtl / textures) into one folder so relative references resolve."""
    dest.mkdir(parents=True, exist_ok=True)
    staged = []
    for src in [*extra_files, *(f for s in specs for f in _referenced_files(s.path))]:
        shutil.copy2(src, dest / src.name)
    for s in specs:
        target = dest / s.path.name
        shutil.copy2(s.path, target)
        staged.append(ModelSpec(target, s.class_name))
    return staged


def blenderproc_command(job_path: Path) -> list[str]:
    exe = shutil.which("blenderproc")
    cmd = [exe] if exe else [sys.executable, "-m", "blenderproc"]
    cmd += ["run", str(RENDER_SCRIPT)]
    blender_path = os.environ.get("CAD2COCO_BLENDER_PATH")
    if blender_path:
        cmd += ["--custom-blender-path", blender_path]
    return cmd + [str(job_path)]


def generate(
    cfg: GenerationConfig,
    specs: list[ModelSpec],
    runs_dir: str | Path = "runs",
    run_name: str | None = None,
    extra_files: list[Path] | None = None,
) -> Iterator[Event]:
    cfg.validate()
    if not specs:
        raise ValueError("Add at least one CAD model")
    for s in specs:
        if s.path.suffix.lower() not in MODEL_EXTENSIONS:
            raise ValueError(f"Unsupported model format: {s.path.name}")

    t0 = time.time()
    categories = assign_categories(specs)
    by_name = {c.name: c.id for c in categories}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_dir).resolve() / f"{stamp}_{slugify(run_name or categories[0].name)}"
    render_dir, dataset_dir = run_dir / "render", run_dir / "dataset"

    staged = stage_models(specs, [Path(p) for p in (extra_files or [])], run_dir / "models")
    job = cfg.to_dict()
    job["render_dir"] = str(render_dir)
    job["models"] = [
        {"path": str(s.path), "name": slugify(s.class_name or s.path.stem),
         "category_id": by_name[s.class_name.strip() or s.path.stem]}
        for s in staged
    ]
    job_path = run_dir / "job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    cfg.to_yaml(run_dir / "config.yaml")

    yield Event("log", f"Run folder: {run_dir}")
    yield Event("log", "Classes: " + ", ".join(f"{c.id}={c.name}" for c in categories))
    yield from _render(job_path, cfg.dataset.num_images)

    yield Event("progress", "Converting masks to COCO polygons", 0.93)
    counts = build_coco_dataset(
        render_dir, dataset_dir, categories,
        valid_ratio=cfg.dataset.valid_ratio, test_ratio=cfg.dataset.test_ratio, seed=cfg.dataset.seed,
        tolerance=cfg.annotation.polygon_tolerance, min_part_area=cfg.annotation.min_part_area,
    )
    shutil.copy2(run_dir / "config.yaml", dataset_dir / "cad2coco_config.yaml")

    yield Event("progress", "Zipping dataset", 0.97)
    zip_path = Path(shutil.make_archive(str(run_dir / f"{run_dir.name}_coco"), "zip", root_dir=dataset_dir))

    result = RunResult(run_dir, dataset_dir, zip_path, counts, dataset_stats(dataset_dir), categories,
                       time.time() - t0)
    yield Event("done", f"Finished in {result.seconds:.0f}s", 1.0, result)


def _render(job_path: Path, total: int) -> Iterator[Event]:
    cmd = blenderproc_command(job_path)
    yield Event("log", "$ " + " ".join(cmd))
    yield Event("progress", "Starting Blender (the first run downloads it)", 0.0)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            tail = (tail + [line])[-60:]
            m = PROGRESS_RE.match(line)
            if m:
                done = int(m.group(1))
                yield Event("progress", f"Rendered {done}/{total}", 0.9 * done / max(total, 1))
            elif line:
                yield Event("log", line)
        proc.wait()
    finally:
        if proc.poll() is None:  # generator closed early (Stop button / Ctrl-C)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    if proc.returncode != 0:
        raise RuntimeError(f"BlenderProc exited with code {proc.returncode}:\n" + "\n".join(tail[-25:]))
