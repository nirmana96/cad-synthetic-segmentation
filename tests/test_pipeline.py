import json
import zipfile
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

from cad2coco import pipeline
from cad2coco.coco import Category, build_coco_dataset, dataset_stats, find_dataset_root
from cad2coco.config import GenerationConfig
from cad2coco.pipeline import ModelSpec, assign_categories, stage_models
from cad2coco.visualize import overlay_images

from .conftest import write_fake_render

CATS = [Category(1, "flange"), Category(2, "bracket")]


def test_build_dataset_is_valid_coco(fake_render, tmp_path):
    out = tmp_path / "dataset"
    counts = build_coco_dataset(fake_render, out, CATS, valid_ratio=0.25, test_ratio=0.25, seed=0)
    assert counts == {"train": 4, "valid": 2, "test": 2}
    for split, n in counts.items():
        coco = COCO(str(out / split / "_annotations.coco.json"))
        assert len(coco.getImgIds()) == n
        assert len(coco.getAnnIds()) == 2 * n
        for ann in coco.loadAnns(coco.getAnnIds()):
            m = coco.annToMask(ann)
            assert m.sum() > 0.85 * ann["area"]
            assert all(len(p) >= 6 and len(p) % 2 == 0 for p in ann["segmentation"])
        for img in coco.loadImgs(coco.getImgIds()):
            assert (out / split / img["file_name"]).exists()


def test_stats_and_overlays(fake_render, tmp_path):
    out = tmp_path / "dataset"
    build_coco_dataset(fake_render, out, CATS, 0.25, 0.25, 0)
    stats = dataset_stats(out)
    assert stats["images"] == 8 and stats["instances"] == 16
    assert stats["per_class"] == {"flange": 8, "bracket": 8}
    items = overlay_images(out, "train", limit=3)
    assert len(items) == 3 and items[0][0].dtype == np.uint8
    only_flange = overlay_images(out, "train", limit=10, category_filter={1})
    assert "1 instance" in only_flange[0][1]


def test_find_root_inside_zip_folder(fake_render, tmp_path):
    build_coco_dataset(fake_render, tmp_path / "wrap" / "ds", CATS, 0.25, 0.25, 0)
    assert find_dataset_root(tmp_path / "wrap") == tmp_path / "wrap" / "ds"


def test_assign_categories_merges_same_name():
    specs = [ModelSpec(Path("a.obj"), "valve"), ModelSpec(Path("b.obj"), "valve"), ModelSpec(Path("c.stl"), "")]
    assert assign_categories(specs) == [Category(1, "valve"), Category(2, "c")]


def test_stage_models_copies_mtl(sample_obj, tmp_path):
    staged = stage_models([ModelSpec(sample_obj, "part")], [], tmp_path / "models")
    assert staged[0].path.exists() and (tmp_path / "models" / "part.mtl").exists()


def test_generate_end_to_end_with_fake_renderer(sample_obj, tmp_path, monkeypatch):
    def fake_render(job_path, total):
        job = json.loads(Path(job_path).read_text())
        cats = [m["category_id"] for m in job["models"]]
        write_fake_render(Path(job["render_dir"]), n=total, category_ids=cats)
        yield pipeline.Event("progress", f"Rendered {total}/{total}", 0.9)

    monkeypatch.setattr(pipeline, "_render", fake_render)
    cfg = GenerationConfig()
    cfg.dataset.num_images = 10
    events = list(pipeline.generate(cfg, [ModelSpec(sample_obj, "valve")], runs_dir=tmp_path / "runs"))
    result = events[-1].result
    assert events[-1].kind == "done"
    assert sum(result.counts.values()) == 10
    assert (result.dataset_dir / "cad2coco_config.yaml").exists()
    names = zipfile.ZipFile(result.zip_path).namelist()
    assert "train/_annotations.coco.json" in names
    job = json.loads((result.run_dir / "job.json").read_text())
    assert job["models"][0]["category_id"] == 1 and job["dataset"]["num_images"] == 10


def test_render_script_compiles():
    import py_compile
    py_compile.compile(str(pipeline.RENDER_SCRIPT), doraise=True)
