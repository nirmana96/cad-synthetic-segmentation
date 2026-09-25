import pandas as pd
import pytest

gr = pytest.importorskip("gradio")

from cad2coco import app, pipeline  # noqa: E402
from cad2coco.config import GenerationConfig  # noqa: E402

from .test_pipeline import write_fake_render  # noqa: E402


@pytest.fixture(scope="module")
def demo():
    return app.build_app()


def test_every_config_field_has_a_widget(demo):
    cfg_keys = {f"{s}.{k}" for s, sec in GenerationConfig().to_dict().items() for k in sec}
    assert set(app.FIELDS) == cfg_keys


def test_widget_values_roundtrip(demo):
    cfg = GenerationConfig()
    cfg.camera.fov_max = 80.0
    assert app.config_from_values(app.values_from_config(cfg)) == cfg


def test_upload_fills_class_table(sample_obj, demo):
    table, dropdown, preview = app.on_upload([str(sample_obj), str(sample_obj.with_suffix(".mtl"))])
    assert list(table["class name"]) == ["part"]
    assert preview.endswith("part.obj")


def test_generate_callback(sample_obj, tmp_path, monkeypatch, demo):
    import json
    from pathlib import Path

    def fake_render(job_path, total):
        job = json.loads(Path(job_path).read_text())
        write_fake_render(Path(job["render_dir"]), n=total, category_ids=[1])
        yield pipeline.Event("log", "blender says hi")

    monkeypatch.setattr(pipeline, "_render", fake_render)
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path / "runs")
    values = app.values_from_config(GenerationConfig())
    values[list(app.FIELDS).index("dataset.num_images")] = 6
    table = pd.DataFrame({"model file": ["part.obj"], "class name": ["valve"]})
    outputs = list(app.on_generate([str(sample_obj)], table, *values, progress=lambda *a, **k: None))
    log, gallery, plot, summary, zip_path, snippet, last = outputs[-1]
    assert "blender says hi" in log
    assert len(gallery) > 0 and plot.value["data"][0][0] == "valve" and plot.y_lim[0] == 0
    assert zip_path.endswith(".zip") and "RFDETRSeg" in snippet
    root = app.open_last(last)
    split, classes = app.inspector_choices(root)
    assert split["value"] == "train" and classes["value"] == ["valve"]
