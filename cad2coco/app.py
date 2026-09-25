"""Gradio front end for CAD2COCO."""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd

from .coco import SPLITS, dataset_stats, find_dataset_root, load_split
from .config import PRESETS, GenerationConfig, apply_preset
from .pipeline import MODEL_EXTENSIONS, ModelSpec, generate
from .visualize import overlay_images

RUNS_DIR = Path(os.environ.get("CAD2COCO_RUNS", "runs")).resolve()
PREVIEWABLE = {".obj", ".glb", ".gltf", ".stl", ".ply"}

CSS = """
#hero {padding: 18px 22px; border-radius: 14px; margin-bottom: 6px;
       background: linear-gradient(120deg, #1f2937 0%, #0f172a 60%, #7c2d12 140%); color: #f8fafc;}
#hero h1 {margin: 0 0 4px 0; font-size: 1.7rem; letter-spacing: .5px; color: #f8fafc;}
#hero p {margin: 0; opacity: .85; color: #e2e8f0;}
#hero .chips span {display: inline-block; margin: 10px 6px 0 0; padding: 2px 10px; border-radius: 999px;
                   font-size: .78rem; background: rgba(251,146,60,.18); color: #fdba74;
                   border: 1px solid rgba(251,146,60,.35);}
.step-title h3 {margin: 2px 0 0 0;}
.step-title h3 span.num {display: inline-flex; width: 24px; height: 24px; border-radius: 50%; margin-right: 8px;
                   align-items: center; justify-content: center; font-size: .85rem;
                   background: var(--color-accent); color: white;}
#generate-btn {min-height: 52px; font-size: 1.05rem;}
footer {display: none !important;}
"""

HERO = """
<div id="hero">
  <h1>CAD2COCO</h1>
  <p>Drop in CAD models &rarr; get a randomised synthetic dataset with
  instance-segmentation polygons in COCO format.</p>
  <div class="chips"><span>BlenderProc</span><span>Domain randomisation</span><span>COCO polygons</span>
  <span>RF-DETR / YOLO-seg ready</span></div>
</div>
"""


def step(num: int, title: str) -> gr.HTML:
    return gr.HTML(f'<div class="step-title"><h3><span class="num">{num}</span>{title}</h3></div>')


# --------------------------------------------------------------------------------------
# widgets <-> config
# --------------------------------------------------------------------------------------

FIELDS: dict[str, gr.components.Component] = {}


def reg(key: str, component):
    FIELDS[key] = component
    return component


def config_from_values(values) -> GenerationConfig:
    cfg = GenerationConfig()
    for key, value in zip(FIELDS, values, strict=True):
        cfg.set(key, value)
    cfg.validate()
    return cfg


def values_from_config(cfg: GenerationConfig) -> list:
    return [cfg.get(key) for key in FIELDS]


# --------------------------------------------------------------------------------------
# callbacks
# --------------------------------------------------------------------------------------

def split_uploads(files: list[str] | None) -> tuple[list[Path], list[Path]]:
    files = [Path(f) for f in (files or [])]
    models = [f for f in files if f.suffix.lower() in MODEL_EXTENSIONS]
    extras = [f for f in files if f.suffix.lower() not in MODEL_EXTENSIONS]
    return models, extras


def on_upload(files):
    models, _ = split_uploads(files)
    table = pd.DataFrame({"model file": [m.name for m in models], "class name": [m.stem for m in models]})
    names = [m.name for m in models]
    first = next((str(m) for m in models if m.suffix.lower() in PREVIEWABLE), None)
    return table, gr.update(choices=names, value=names[0] if names else None), first


def on_preview_pick(files, name):
    models, _ = split_uploads(files)
    for m in models:
        if m.name == name and m.suffix.lower() in PREVIEWABLE:
            return str(m)
    return None


def on_preset(name):
    cfg = GenerationConfig()
    if name in PRESETS:
        apply_preset(cfg, name)
    return values_from_config(cfg)


def on_save_config(*values):
    try:
        cfg = config_from_values(values)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    path = Path(tempfile.mkdtemp()) / "cad2coco_config.yaml"
    cfg.to_yaml(path)
    return path


def on_load_config(file):
    if not file:
        return [gr.skip()] * len(FIELDS)
    try:
        cfg = GenerationConfig.from_yaml(file)
    except (ValueError, TypeError) as exc:
        raise gr.Error(f"Could not read config: {exc}") from exc
    gr.Info("Config loaded")
    return values_from_config(cfg)


def class_plot_df(stats: dict) -> pd.DataFrame:
    return pd.DataFrame({"class": list(stats["per_class"]), "instances": list(stats["per_class"].values())})


def class_plot(stats: dict):
    """Bar chart update with the y-axis anchored at zero (auto-scaling would start at the minimum)."""
    df = class_plot_df(stats)
    top = int(df["instances"].max()) if len(df) else 1
    return gr.BarPlot(value=df, y_lim=[0, max(1, round(top * 1.1))])


def summary_markdown(stats: dict, seconds: float | None = None) -> str:
    rows = "\n".join(f"| {s} | {v['images']} | {v['instances']} |" for s, v in stats["per_split"].items())
    took = f" in **{seconds:.0f}s**" if seconds is not None else ""
    return (
        f"**{stats['images']} images · {stats['instances']} instances**{took}  \n"
        f"{stats['instances_per_image']:.2f} instances / image · "
        f"{stats['mean_polygon_vertices']:.1f} vertices / polygon · "
        f"median instance area {stats['median_instance_area']:.0f} px\n\n"
        f"| split | images | instances |\n|---|---|---|\n{rows}"
    )


def train_snippet(dataset_dir: Path) -> str:
    return (
        "# RF-DETR (reads this folder layout directly)\n"
        "from rfdetr import RFDETRSegMedium\n\n"
        "model = RFDETRSegMedium()\n"
        f"model.train(dataset_dir=r\"{dataset_dir}\", epochs=50, batch_size=4)\n\n"
        "# YOLO-seg: convert first\n"
        "# from ultralytics.data.converter import convert_coco\n"
        f"# convert_coco(r\"{dataset_dir}/train\", use_segments=True)\n"
    )


def on_generate(files, class_table, *values, progress=gr.Progress()):
    models, extras = split_uploads(files)
    if not models:
        raise gr.Error("Upload at least one CAD model (.obj, .stl, .ply, .glb, .gltf or .fbx).")
    try:
        cfg = config_from_values(values)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    names = {}
    if isinstance(class_table, pd.DataFrame) and len(class_table):
        names = dict(zip(class_table.iloc[:, 0].astype(str), class_table.iloc[:, 1].astype(str), strict=False))
    specs = [ModelSpec(m, (names.get(m.name) or m.stem).strip() or m.stem) for m in models]

    log: list[str] = []
    skip = gr.skip()
    try:
        for ev in generate(cfg, specs, runs_dir=RUNS_DIR, extra_files=extras):
            if ev.kind == "log":
                log.append(ev.message)
                log = log[-400:]
                yield "\n".join(log), skip, skip, skip, skip, skip, skip
            elif ev.kind == "progress":
                progress(ev.fraction, desc=ev.message)
                log.append(f"» {ev.message}")
                yield "\n".join(log[-400:]), skip, skip, skip, skip, skip, skip
            elif ev.kind == "done":
                r = ev.result
                gallery = overlay_images(r.dataset_dir, "train", limit=16)
                yield ("\n".join(log + [ev.message]), gallery, class_plot(r.stats),
                       summary_markdown(r.stats, r.seconds), str(r.zip_path), train_snippet(r.dataset_dir),
                       str(r.dataset_dir))
    except (RuntimeError, ValueError) as exc:
        raise gr.Error(str(exc)[-1500:], duration=None) from exc


# ---- inspector -------------------------------------------------------------------------

def open_last(path):
    if not path:
        raise gr.Error("Generate a dataset first.")
    return path


def on_inspect_zip(zip_file):
    if not zip_file:
        raise gr.Error("Upload a dataset .zip first.")
    dest = RUNS_DIR / "_inspect" / datetime.now().strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(dest)
    try:
        root = find_dataset_root(dest)
    except FileNotFoundError as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise gr.Error(str(exc)) from exc
    return str(root)


def inspector_choices(root: str | None):
    if not root:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=[])
    splits = [s for s in SPLITS if load_split(root, s) is not None]
    coco = load_split(root, splits[0]) if splits else None
    classes = [c["name"] for c in coco["categories"]] if coco else []
    return (gr.update(choices=splits, value=splits[0] if splits else None),
            gr.update(choices=classes, value=classes))


def inspector_render(root, split, classes, limit):
    if not root or not split:
        return [], pd.DataFrame({"class": [], "instances": []}), ""
    coco = load_split(root, split)
    wanted = {c["id"] for c in coco["categories"] if c["name"] in (classes or [])}
    gallery = overlay_images(root, split, limit=int(limit), category_filter=wanted)
    stats = dataset_stats(root)
    return gallery, class_plot(stats), summary_markdown(stats)


# --------------------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    FIELDS.clear()
    d = GenerationConfig()

    with gr.Blocks(title="CAD2COCO") as demo:
        gr.HTML(HERO)
        last_dataset = gr.State(None)

        with gr.Tabs():
            # ============================ GENERATE ============================
            with gr.Tab("Generate", id="generate"):
                with gr.Row(equal_height=False):
                    # ---- 1. models --------------------------------------------------------
                    with gr.Column(scale=5, min_width=320):
                        step(1, "CAD models")
                        uploads = gr.File(
                            label="Models (+ optional .mtl / textures)", file_count="multiple", type="filepath",
                            file_types=[*sorted(MODEL_EXTENSIONS), ".mtl", ".png", ".jpg", ".jpeg"], height=150)
                        class_table = gr.Dataframe(
                            headers=["model file", "class name"], datatype=["str", "str"], interactive=True,
                            static_columns=[0], label="Classes — edit names; same name = same class",
                            value=pd.DataFrame({"model file": [], "class name": []}))
                        preview_pick = gr.Dropdown(label="Preview", choices=[], interactive=True)
                        viewer = gr.Model3D(label="3D preview", height=300, clear_color=(0.93, 0.94, 0.96, 1.0),
                                           camera_position=(45, 65, None))

                    # ---- 2. randomisation -------------------------------------------------
                    with gr.Column(scale=7, min_width=380):
                        step(2, "Randomisation")
                        with gr.Row():
                            preset = gr.Radio(list(PRESETS), label="Preset", value=None, scale=3)
                            reset_btn = gr.Button("Reset", size="sm", scale=1)
                        with gr.Tabs():
                            with gr.Tab("Dataset"):
                                reg("dataset.num_images", gr.Slider(1, 5000, d.dataset.num_images, step=1,
                                                                    label="Images"))
                                with gr.Row():
                                    reg("dataset.width", gr.Slider(256, 1920, d.dataset.width, step=32, label="Width"))
                                    reg("dataset.height", gr.Slider(256, 1920, d.dataset.height, step=32,
                                                                    label="Height"))
                                with gr.Row():
                                    reg("dataset.valid_ratio", gr.Slider(0, 0.4, d.dataset.valid_ratio, step=0.05,
                                                                         label="Valid split"))
                                    reg("dataset.test_ratio", gr.Slider(0, 0.4, d.dataset.test_ratio, step=0.05,
                                                                        label="Test split"))
                                reg("dataset.seed", gr.Number(d.dataset.seed, precision=0, label="Seed"))
                            with gr.Tab("Scene"):
                                with gr.Row():
                                    reg("scene.instances_min", gr.Slider(1, 12, d.scene.instances_min, step=1,
                                                                         label="Parts per image (min)"))
                                    reg("scene.instances_max", gr.Slider(1, 12, d.scene.instances_max, step=1,
                                                                         label="Parts per image (max)"))
                                reg("scene.distractors_max", gr.Slider(0, 16, d.scene.distractors_max, step=1,
                                                                       label="Distractor objects (max)",
                                                                       info="unlabelled clutter and occluders"))
                                reg("scene.placement", gr.Radio(
                                    [("Random 6-DoF pose", "free"), ("Physics drop onto floor", "physics")],
                                    value=d.scene.placement, label="Placement"))
                                with gr.Row():
                                    reg("scene.object_size", gr.Slider(0.02, 2.0, d.scene.object_size, step=0.01,
                                                                       label="Part size (m)",
                                                                       info="longest side after normalising"))
                                    reg("scene.size_jitter", gr.Slider(0, 0.5, d.scene.size_jitter, step=0.01,
                                                                       label="Size jitter (±)"))
                                with gr.Row():
                                    reg("scene.spread_radius", gr.Slider(0.0, 2.0, d.scene.spread_radius, step=0.01,
                                                                         label="Spread radius (m)"))
                                    reg("scene.floor_probability", gr.Slider(0, 1, d.scene.floor_probability,
                                                                             step=0.05, label="Floor probability"))
                            with gr.Tab("Camera"):
                                with gr.Row():
                                    reg("camera.dist_min", gr.Slider(0.05, 10, d.camera.dist_min, step=0.05,
                                                                     label="Distance min (m)"))
                                    reg("camera.dist_max", gr.Slider(0.05, 10, d.camera.dist_max, step=0.05,
                                                                     label="Distance max (m)"))
                                with gr.Row():
                                    reg("camera.elev_min", gr.Slider(0, 90, d.camera.elev_min, step=1,
                                                                     label="Elevation min (°)"))
                                    reg("camera.elev_max", gr.Slider(0, 90, d.camera.elev_max, step=1,
                                                                     label="Elevation max (°)"))
                                with gr.Row():
                                    reg("camera.fov_min", gr.Slider(20, 100, d.camera.fov_min, step=1,
                                                                    label="FOV min (°)"))
                                    reg("camera.fov_max", gr.Slider(20, 100, d.camera.fov_max, step=1,
                                                                    label="FOV max (°)"))
                                reg("camera.roll_max", gr.Slider(0, 180, d.camera.roll_max, step=1,
                                                                 label="In-plane roll (± °)"))
                            with gr.Tab("Appearance"):
                                reg("appearance.material_mode", gr.Radio(
                                    [("Random finishes", "random"), ("Keep model materials", "original"),
                                     ("Mix both", "mixed")], value=d.appearance.material_mode,
                                    label="Materials", info="metal / painted / plastic / rubber"))
                                with gr.Row():
                                    reg("appearance.lights_max", gr.Slider(1, 6, d.appearance.lights_max, step=1,
                                                                           label="Lights (max)"))
                                    reg("appearance.samples", gr.Slider(8, 512, d.appearance.samples, step=8,
                                                                        label="Render samples"))
                                with gr.Row():
                                    reg("appearance.light_min", gr.Slider(1, 2000, d.appearance.light_min, step=1,
                                                                          label="Light power min"))
                                    reg("appearance.light_max", gr.Slider(1, 2000, d.appearance.light_max, step=1,
                                                                          label="Light power max"))
                                with gr.Row():
                                    reg("appearance.noise_max", gr.Slider(0, 30, d.appearance.noise_max, step=0.5,
                                                                          label="Sensor noise (max σ)"))
                                    reg("appearance.blur_prob", gr.Slider(0, 1, d.appearance.blur_prob, step=0.05,
                                                                          label="Blur probability"))
                            with gr.Tab("Annotations"):
                                reg("annotation.min_visible_px", gr.Slider(
                                    0, 5000, d.annotation.min_visible_px, step=10, label="Min visible pixels",
                                    info="smaller (heavily occluded / tiny) instances are dropped"))
                                reg("annotation.polygon_tolerance", gr.Slider(
                                    0, 5, d.annotation.polygon_tolerance, step=0.25,
                                    label="Polygon simplification (px)",
                                    info="0 = keep every contour pixel"))
                                reg("annotation.min_part_area", gr.Slider(
                                    0, 500, d.annotation.min_part_area, step=1, label="Min polygon part area (px²)"))
                        with gr.Accordion("Save / load settings", open=False):
                            with gr.Row():
                                save_btn = gr.Button("Save config as YAML", size="sm")
                                load_file = gr.File(label="Load YAML", file_types=[".yaml", ".yml"], height=80)
                            saved_cfg = gr.File(label="Config", interactive=False, height=80)

                # ---- 3. generate ----------------------------------------------------------
                step(3, "Generate")
                with gr.Row():
                    gen_btn = gr.Button("Generate dataset", variant="primary", elem_id="generate-btn", scale=4)
                    stop_btn = gr.Button("Stop", variant="stop", scale=1)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7):
                        gallery = gr.Gallery(label="Annotated samples (train split)", columns=4, height=460,
                                             object_fit="contain")
                    with gr.Column(scale=4):
                        summary = gr.Markdown("*Nothing generated yet.*")
                        class_chart = gr.BarPlot(x="class", y="instances", title="Instances per class", height=220)
                        zip_out = gr.File(label="Download dataset (.zip)", interactive=False)
                with gr.Accordion("Train with it", open=False):
                    snippet = gr.Code(language="python", show_label=False)
                with gr.Accordion("Blender log", open=False):
                    log_box = gr.Code(language="shell", lines=14, max_lines=14, show_label=False)

            # ============================ INSPECT ============================
            with gr.Tab("Inspect dataset", id="inspect"):
                gr.Markdown("Browse any COCO dataset in this layout — e.g. one generated earlier.")
                inspect_root = gr.State(None)
                with gr.Row():
                    inspect_zip = gr.File(label="Dataset .zip", file_types=[".zip"], height=90, scale=3)
                    with gr.Column(scale=1):
                        open_zip_btn = gr.Button("Open zip")
                        open_last_btn = gr.Button("Open last generated", variant="secondary")
                with gr.Row():
                    insp_split = gr.Radio([], label="Split")
                    insp_classes = gr.CheckboxGroup([], label="Classes")
                    insp_limit = gr.Slider(4, 64, 16, step=4, label="Images")
                with gr.Row(equal_height=False):
                    insp_gallery = gr.Gallery(columns=4, height=520, object_fit="contain", scale=7, label="Samples")
                    with gr.Column(scale=4):
                        insp_summary = gr.Markdown()
                        insp_plot = gr.BarPlot(x="class", y="instances", title="Instances per class", height=220)

            # ============================ ABOUT ============================
            with gr.Tab("How it works", id="about"):
                gr.Markdown(ABOUT)

        # ---- wiring ------------------------------------------------------------------------
        fields = list(FIELDS.values())
        uploads.change(on_upload, uploads, [class_table, preview_pick, viewer])
        preview_pick.change(on_preview_pick, [uploads, preview_pick], viewer)
        preset.change(on_preset, preset, fields)
        reset_btn.click(lambda: [None, *values_from_config(GenerationConfig())], None, [preset, *fields])
        save_btn.click(on_save_config, fields, saved_cfg)
        load_file.change(on_load_config, load_file, fields)

        gen_event = gen_btn.click(
            on_generate, [uploads, class_table, *fields],
            [log_box, gallery, class_chart, summary, zip_out, snippet, last_dataset],
            concurrency_limit=1)
        stop_btn.click(None, None, None, cancels=[gen_event])

        insp_outputs = [insp_gallery, insp_plot, insp_summary]
        insp_inputs = [inspect_root, insp_split, insp_classes, insp_limit]
        open_zip_btn.click(on_inspect_zip, inspect_zip, inspect_root)
        open_last_btn.click(open_last, last_dataset, inspect_root)
        inspect_root.change(inspector_choices, inspect_root, [insp_split, insp_classes]).then(
            inspector_render, insp_inputs, insp_outputs)
        for comp in (insp_split, insp_classes, insp_limit):
            comp.input(inspector_render, insp_inputs, insp_outputs)

    return demo


ABOUT = """
### Pipeline

1. **Normalise** — every model is centred and scaled so its longest side is 1 m, then resized to *Part size*.
   CAD exports arrive in mm, cm, m or inches; normalising makes the unit irrelevant.
2. **Randomise** (per image) — number and choice of parts, pose (random 6-DoF or a physics drop),
   material finish (metal / painted / plastic / rubber), procedural floor texture, world colour,
   1–N coloured point lights, distractor primitives, camera distance / elevation / azimuth / FOV / roll,
   plus sensor noise, blur and JPEG quality.
3. **Render** with BlenderProc (Cycles) — RGB plus an instance-index pass where only the target parts
   get an id, so distractors correctly occlude them.
4. **Annotate** — each visible instance mask becomes outer-contour polygons (Douglas-Peucker simplified),
   a tight bbox and its pixel area. Too-small or fully hidden instances are dropped.
5. **Split & export** — `train/ valid/ test/`, each with `_annotations.coco.json`, the layout RF-DETR reads.
   The exact config is saved next to the data for reproducibility.

### Tips
* Start with the **Quick look** preset to check scale and framing, then scale up.
* If parts look tiny or out of frame, bring *Distance* closer to *Part size* (≈ 2–5× works well).
* Real test images are what tell you whether the sim-to-real gap is closed — keep a small labelled real set.
"""


def launch(**kwargs) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_app()
    theme = gr.themes.Soft(primary_hue="orange", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "sans-serif"],
                           font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"])
    demo.queue().launch(theme=theme, css=CSS, allowed_paths=[str(RUNS_DIR)], **kwargs)
