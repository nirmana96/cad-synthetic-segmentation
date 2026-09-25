"""Build the animated walkthrough (GIF + MP4) of the CAD2COCO pipeline.

Scenes: title -> load CAD -> randomise -> render -> annotate -> export -> Gradio app -> end card.

Every annotation shown comes from a real COCO dataset produced by cad2coco.coco; the
images come from whatever render folder you point it at. Use the output of a real
BlenderProc run for the final version:

    python scripts/demo/make_demo_gif.py \
        --render-dir runs/<run>/render --dataset-dir runs/<run>/dataset \
        --models examples/models/pipe_flange.obj examples/models/l_bracket.obj examples/models/pipe_tee.obj \
        --classes flange bracket tee --app-captures captures/ --out docs/demo

Requires ffmpeg on PATH, plus numba + trimesh for the CAD turntables.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from functools import cache, lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import preview_render as pr  # noqa: E402

from cad2coco.coco import SPLITS, dataset_stats, load_split  # noqa: E402
from cad2coco.visualize import category_color, draw_annotations  # noqa: E402

W, H = 1280, 720
FPS = 12.5

BG_TOP, BG_BOT = (11, 18, 32), (15, 23, 42)
PANEL, BORDER = (17, 26, 46), (31, 42, 68)
TEXT, MUTED, DIM = (241, 245, 249), (148, 163, 184), (71, 85, 105)
ACCENT = (249, 115, 22)
INSTANCE_COLORS = [(249, 115, 22), (34, 211, 238), (163, 230, 53), (232, 121, 249), (250, 204, 21),
                   (96, 165, 250), (251, 113, 133), (52, 211, 153)]
STEPS = ["Load CAD", "Randomise", "Render", "Annotate", "Export"]

CONTENT_TOP, CONTENT_BOTTOM = 100, 606


# --------------------------------------------------------------------------------------
# fonts and drawing helpers
# --------------------------------------------------------------------------------------

# IBM Plex (SIL Open Font License, see fonts/OFL.txt), bundled so the output looks the same everywhere
FONT_FILES = {
    "sans": "IBMPlexSans-Regular", "sans-semi": "IBMPlexSans-Bold", "sans-bold": "IBMPlexSans-Bold",
    "mono": "IBMPlexMono-Regular", "mono-semi": "IBMPlexMono-Bold",
}


@cache
def font(key: str, size: int) -> ImageFont.FreeTypeFont:
    path = HERE / "fonts" / f"{FONT_FILES[key]}.ttf"
    if path.exists():
        return ImageFont.truetype(str(path), size)
    fallback = "DejaVuSansMono.ttf" if key.startswith("mono") else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(fallback, size)
    except OSError:
        return ImageFont.load_default(size)


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def clamp01(t: float) -> float:
    return min(max(t, 0.0), 1.0)


def mix(c1, c2, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2, strict=True))


@lru_cache(maxsize=1)
def _background() -> np.ndarray:
    t = np.linspace(0, 1, H)[:, None, None]
    bg = np.array(BG_TOP)[None, None, :] * (1 - t) + np.array(BG_BOT)[None, None, :] * t
    bg = np.broadcast_to(bg, (H, W, 3)).copy()
    # soft accent glow top-right
    yy, xx = np.mgrid[0:H, 0:W]
    glow = np.exp(-(((xx - 1150) / 520) ** 2 + ((yy + 80) / 300) ** 2))
    bg += glow[..., None] * np.array([40, 16, 0])
    return np.clip(bg, 0, 255).astype(np.uint8)


def new_canvas() -> Image.Image:
    return Image.fromarray(_background())


def rounded_mask(w, h, r):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], r, fill=255)
    return m


def paste_image(canvas: Image.Image, img, xy, radius=12, border=True, opacity=1.0):
    im = img if isinstance(img, Image.Image) else Image.fromarray(img)
    mask = rounded_mask(im.width, im.height, radius)
    if opacity < 1:
        mask = mask.point(lambda v: int(v * opacity))
    canvas.paste(im.convert("RGB"), xy, mask)
    if border:
        ImageDraw.Draw(canvas).rounded_rectangle(
            [xy[0], xy[1], xy[0] + im.width - 1, xy[1] + im.height - 1], radius, outline=BORDER, width=2)


def panel(draw, box, radius=14, fill=PANEL):
    draw.rounded_rectangle(box, radius, fill=fill, outline=BORDER, width=2)


def chip(draw, xy, label, fill=(30, 41, 59), color=TEXT, fnt=None, pad=(12, 6), outline=None, dot=None):
    fnt = fnt or font("sans-semi", 16)
    x, y = xy
    tw = draw.textlength(label, font=fnt)
    extra = 16 if dot else 0
    h = fnt.size + 2 * pad[1]
    draw.rounded_rectangle([x, y, x + tw + 2 * pad[0] + extra, y + h], h // 2, fill=fill, outline=outline,
                           width=1 if outline else 0)
    if dot:
        cy = y + h // 2
        draw.ellipse([x + pad[0] - 2, cy - 5, x + pad[0] + 8, cy + 5], fill=dot)
    draw.text((x + pad[0] + extra, y + h / 2), label, font=fnt, fill=color, anchor="lm")
    return x + tw + 2 * pad[0] + extra


def draw_check(draw, cx, cy, s, color):
    draw.line([(cx - s, cy), (cx - s * 0.3, cy + s * 0.7), (cx + s, cy - s * 0.7)], fill=color, width=3)


def logo(draw, x, y, s=30):
    draw.rounded_rectangle([x, y, x + s, y + s], 8, fill=ACCENT)
    pts = [(x + s * 0.22, y + s * 0.72), (x + s * 0.38, y + s * 0.26), (x + s * 0.78, y + s * 0.34),
           (x + s * 0.7, y + s * 0.74)]
    draw.line(pts + [pts[0]], fill=(255, 255, 255), width=2)
    for p in pts:
        draw.ellipse([p[0] - 2.5, p[1] - 2.5, p[0] + 2.5, p[1] + 2.5], fill=(255, 255, 255))


def title_lockup(d, cy, color, size=86, logo_size=60, gap=22):
    f = font("sans-bold", size)
    tw = d.textlength("CAD2COCO", font=f)
    x = W / 2 - (logo_size + gap + tw) / 2
    logo(d, int(x), int(cy - logo_size / 2), logo_size)
    d.text((x + logo_size + gap, cy), "CAD2COCO", font=f, fill=color, anchor="lm")


def frame_chrome(img: Image.Image, active: int | None, caption: str, sub: str, progress: float,
                 all_done=False) -> ImageDraw.ImageDraw:
    d = ImageDraw.Draw(img)
    logo(d, 40, 23)
    d.text((82, 38), "CAD2COCO", font=font("sans-bold", 26), fill=TEXT, anchor="lm")
    # step pills, right aligned
    f = font("sans-semi", 15)
    widths = [d.textlength(f"{i + 1}  {s}", font=f) + 30 for i, s in enumerate(STEPS)]
    x = W - 40 - sum(widths) - 8 * (len(STEPS) - 1)
    for i, (label, w) in enumerate(zip(STEPS, widths, strict=True)):
        box = [x, 21, x + w, 55]
        done = all_done or (active is not None and i < active)
        if active == i:
            d.rounded_rectangle(box, 17, fill=ACCENT)
            d.text((x + 15, 38), f"{i + 1}  {label}", font=f, fill=(15, 23, 42), anchor="lm")
        elif done:
            d.rounded_rectangle(box, 17, fill=(30, 41, 59))
            d.text((x + 15, 38), f"{i + 1}  {label}", font=f, fill=(203, 213, 225), anchor="lm")
        else:
            d.rounded_rectangle(box, 17, outline=(51, 65, 85), width=1)
            d.text((x + 15, 38), f"{i + 1}  {label}", font=f, fill=DIM, anchor="lm")
        x += w + 8
    d.line([(40, 76), (W - 40, 76)], fill=(30, 41, 59), width=1)
    # caption
    d.text((40, 640), caption, font=font("sans-semi", 27), fill=TEXT, anchor="lm")
    d.text((40, 676), sub, font=font("sans", 18), fill=MUTED, anchor="lm")
    # timeline
    d.rectangle([0, H - 5, W, H], fill=(30, 41, 59))
    d.rectangle([0, H - 5, int(W * progress), H], fill=ACCENT)
    return d


def fit(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def read_rgb(path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------


class Data:
    def __init__(self, args):
        self.render_dir = Path(args.render_dir)
        self.dataset_dir = Path(args.dataset_dir)
        self.frames = [json.loads(line) for line in (self.render_dir / "frames.jsonl").read_text().splitlines()
                       if line.strip()]
        meta_path = self.render_dir / "scene_meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else [None] * len(self.frames)
        self.coco = {s: load_split(self.dataset_dir, s) for s in SPLITS}
        self.names = {}
        for c in self.coco.values():
            if c:
                self.names.update({cat["id"]: cat["name"] for cat in c["categories"]})
        self.stats = dataset_stats(self.dataset_dir)
        self.index = {}  # file name -> (split, image entry, annotations)
        for split, c in self.coco.items():
            if not c:
                continue
            anns = {}
            for a in c["annotations"]:
                anns.setdefault(a["image_id"], []).append(a)
            for img in c["images"]:
                self.index[img["file_name"]] = (split, img, anns.get(img["id"], []))
        self.models = [Path(m) for m in args.models]
        self.classes = args.classes or [m.stem for m in self.models]
        self.scenes = args.scenes or list(range(min(8, len(self.frames))))
        self.hero = args.hero if args.hero is not None else self._pick_hero()
        self.grid = args.grid or self._pick_grid()

    def image(self, i) -> np.ndarray:
        return read_rgb(self.render_dir / self.frames[i]["image"])

    def instances(self, i) -> np.ndarray:
        return np.load(self.render_dir / self.frames[i]["mask"])["instances"]

    def annotations(self, i):
        return self.index.get(Path(self.frames[i]["image"]).name, (None, None, []))

    def _pick_hero(self):
        return max(range(len(self.frames)),
                   key=lambda i: (min(len(self.frames[i]["instances"]), 4),
                                  sum(x["visible_px"] for x in self.frames[i]["instances"])))

    def _pick_grid(self):
        picks = [i for i in range(len(self.frames)) if i not in self.scenes and i != self.hero]
        return (picks + self.scenes)[:9]

    def scene_rows(self, i):
        fr = self.frames[i]
        cats = [self.names.get(x["category_id"], str(x["category_id"])) for x in fr["instances"]]
        m = self.meta[i] if i < len(self.meta) else None
        rows = [("Parts", f"{len(cats)} · " + ", ".join(cats))]
        if m:
            cam = m["camera"]
            rows += [
                ("Finish", " · ".join(m["finishes"])),
                ("Lights", f"{m['lights']} point light{'s' if m['lights'] != 1 else ''}"),
                ("Camera", f"{cam['distance']:.2f} m · {cam['elevation']:.0f}° elev · FOV {cam['fov']:.0f}°"),
                ("Roll", f"{cam['roll']:+.0f}°"),
                ("Floor", {"none": "off (world colour)", "noise": "procedural noise",
                           "tiles": "procedural tiles"}[m["floor"]]),
                ("Distractors", f"{m['distractors']} primitive{'s' if m['distractors'] != 1 else ''}"),
            ]
        elif "camera_to_world" in fr:
            c2w = np.array(fr["camera_to_world"])
            K = np.array(fr["camera_K"])
            centre = np.mean([np.array(x["object_to_world"])[:3, 3] for x in fr["instances"]], axis=0)
            v = c2w[:3, 3] - centre
            fov = math.degrees(2 * math.atan(fr["width"] / (2 * K[0][0])))
            rows += [("Camera", f"{np.linalg.norm(v):.2f} m · "
                                f"{math.degrees(math.asin(v[2] / np.linalg.norm(v))):.0f}° elev · FOV {fov:.0f}°")]
        return rows


# --------------------------------------------------------------------------------------
# scenes
# --------------------------------------------------------------------------------------


def scene_title(n, total_before, total):
    for k in range(n):
        img = new_canvas()
        d = ImageDraw.Draw(img)
        t = ease(k / (n * 0.55))
        col = mix(BG_BOT, TEXT, t)
        title_lockup(d, 274 - int(20 * (1 - t)), col)
        d.text((W // 2, 362), "CAD models in  →  annotated segmentation dataset out",
               font=font("sans", 30), fill=mix(BG_BOT, MUTED, ease(k / (n * 0.8))), anchor="mm")
        if k > n * 0.35:
            labels = ["BlenderProc", "Domain randomisation", "COCO polygons", "Gradio app"]
            fnt = font("sans-semi", 18)
            widths = [d.textlength(s, font=fnt) + 28 for s in labels]
            x = W // 2 - (sum(widths) + 12 * (len(labels) - 1)) / 2
            for s, w in zip(labels, widths, strict=True):
                chip(d, (x, 418), s, fill=(40, 26, 18), color=(253, 186, 116), fnt=fnt, outline=(124, 58, 18))
                x += w + 12
        d.rectangle([0, H - 5, W, H], fill=(30, 41, 59))
        d.rectangle([0, H - 5, int(W * (total_before + k) / total), H], fill=ACCENT)
        yield img


def scene_load(data: Data, n, t0, total):
    size = 330
    parts = [pr.load_part(m) for m in data.models]
    raw = []
    for m in data.models:
        mesh = pr.trimesh.load(m, force="mesh")
        raw.append(np.round(mesh.extents, 1))
    spins = [pr.turntable(pr._shaded(p), n, size, size) for p in parts]
    k_parts = len(parts)
    pw = 380
    gap = (W - 80 - k_parts * pw) // max(k_parts - 1, 1)
    for k in range(n):
        img = new_canvas()
        d = frame_chrome(img, 0, "Load CAD models",
                         "Any mesh format (.obj .stl .ply .glb .fbx). Every part is centred and scaled, "
                         "so mm / cm / inch exports all behave the same.", (t0 + k) / total)
        for j in range(k_parts):
            x = 40 + j * (pw + gap)
            appear = ease((k - 3 * j) / 8)
            y = CONTENT_TOP + 8 + int(24 * (1 - appear))
            panel(d, [x, y, x + pw, y + 486])
            rgb, alpha = spins[j][k]
            comp = rgb.astype(np.float32) + np.array(PANEL, np.float32)[None, None, :] * (1 - alpha[..., None])
            tile = Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8))
            img.paste(tile, (x + (pw - size) // 2, y + 18), rounded_mask(size, size, 10) if appear >= 1 else
                      rounded_mask(size, size, 10).point(lambda v, a=appear: int(v * a)))
            d.text((x + 24, y + 372), data.models[j].name, font=font("mono-semi", 19), fill=TEXT, anchor="lm")
            e = raw[j]
            d.text((x + 24, y + 402), f"raw extents {e[0]:g} × {e[1]:g} × {e[2]:g}", font=font("sans", 16),
                   fill=MUTED, anchor="lm")
            if k > 14 + 3 * j:
                cat = category_color(j + 1)
                chip(d, (x + 22, y + 428), f"class {j + 1} · {data.classes[j]}", fill=(30, 41, 59),
                     fnt=font("sans-semi", 15), dot=cat)
        yield img


def scene_randomise(data: Data, n, t0, total):
    size = 486
    scenes = data.scenes
    per = n // len(scenes)
    imgs = [fit(data.image(i), size) for i in scenes]
    thumbs = [fit(data.image(i), 70) for i in scenes]
    rows_all = [data.scene_rows(i) for i in scenes]
    for k in range(n):
        idx = min(k // per, len(scenes) - 1)
        local = k - idx * per
        img = new_canvas()
        d = frame_chrome(img, 1, "Randomise every image",
                         "Pose, material finish, lights, floor texture, camera and clutter change "
                         "on every frame (domain randomisation).", (t0 + k) / total)
        cur = imgs[idx]
        if local < 2 and idx > 0:
            a = (local + 1) / 3
            cur = (imgs[idx - 1] * (1 - a) + imgs[idx] * a).astype(np.uint8)
        paste_image(img, cur, (40, CONTENT_TOP + 4))
        x0, x1 = 560, W - 40
        panel(d, [x0, CONTENT_TOP + 4, x1, CONTENT_TOP + 4 + size])
        d.text((x0 + 28, CONTENT_TOP + 38), "Sampled for this image", font=font("sans-semi", 20), fill=TEXT,
               anchor="lm")
        d.text((x1 - 28, CONTENT_TOP + 38), f"image {idx + 1} / {len(scenes)}", font=font("mono", 16),
               fill=MUTED, anchor="rm")
        prev = dict(rows_all[idx - 1]) if idx > 0 else {}
        y = CONTENT_TOP + 84
        for label, value in rows_all[idx]:
            changed = prev.get(label) != value and idx > 0
            flash = changed and local < 4
            d.text((x0 + 28, y), label, font=font("sans", 17), fill=MUTED, anchor="lm")
            vcol = ACCENT if flash else TEXT
            d.text((x0 + 170, y), value, font=font("mono-semi", 18), fill=vcol, anchor="lm")
            y += 40
        ty = CONTENT_TOP + size - 70 - 18
        for j, th in enumerate(thumbs):
            tx = x0 + 28 + j * 78
            if tx + 70 > x1 - 20:
                break
            paste_image(img, th, (tx, ty), radius=8, border=False, opacity=1.0 if j <= idx else 0.35)
            if j == idx:
                d.rounded_rectangle([tx - 3, ty - 3, tx + 72, ty + 72], 10, outline=ACCENT, width=3)
        yield img


def colorize_instances(inst: np.ndarray) -> np.ndarray:
    out = np.zeros(inst.shape + (3,), np.uint8)
    out[:] = (9, 13, 24)
    for i in np.unique(inst):
        if i == 0:
            continue
        out[inst == i] = INSTANCE_COLORS[(int(i) - 1) % len(INSTANCE_COLORS)]
    return out


def scene_render(data: Data, n, t0, total):
    size = 430
    rgb = fit(data.image(data.hero), size)
    inst = data.instances(data.hero)
    ids = cv2.resize(colorize_instances(inst), (size, size), interpolation=cv2.INTER_NEAREST)
    fr = data.frames[data.hero]
    gap = 110
    xl = (W - 2 * size - gap) // 2
    xr = xl + size + gap
    y = CONTENT_TOP + 38
    for k in range(n):
        img = new_canvas()
        d = frame_chrome(img, 2, "Render RGB + an instance-ID pass",
                         "Only target parts get an ID. Distractors render as background, "
                         "so occlusions cut into the masks correctly.", (t0 + k) / total)
        d.text((xl, CONTENT_TOP + 16), "RGB", font=font("sans-semi", 19), fill=TEXT, anchor="lm")
        d.text((xr, CONTENT_TOP + 16), "Instance IDs", font=font("sans-semi", 19), fill=TEXT, anchor="lm")
        paste_image(img, rgb, (xl, y))
        w = ease((k - 6) / 16)
        cut = int(size * w)
        right = rgb.copy()
        right[:, :cut] = ids[:, :cut]
        paste_image(img, right, (xr, y))
        if 0 < w < 1:
            d.line([(xr + cut, y), (xr + cut, y + size)], fill=ACCENT, width=3)
        # arrow between
        ax = xl + size + 18
        a = ease((k - 2) / 8)
        if a > 0:
            d.line([(ax, y + size / 2), (ax + (gap - 36) * a, y + size / 2)], fill=ACCENT, width=4)
            if a >= 1:
                tip = ax + gap - 36
                d.polygon([(tip + 10, y + size / 2), (tip - 4, y + size / 2 - 9), (tip - 4, y + size / 2 + 9)],
                          fill=ACCENT)
        if k > 22:
            lx = xr
            for x in fr["instances"][:5]:
                col = INSTANCE_COLORS[(x["id"] - 1) % len(INSTANCE_COLORS)]
                lx = chip(d, (lx, y + size + 14), f"id {x['id']} · {data.names.get(x['category_id'])}",
                          fill=(30, 41, 59), fnt=font("sans-semi", 14), dot=col, pad=(9, 5)) + 8
        yield img


def json_lines(ann, img_entry, name, idx, count):
    seg = ann["segmentation"][0]
    shown = ", ".join(f"{v:g}" for v in seg[:6])
    nverts = sum(len(p) // 2 for p in ann["segmentation"])
    bbox = ", ".join(f"{v:g}" for v in ann["bbox"])
    return [
        (f"// annotation {idx + 1} of {count} · {name}", MUTED),
        ("{", TEXT),
        (f'  "id": {ann["id"]},', TEXT),
        (f'  "image_id": {ann["image_id"]},   // {img_entry["file_name"]}', TEXT),
        (f'  "category_id": {ann["category_id"]},', TEXT),
        (f'  "segmentation": [[{shown}, …]],', TEXT),
        (f"  //  {nverts} polygon vertices, {len(ann['segmentation'])} part(s)", MUTED),
        (f'  "bbox": [{bbox}],', TEXT),
        (f'  "area": {ann["area"]:g},', TEXT),
        ('  "iscrowd": 0', TEXT),
        ("}", TEXT),
    ]


def scene_annotate(data: Data, n, t0, total):
    size = 486
    split, img_entry, anns = data.annotations(data.hero)
    anns = sorted(anns, key=lambda a: -a["area"])[:4]
    base = data.image(data.hero)
    scale = size / base.shape[1]
    base_small = fit(base, size)
    # the first annotation is drawn and typed slowly enough to read; the rest follow faster
    timing = [dict(start=0, draw=14, fill=6, type_at=3, type_len=22)]
    for i in range(1, len(anns)):
        timing.append(dict(start=42 + 12 * (i - 1), draw=7, fill=4, type_at=0, type_len=6))
    ss = 2
    for k in range(n):
        img = new_canvas()
        d = frame_chrome(img, 3, "Masks become COCO polygons",
                         "Outer contours (Douglas–Peucker simplified), plus bbox and pixel area. "
                         "No manual labelling.", (t0 + k) / total)
        canvas = cv2.resize(base_small, (size * ss, size * ss), interpolation=cv2.INTER_LINEAR)
        cur = max(i for i, tm in enumerate(timing) if tm["start"] <= k)
        local = k - timing[cur]["start"]
        tc = timing[cur]
        for i, ann in enumerate(anns[:cur + 1]):
            col = category_color(ann["category_id"])
            prog = 1.0 if i < cur else clamp01(local / tc["draw"])
            fill_a = 0.32 if i < cur else 0.32 * clamp01((local - tc["draw"]) / tc["fill"])
            polys = [np.array(p, np.float32).reshape(-1, 2) * scale * ss for p in ann["segmentation"]]
            if fill_a > 0:
                over = canvas.copy()
                cv2.fillPoly(over, [np.round(p).astype(np.int32) for p in polys], col)
                canvas = cv2.addWeighted(over, fill_a, canvas, 1 - fill_a, 0)
            for p in polys:
                closed = np.vstack([p, p[:1]])
                seglen = np.linalg.norm(np.diff(closed, axis=0), axis=1)
                cum = np.concatenate([[0], np.cumsum(seglen)])
                target = cum[-1] * prog
                m = int(np.searchsorted(cum, target))
                pts = closed[:max(m, 1)]
                if m < len(closed) and m > 0:
                    f = (target - cum[m - 1]) / max(seglen[m - 1], 1e-6)
                    pts = np.vstack([pts, closed[m - 1] + (closed[m] - closed[m - 1]) * f])
                cv2.polylines(canvas, [np.round(pts).astype(np.int32)], False, col, 3 * ss // 2 + 1, cv2.LINE_AA)
                for v in p[: max(1, int(len(p) * prog))]:
                    cv2.circle(canvas, (int(v[0]), int(v[1])), 3 * ss // 2, (255, 255, 255), -1, cv2.LINE_AA)
                if 0 < prog < 1:
                    cv2.circle(canvas, tuple(np.round(pts[-1]).astype(int)), 6 * ss // 2, col, -1, cv2.LINE_AA)
        shown = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
        paste_image(img, shown, (40, CONTENT_TOP + 4))
        # label chips on image
        for i, ann in enumerate(anns[:cur + 1]):
            if i == cur and local < tc["draw"]:
                continue
            x, y0 = ann["bbox"][0] * scale + 40, ann["bbox"][1] * scale + CONTENT_TOP + 4
            chip(d, (max(44, x), max(CONTENT_TOP + 8, y0 - 26)), data.names[ann["category_id"]],
                 fill=category_color(ann["category_id"]), color=(15, 23, 42), fnt=font("sans-semi", 13),
                 pad=(7, 3))
        # json panel
        x0, x1 = 560, W - 40
        panel(d, [x0, CONTENT_TOP + 4, x1, CONTENT_TOP + 4 + size], fill=(13, 20, 36))
        d.text((x0 + 24, CONTENT_TOP + 34), f"{split}/_annotations.coco.json", font=font("mono-semi", 16),
               fill=MUTED, anchor="lm")
        for i in range(3):
            d.ellipse([x1 - 30 - i * 20, CONTENT_TOP + 28, x1 - 20 - i * 20, CONTENT_TOP + 38],
                      fill=[(248, 113, 113), (250, 204, 21), (74, 222, 128)][i])
        ann = anns[cur]
        lines = json_lines(ann, img_entry, data.names[ann["category_id"]], cur, len(anns))
        chars = int(sum(len(s) for s, _ in lines) * clamp01((local - tc["type_at"]) / tc["type_len"]))
        y = CONTENT_TOP + 74
        fnt = font("mono", 16)
        for s, colr in lines:
            part = s[:max(0, chars)]
            chars -= len(s)
            if s.startswith('  "category_id"') and part:
                colr = category_color(ann["category_id"])
            d.text((x0 + 24, y), part, font=fnt, fill=colr, anchor="lm")
            y += 30
        yield img


def scene_export(data: Data, n, t0, total):
    cell, gap = 156, 9
    tiles = []
    for i in data.grid:
        split, entry, anns = data.annotations(i)
        tiles.append(fit(draw_annotations(data.image(i), anns, data.names, alpha=0.35), cell))
    st = data.stats
    for k in range(n):
        img = new_canvas()
        d = frame_chrome(img, 4, "Export train / valid / test splits",
                         "COCO JSON per split, same layout RF-DETR reads directly. "
                         "The config is saved alongside for reproducibility.", (t0 + k) / total)
        for j, tile in enumerate(tiles):
            a = ease((k - 2 * j) / 5)
            if a <= 0:
                continue
            r, c = divmod(j, 3)
            x, y = 40 + c * (cell + gap), CONTENT_TOP + 4 + r * (cell + gap)
            paste_image(img, tile, (x, y), radius=8, border=False, opacity=a)
        x0, x1 = 560, W - 40
        panel(d, [x0, CONTENT_TOP + 4, x1, CONTENT_TOP + 490])
        c = ease((k - 6) / 22)
        y = CONTENT_TOP + 40
        d.text((x0 + 28, y), "dataset/", font=font("mono-semi", 20), fill=TEXT, anchor="lm")
        y += 38
        splits = [s for s in SPLITS if s in st["per_split"]]
        for i, s in enumerate(splits):
            branch = "└─" if i == len(splits) - 1 else "├─"
            v = st["per_split"][s]
            d.text((x0 + 28, y), f"{branch} {s}/", font=font("mono-semi", 19), fill=TEXT, anchor="lm")
            d.text((x0 + 170, y), f"{round(v['images'] * c):>4} images  {round(v['instances'] * c):>4} instances",
                   font=font("mono", 18), fill=MUTED, anchor="lm")
            y += 36
        d.text((x0 + 28, y), "   cad2coco_config.yaml", font=font("mono", 18), fill=DIM, anchor="lm")
        y += 52
        d.text((x0 + 28, y), "Instances per class", font=font("sans-semi", 18), fill=TEXT, anchor="lm")
        y += 34
        top = max(st["per_class"].values()) if st["per_class"] else 1
        bw = x1 - x0 - 28 - 150 - 90
        for cid, name in sorted(data.names.items()):
            v = st["per_class"].get(name, 0)
            col = category_color(cid)
            d.text((x0 + 28, y), name, font=font("sans", 17), fill=MUTED, anchor="lm")
            L = int(bw * v / top * c)
            d.rounded_rectangle([x0 + 150, y - 11, x0 + 150 + max(L, 4), y + 11], 5, fill=col)
            d.text((x0 + 160 + max(L, 4), y), f"{round(v * c)}", font=font("mono-semi", 16), fill=TEXT, anchor="lm")
            y += 36
        if k > n * 0.55:
            a = ease((k - n * 0.55) / 6)
            yy = CONTENT_TOP + 440
            d.text((x0 + 28, yy), "Train with", font=font("sans", 17), fill=mix(PANEL, MUTED, a), anchor="lm")
            x = x0 + 120
            for s in ("RF-DETR", "YOLO-seg", "Detectron2"):
                x = chip(d, (x, yy - 15), s, fill=mix(PANEL, (40, 26, 18), a), color=mix(PANEL, (253, 186, 116), a),
                         fnt=font("sans-semi", 15), outline=mix(PANEL, (124, 58, 18), a)) + 10
        yield img


def cursor(draw, x, y, click=0.0):
    if click > 0:
        r = 10 + 26 * click
        draw.ellipse([x - r, y - r, x + r, y + r], outline=mix((255, 255, 255), BG_TOP, click), width=3)
    pts = [(x, y), (x, y + 24), (x + 6, y + 18), (x + 11, y + 28), (x + 15, y + 26), (x + 10, y + 16), (x + 18, y + 16)]
    draw.polygon(pts, fill=(255, 255, 255), outline=(15, 23, 42))


def scene_app(cap_dir: Path, t0, total):
    """Screen captures of the real Gradio app (1280x720 CSS px at any device scale)."""
    clicks = json.loads((cap_dir / "clicks.json").read_text()) if (cap_dir / "clicks.json").exists() else {}
    sample = Image.open(cap_dir / "a1.png")
    dpr = sample.width / 1280
    box_w = W - 80
    crops = {"A": (0, 170, 1280, 720), "B": (0, 52, 1280, 602), "C": (0, 116, 1280, 666)}
    box_h = int(box_w * 550 / 1280)
    progress_frames = sorted(p for p in cap_dir.glob("b*.png") if p.stem != "b00")
    picks = sorted(set(np.linspace(0, len(progress_frames) - 1, min(16, len(progress_frames))).round().astype(int)))
    plan = [("a1", "A", 12, None)]
    plan += [(f"a{i}", "A", 6, f"a{i}") for i in (2, 3, 4)]
    plan += [("a5", "A", 9, "a5"), ("b00", "B", 7, "b00")]
    plan += [(progress_frames[i].stem, "B", 1, None) for i in picks]
    plan += [("c0", "C", 26, "zoom")]
    frames = []
    for name, crop_key, reps, extra in plan:
        path = cap_dir / f"{name}.png"
        if path.exists():
            frames += [(path, crop_key, r, reps, extra) for r in range(reps)]
    y0 = CONTENT_TOP - 4
    for k, (path, crop_key, r, reps, extra) in enumerate(frames):
        img = new_canvas()
        d = frame_chrome(img, None, "All from a Gradio app",
                         "Upload models, name classes, pick a preset, generate. "
                         "Or run it headless: cad2coco generate part.obj -n 500", (t0 + k) / total, all_done=True)
        shot = Image.open(path).convert("RGB")
        cx0, cy0, cx1, cy1 = crops[crop_key]
        if extra == "zoom":
            z = 1 + 0.10 * ease(r / reps)
            cw, ch = (cx1 - cx0) / z, (cy1 - cy0) / z
            cx0, cy0 = cx0 + (cx1 - cx0 - cw) * 0.3, cy0 + (cy1 - cy0 - ch) * 0.3
            cx1, cy1 = cx0 + cw, cy0 + ch
        crop = shot.crop(tuple(int(v * dpr) for v in (cx0, cy0, cx1, cy1))).resize((box_w, box_h), Image.LANCZOS)
        # drop shadow
        sh = Image.new("L", (box_w + 40, box_h + 40), 0)
        ImageDraw.Draw(sh).rounded_rectangle([20, 24, box_w + 20, box_h + 24], 16, fill=110)
        sh = sh.filter(ImageFilter.GaussianBlur(12))
        img.paste(Image.new("RGB", sh.size, (0, 0, 0)), (20, y0 - 20), sh)
        paste_image(img, crop, (40, y0), radius=14)
        if extra in clicks:
            px, py = clicks[extra]
            sx = 40 + (px - crops[crop_key][0]) * box_w / (crops[crop_key][2] - crops[crop_key][0])
            sy = y0 + (py - crops[crop_key][1]) * box_h / (crops[crop_key][3] - crops[crop_key][1])
            cursor(d, sx, sy, click=clamp01(r / 4) if r < 4 else 0.0)
        yield img


def scene_end(n, t0, total, footer):
    for k in range(n):
        img = new_canvas()
        d = ImageDraw.Draw(img)
        a = ease(k / 8)
        title_lockup(d, 242, mix(BG_BOT, TEXT, a))
        d.text((W // 2, 326), "Synthetic instance-segmentation data from CAD, in minutes",
               font=font("sans", 28), fill=mix(BG_BOT, MUTED, a), anchor="mm")
        labels = ["Python", "BlenderProc", "Gradio", "OpenCV", "pycocotools"]
        fnt = font("sans-semi", 17)
        widths = [d.textlength(s, font=fnt) + 24 for s in labels]
        x = W // 2 - (sum(widths) + 10 * (len(labels) - 1)) / 2
        for s, w in zip(labels, widths, strict=True):
            chip(d, (x, 380), s, fill=mix(BG_BOT, (30, 41, 59), a), color=mix(BG_BOT, (203, 213, 225), a), fnt=fnt)
            x += w + 10
        if footer:
            d.text((W // 2, 460), footer, font=font("sans-semi", 20), fill=mix(BG_BOT, ACCENT, a),
                   anchor="mm")
        d.rectangle([0, H - 5, W, H], fill=(30, 41, 59))
        d.rectangle([0, H - 5, int(W * min(1, (t0 + k + 1) / total)), H], fill=ACCENT)
        yield img


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def encode(frames_dir: Path, out: Path, gif_width: int, gif_fps: float):
    ff = shutil.which("ffmpeg")
    if not ff:
        raise SystemExit("ffmpeg not found on PATH")
    pattern = str(frames_dir / "f%04d.png")
    mp4 = out.with_suffix(".mp4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pattern,
                    "-vf", "fps=25,format=yuv420p", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                    "-movflags", "+faststart", str(mp4)], check=True)
    gif = out.with_suffix(".gif")
    vf = (f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];"
          "[a]palettegen=max_colors=256:stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a:"
          "diff_mode=rectangle")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pattern, "-vf", vf,
                    "-loop", "0", str(gif)], check=True)
    return gif, mp4


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render-dir", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--classes", nargs="+")
    ap.add_argument("--app-captures", help="folder from record_app.py (optional)")
    ap.add_argument("--scenes", type=int, nargs="+", help="frame indices for the randomise montage")
    ap.add_argument("--hero", type=int, help="frame index for the render/annotate scenes")
    ap.add_argument("--grid", type=int, nargs="+", help="9 frame indices for the export grid")
    ap.add_argument("--footer", default="", help="optional last line of the end card, e.g. your repo URL")
    ap.add_argument("--out", default="demo", help="output path without extension")
    ap.add_argument("--gif-width", type=int, default=960)
    ap.add_argument("--gif-fps", type=float, default=12.5)
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args(argv)

    data = Data(args)
    lengths = {"title": 20, "load": 44, "rand": 8 * 8, "render": 44, "annotate": 86, "export": 62, "end": 32}
    app_dir = Path(args.app_captures) if args.app_captures else None
    app_len = 0
    if app_dir and (app_dir / "a1.png").exists():
        n_prog = len([p for p in app_dir.glob("b*.png") if p.stem != "b00"])
        app_len = 12 + 18 + 9 + 7 + min(16, n_prog) + 26
    total = sum(lengths.values()) + app_len

    frames_dir = Path(tempfile.mkdtemp(prefix="cad2coco_gif_"))
    t = 0
    order = [("title", lambda t0: scene_title(lengths["title"], t0, total)),
             ("load", lambda t0: scene_load(data, lengths["load"], t0, total)),
             ("rand", lambda t0: scene_randomise(data, lengths["rand"], t0, total)),
             ("render", lambda t0: scene_render(data, lengths["render"], t0, total)),
             ("annotate", lambda t0: scene_annotate(data, lengths["annotate"], t0, total)),
             ("export", lambda t0: scene_export(data, lengths["export"], t0, total))]
    if app_len:
        order.append(("app", lambda t0: scene_app(app_dir, t0, total)))
    order.append(("end", lambda t0: scene_end(lengths["end"], t0, total, args.footer)))
    for name, make in order:
        count = 0
        for frame in make(t):
            frame.save(frames_dir / f"f{t:04d}.png")
            t += 1
            count += 1
        print(f"{name:9s} {count:3d} frames")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gif, mp4 = encode(frames_dir, out, args.gif_width, args.gif_fps)
    print(f"{t} frames, {t / FPS:.1f}s")
    for f in (gif, mp4):
        print(f"{f}  {f.stat().st_size / 1e6:.1f} MB")
    if args.keep_frames:
        print("frames kept in", frames_dir)
    else:
        shutil.rmtree(frames_dir)


if __name__ == "__main__":
    main()
