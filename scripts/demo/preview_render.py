"""Lightweight CPU preview renderer, used only to build the demo GIF without Blender.

This is NOT the dataset renderer (that is cad2coco/blender/render_scene.py, BlenderProc +
Cycles). It is a small z-buffer rasteriser with Blinn-Phong shading, projected soft shadows
and the same randomisation ideas, and it writes the exact same output files
(images/, masks/, frames.jsonl). That means the real COCO conversion in cad2coco.coco
can run on its output unchanged.

    pip install numba trimesh
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numba
import numpy as np
import trimesh

# --------------------------------------------------------------------------------------
# rasteriser
# --------------------------------------------------------------------------------------


@numba.njit(cache=True)
def _raster(px, py, iz, attr, tri_obj, W, H):
    """Perspective-correct z-buffer rasterisation of screen-space triangles.

    px, py, iz: (T, 3) pixel x, pixel y and 1/depth per corner
    attr:       (T, 3, A) per-corner attributes (interpolated perspective-correct)
    tri_obj:    (T,) object id per triangle (>0)
    """
    T = px.shape[0]
    A = attr.shape[2]
    invz = np.zeros((H, W))
    ids = np.zeros((H, W), np.int32)
    out = np.zeros((H, W, A))
    for t in range(T):
        x0, x1, x2 = px[t, 0], px[t, 1], px[t, 2]
        y0, y1, y2 = py[t, 0], py[t, 1], py[t, 2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        minx = max(int(math.floor(min(x0, min(x1, x2)))), 0)
        maxx = min(int(math.ceil(max(x0, max(x1, x2)))), W - 1)
        miny = max(int(math.floor(min(y0, min(y1, y2)))), 0)
        maxy = min(int(math.ceil(max(y0, max(y1, y2)))), H - 1)
        if minx > maxx or miny > maxy:
            continue
        z0, z1, z2 = iz[t, 0], iz[t, 1], iz[t, 2]
        for y in range(miny, maxy + 1):
            fy = y + 0.5
            for x in range(minx, maxx + 1):
                fx = x + 0.5
                w0 = ((x1 - fx) * (y2 - fy) - (x2 - fx) * (y1 - fy)) / area
                w1 = ((x2 - fx) * (y0 - fy) - (x0 - fx) * (y2 - fy)) / area
                w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z > invz[y, x]:
                    invz[y, x] = z
                    ids[y, x] = tri_obj[t]
                    for a in range(A):
                        out[y, x, a] = (w0 * attr[t, 0, a] * z0 + w1 * attr[t, 1, a] * z1
                                        + w2 * attr[t, 2, a] * z2) / z
    return invz, ids, out


# --------------------------------------------------------------------------------------
# scene description
# --------------------------------------------------------------------------------------


@dataclass
class Item:
    V: np.ndarray  # (n, 3) world vertices
    F: np.ndarray  # (m, 3) faces
    N: np.ndarray  # (n, 3) world vertex normals
    base: np.ndarray  # linear RGB
    metallic: float
    rough: float
    inst: int = 0  # 0 = distractor (not annotated), 1..k = target instance
    category_id: int = 0
    finish: str = ""


@dataclass
class Light:
    pos: np.ndarray
    color: np.ndarray
    irradiance: float  # at the scene centre


@dataclass
class Camera:
    loc: np.ndarray
    target: np.ndarray
    fov: float  # radians
    roll: float = 0.0  # radians

    def basis(self):
        f = self.target - self.loc
        f = f / np.linalg.norm(f)
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up)
        if np.linalg.norm(r) < 1e-6:
            r = np.array([1.0, 0.0, 0.0])
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)
        c, s = math.cos(self.roll), math.sin(self.roll)
        return f, c * r + s * u, -s * r + c * u


@dataclass
class Scene:
    items: list[Item]
    lights: list[Light]
    camera: Camera
    world_color: np.ndarray
    world_strength: float
    floor: dict | None = None  # {"tex": HxWx3 linear, "extent": m, "rough": float}
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------

SHADOW_EXTENT = 1.6  # half-size of the floor shadow grid, metres
SHADOW_RES = 800


def _floor_shadow(items, light_pos):
    """Soft shadow of all items on the z=0 plane from a point light, as a floor-space grid."""
    grid = np.zeros((SHADOW_RES, SHADOW_RES), np.uint8)
    polys = []
    for it in items:
        P = it.V
        denom = light_pos[2] - P[:, 2]
        ok = denom > 1e-4
        k = np.where(ok, light_pos[2] / np.maximum(denom, 1e-4), 0.0)
        Pf = light_pos[None, :] + (P - light_pos[None, :]) * k[:, None]
        g = (Pf[:, :2] + SHADOW_EXTENT) / (2 * SHADOW_EXTENT) * SHADOW_RES
        tris = g[it.F]
        valid = ok[it.F].all(axis=1) & np.isfinite(tris).all(axis=(1, 2))
        valid &= (np.abs(tris) < 4 * SHADOW_RES).all(axis=(1, 2))
        polys.append(np.round(tris[valid]).astype(np.int32))
    if polys:
        cv2.fillPoly(grid, np.concatenate(polys), 1)
    soft = cv2.GaussianBlur(grid.astype(np.float32), (0, 0), 4.0)
    return soft


def _contact_ao(items):
    grid = np.zeros((SHADOW_RES, SHADOW_RES), np.uint8)
    for it in items:
        g = (it.V[:, :2] + SHADOW_EXTENT) / (2 * SHADOW_EXTENT) * SHADOW_RES
        cv2.fillPoly(grid, np.round(g[it.F]).astype(np.int32), 1)
    return cv2.GaussianBlur(grid.astype(np.float32), (0, 0), 9.0)


def _sample_grid(grid, wx, wy):
    mx = ((wx + SHADOW_EXTENT) / (2 * SHADOW_EXTENT) * SHADOW_RES - 0.5).astype(np.float32)
    my = ((wy + SHADOW_EXTENT) / (2 * SHADOW_EXTENT) * SHADOW_RES - 0.5).astype(np.float32)
    return cv2.remap(grid, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _tonemap(x, exposure=1.5):
    x = 1.0 - np.exp(-np.maximum(x, 0) * exposure)
    return np.clip(x, 0, 1) ** (1 / 2.2)


def render(scene: Scene, W: int, H: int, ss: int = 2, background=None, return_alpha: bool = False):
    """Returns (rgb uint8 HxWx3, instance map uint16 HxW[, object coverage alpha HxW]).

    background: optional linear RGB used for empty pixels instead of the world colour
    (black gives a premultiplied image that can be composited with the alpha).
    """
    Ws, Hs = W * ss, H * ss
    cam = scene.camera
    f, r, u = cam.basis()
    fpx = (Ws / 2) / math.tan(cam.fov / 2)

    # ---- objects -------------------------------------------------------------------------
    PX, PY, IZ, ATTR, OBJ = [], [], [], [], []
    for i, it in enumerate(scene.items):
        rel = it.V - cam.loc
        zc = rel @ f
        xc, yc = rel @ r, rel @ u
        zsafe = np.maximum(zc, 1e-4)
        vx = Ws / 2 + fpx * xc / zsafe
        vy = Hs / 2 - fpx * yc / zsafe
        keep = (zc[it.F] > 1e-3).all(axis=1)
        F = it.F[keep]
        PX.append(vx[F])
        PY.append(vy[F])
        IZ.append(1.0 / zsafe[F])
        ATTR.append(np.concatenate([it.V, it.N], axis=1)[F])
        OBJ.append(np.full(len(F), i + 1, np.int32))
    if PX:
        invz, ids, attr = _raster(np.concatenate(PX), np.concatenate(PY), np.concatenate(IZ),
                                  np.concatenate(ATTR), np.concatenate(OBJ), Ws, Hs)
    else:
        invz, ids, attr = np.zeros((Hs, Ws)), np.zeros((Hs, Ws), np.int32), np.zeros((Hs, Ws, 6))

    # ---- per-pixel rays and floor ------------------------------------------------------------
    xs = (np.arange(Ws) + 0.5 - Ws / 2) / fpx
    ys = -(np.arange(Hs) + 0.5 - Hs / 2) / fpx
    XN, YN = np.meshgrid(xs, ys)
    D = f[None, None, :] + r[None, None, :] * XN[..., None] + u[None, None, :] * YN[..., None]
    wc = np.asarray(scene.world_color) * scene.world_strength
    grad = 0.85 + 0.3 * np.clip(YN / (np.abs(YN).max() + 1e-9), -1, 1)[..., None] * 0.5
    color = wc[None, None, :] * grad
    if background is not None:
        color = np.broadcast_to(np.asarray(background, float), color.shape).copy()

    lights = scene.lights
    if scene.floor is not None:
        dz = D[..., 2]
        hit = dz < -1e-6
        t = np.where(hit, -cam.loc[2] / np.where(hit, dz, -1), np.inf)
        floor_invz = np.where(hit, 1.0 / t, 0.0)
        floor_px = hit & (floor_invz >= invz)
        wx = cam.loc[0] + t * D[..., 0]
        wy = cam.loc[1] + t * D[..., 1]
        wx = np.where(floor_px, wx, 0.0)
        wy = np.where(floor_px, wy, 0.0)
        tex, ext = scene.floor["tex"], scene.floor["extent"]
        th, tw = tex.shape[:2]
        mx = ((wx / ext) % 1.0 * tw).astype(np.float32)
        my = ((wy / ext) % 1.0 * th).astype(np.float32)
        alb = cv2.remap(tex.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        P = np.stack([wx, wy, np.zeros_like(wx)], -1)
        light_sum = np.zeros_like(alb)
        for L in lights:
            Lv = L.pos[None, None, :] - P
            dist = np.linalg.norm(Lv, axis=-1)
            ndl = np.clip(Lv[..., 2] / dist, 0, 1)
            ref = np.linalg.norm(L.pos)
            E = L.irradiance * (ref / dist) ** 2
            sh = _sample_grid(_floor_shadow(scene.items, L.pos), wx, wy)
            light_sum += (E * ndl * (1 - 0.85 * np.clip(sh, 0, 1)))[..., None] * L.color[None, None, :]
        ao = 1 - 0.35 * np.clip(_sample_grid(_contact_ao(scene.items), wx, wy), 0, 1)
        floor_col = alb * (light_sum + (wc * 0.55 + 0.02)[None, None, :]) * ao[..., None]
        fog = np.clip((np.hypot(wx, wy) - 1.1) / 1.4, 0, 1)[..., None]
        floor_col = floor_col * (1 - fog) + color * fog
        color = np.where(floor_px[..., None], floor_col, color)
        obj_px = (ids > 0) & (invz > floor_invz)
        floor_mean = alb[floor_px].mean(axis=0) if floor_px.any() else wc
    else:
        obj_px = ids > 0
        floor_mean = wc * 0.4

    # ---- object shading -----------------------------------------------------------------------
    if obj_px.any():
        sid = ids[obj_px] - 1
        P = attr[obj_px][:, :3]
        N = attr[obj_px][:, 3:]
        N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
        V = cam.loc[None, :] - P
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        flip = (N * V).sum(1) < 0
        N[flip] *= -1
        base = np.array([it.base for it in scene.items])[sid]
        metal = np.array([it.metallic for it in scene.items])[sid][:, None]
        rough = np.array([it.rough for it in scene.items])[sid][:, None]
        diff_alb = base * (1 - metal)
        spec_col = 0.04 * (1 - metal) + base * metal
        shin = 2.0 / np.maximum(rough ** 2, 0.01) + 2
        acc_d = np.zeros_like(base)
        acc_s = np.zeros_like(base)
        for L in lights:
            Lv = L.pos[None, :] - P
            dist = np.linalg.norm(Lv, axis=1, keepdims=True)
            Ld = Lv / dist
            ndl = np.clip((N * Ld).sum(1, keepdims=True), 0, 1)
            Hh = Ld + V
            Hh /= np.linalg.norm(Hh, axis=1, keepdims=True) + 1e-9
            ndh = np.clip((N * Hh).sum(1, keepdims=True), 0, 1)
            E = L.irradiance * (np.linalg.norm(L.pos) / dist) ** 2
            acc_d += E * ndl * L.color[None, :]
            acc_s += E * ndl * (ndh ** shin) * (shin + 8) / (8 * math.pi) * L.color[None, :]
        R = 2 * (N * V).sum(1, keepdims=True) * N - V
        up = np.clip(R[:, 2:3] * 0.5 + 0.5, 0, 1)
        env = (wc * 1.2 + 0.08)[None, :] * up + (floor_mean * 0.35)[None, :] * (1 - up)
        amb = wc * 0.45 + 0.03
        shade = diff_alb * (acc_d + amb[None, :]) + spec_col * acc_s + spec_col * env * (1 - 0.6 * rough)
        color[obj_px] = shade

    rgb = (_tonemap(color) * 255).astype(np.float32)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    inst_lut = np.array([0] + [it.inst for it in scene.items], np.uint16)
    inst_full = np.where(obj_px, inst_lut[ids], 0).astype(np.uint16)
    inst = cv2.resize(inst_full, (W, H), interpolation=cv2.INTER_NEAREST)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if return_alpha:
        alpha = cv2.resize(obj_px.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        return rgb, inst, alpha
    return rgb, inst


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------


def load_part(path) -> trimesh.Trimesh:
    """Load and normalise like render_scene.py: centre on bbox, longest side = 1."""
    mesh = trimesh.load(path, force="mesh", process=True)
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2)
    mesh.apply_scale(1.0 / (hi - lo).max())
    return mesh


def _shaded(mesh: trimesh.Trimesh):
    sm = mesh.smooth_shaded
    return np.asarray(sm.vertices, float), np.asarray(sm.faces), np.asarray(sm.vertex_normals, float)


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return trimesh.transformations.quaternion_matrix(q)[:3, :3]


def hsv_to_rgb(h, s, v):
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return np.array([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i])


def random_finish(rng):
    finish = ("metal", "painted", "plastic", "rubber")[int(rng.integers(4))]
    if finish == "metal":
        color = np.clip(rng.uniform(0.35, 0.9) + rng.uniform(-0.05, 0.05, 3), 0, 1)
        return finish, color, rng.uniform(0.75, 1.0), rng.uniform(0.12, 0.55)
    if finish == "painted":
        color = hsv_to_rgb(rng.uniform(), rng.uniform(0.4, 0.95), rng.uniform(0.25, 0.95)) ** 2.2
        return finish, color, rng.uniform(0, 0.2), rng.uniform(0.25, 0.7)
    if finish == "plastic":
        return finish, rng.uniform(0.02, 0.95, 3) ** 2.2, 0.0, rng.uniform(0.3, 0.8)
    return finish, np.full(3, rng.uniform(0.02, 0.15)), 0.0, rng.uniform(0.7, 1.0)


def floor_texture(rng, res=512):
    tex = np.zeros((res, res), np.float32)
    for s, w in ((4, 0.5), (16, 0.3), (64, 0.2)):
        small = rng.random((s, s)).astype(np.float32)
        small = np.concatenate([small, small[:1]], 0)
        small = np.concatenate([small, small[:, :1]], 1)
        tex += w * cv2.resize(small, (res + res // s, res + res // s), interpolation=cv2.INTER_CUBIC)[:res, :res]
    tex = (tex - tex.min()) / (np.ptp(tex) + 1e-6)
    kind = "noise"
    if rng.uniform() < 0.35:  # workshop tiles
        kind = "tiles"
        n = int(rng.integers(2, 7))
        lines = np.zeros_like(tex)
        step = res // n
        lines[::step, :] = 1
        lines[:, ::step] = 1
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
        tex = np.clip(tex * 0.6 + 0.2 - 0.5 * lines, 0, 1)
    base = rng.uniform(0.05, 0.85, 3)
    other = np.clip(base + rng.uniform(-0.25, 0.25, 3), 0, 1)
    rgb = (base[None, None, :] * (1 - tex[..., None]) + other[None, None, :] * tex[..., None]) ** 2.2
    return {"tex": rgb.astype(np.float32), "extent": float(rng.uniform(0.3, 2.0)), "kind": kind}


PRIMS = {
    "cube": lambda: trimesh.creation.box(extents=[1, 1, 1]),
    "cylinder": lambda: trimesh.creation.cylinder(radius=0.5, height=1, sections=32),
    "cone": lambda: trimesh.creation.cone(radius=0.5, height=1, sections=32),
    "sphere": lambda: trimesh.creation.icosphere(subdivisions=3, radius=0.5),
}


def place(mesh_vfn, R, scale, xy, lift=0.0):
    V, F, N = mesh_vfn
    Vw = (V * scale) @ R.T
    Nw = N @ R.T if np.isscalar(scale) else (N / np.asarray(scale)) @ R.T
    Vw[:, 2] -= Vw[:, 2].min() - lift
    Vw[:, :2] += xy
    return Vw, F, Nw


def sample_positions(k, radius, min_dist, rng, tries=200):
    pts = []
    for _ in range(tries):
        if len(pts) == k:
            break
        rr = radius * math.sqrt(rng.uniform())
        a = rng.uniform(0, 2 * math.pi)
        p = np.array([rr * math.cos(a), rr * math.sin(a)])
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    while len(pts) < k:
        pts.append(rng.uniform(-radius, radius, 2))
    return pts


# --------------------------------------------------------------------------------------
# random scenes (mirrors the randomisation in render_scene.py)
# --------------------------------------------------------------------------------------


def random_scene(parts, rng, size=0.2, spread=0.25, inst_range=(1, 4), max_distractors=4,
                 dist=(0.45, 1.0), elev=(15, 80), fov=(45, 65), roll=15, floor_prob=0.85):
    """parts: list of (shaded (V, F, N), category_id, class name)."""
    k = int(rng.integers(inst_range[0], inst_range[1] + 1))
    items = []
    for j, xy in enumerate(sample_positions(k, spread, 0.9 * size, rng)):
        vfn, cid, _ = parts[int(rng.integers(len(parts)))]
        s = size * rng.uniform(0.85, 1.15)
        V, F, N = place(vfn, random_rotation(rng), s, xy)
        finish, base, metal, rough = random_finish(rng)
        items.append(Item(V, F, N, base, metal, rough, inst=j + 1, category_id=cid, finish=finish))
    n_dis = int(rng.integers(0, max_distractors + 1))
    for _ in range(n_dis):
        name = list(PRIMS)[int(rng.integers(len(PRIMS)))]
        vfn = _shaded(PRIMS[name]())
        dims = size * rng.uniform(0.15, 0.6) * rng.uniform(0.5, 1.5, 3)
        a = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(0.2, 1.4) * (spread + size)
        Rz = trimesh.transformations.rotation_matrix(rng.uniform(0, 2 * math.pi), [0, 0, 1])[:3, :3]
        V, F, N = place(vfn, Rz, dims, np.array([rr * math.cos(a), rr * math.sin(a)]))
        _, base, metal, rough = random_finish(rng)
        items.append(Item(V, F, N, base, metal, rough, inst=0))

    targets = [it for it in items if it.inst > 0]
    centre = np.mean([it.V.mean(0) for it in targets], axis=0)
    n_l = int(rng.integers(1, 4))
    lights = []
    for _ in range(n_l):
        d = rng.uniform(1.0, 3.0) * size * 4
        el, az = rng.uniform(math.radians(25), math.radians(85)), rng.uniform(0, 2 * math.pi)
        pos = centre + d * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        t = rng.uniform()
        col = np.array([1.0, 0.82, 0.62]) * (1 - t) + np.array([0.78, 0.88, 1.0]) * t
        lights.append(Light(pos, col, rng.uniform(0.9, 2.2) / math.sqrt(n_l)))

    world = rng.uniform(0.05, 0.9, 3)
    strength = rng.uniform(0.15, 0.9)
    floor = floor_texture(rng) if rng.uniform() < floor_prob else None
    for _ in range(50):
        fv = math.radians(rng.uniform(*fov))
        dd = rng.uniform(*dist)
        e, a = math.radians(rng.uniform(*elev)), rng.uniform(0, 2 * math.pi)
        target = centre + rng.normal(0, 0.15 * size, 3)
        loc = target + dd * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        cam = Camera(loc, target, fv, math.radians(rng.uniform(-roll, roll)))
        f, _, _ = cam.basis()
        if all(((it.V - loc) @ f).min() > 0.03 for it in items):
            break
    meta = {
        "parts": k, "distractors": n_dis, "lights": n_l,
        "finishes": [it.finish for it in targets],
        "camera": {"distance": dd, "elevation": math.degrees(e), "fov": math.degrees(fv),
                   "roll": math.degrees(cam.roll)},
        "floor": floor["kind"] if floor else "none",
    }
    return Scene(items, lights, cam, world, strength, floor, meta)


def write_dataset_frames(parts, out_dir, n, rng, W=640, H=640, min_visible_px=150, on_frame=None, **scene_kw):
    """Render n frames in render_scene.py's output format. Returns (frames, scenes)."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    records, scenes, saved = [], [], 0
    while saved < n:
        sc = random_scene(parts, rng, **scene_kw)
        rgb, inst = render(sc, W, H)
        kept = []
        for it in sc.items:
            if it.inst == 0:
                continue
            px = int((inst == it.inst).sum())
            if px >= min_visible_px:
                kept.append({"id": it.inst, "category_id": it.category_id, "visible_px": px})
            else:
                inst[inst == it.inst] = 0
        if not kept:
            continue
        stem = f"{saved:06d}"
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        np.savez_compressed(out / "masks" / f"{stem}.npz", instances=inst)
        rec = {"image": f"images/{stem}.jpg", "mask": f"masks/{stem}.npz", "width": W, "height": H,
               "instances": kept}
        records.append(rec)
        scenes.append(sc.meta)
        saved += 1
        if on_frame is not None:
            on_frame(saved, n)
    (out / "frames.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (out / "scene_meta.json").write_text(json.dumps(scenes, indent=1))
    return records, scenes


def turntable(vfn, n_frames, W, H, elevation=28.0, base=(0.55, 0.57, 0.6), bg=(0.09, 0.11, 0.15)):
    """Studio turntable of one normalised part: list of (premultiplied RGB, alpha) frames."""
    V, F, N = vfn
    R = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])[:3, :3]
    Vw, F, Nw = place((V, F, N), R, 1.0, np.zeros(2), lift=0.0)
    Vw[:, 2] -= Vw[:, 2].max() / 2
    item = Item(Vw, F, Nw, np.array(base), 0.85, 0.35, inst=1)
    frames = []
    for i in range(n_frames):
        a = 2 * math.pi * i / n_frames + math.radians(30)
        e = math.radians(elevation)
        loc = 2.3 * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        cam = Camera(loc, np.zeros(3), math.radians(38))
        key = Light(loc + np.array([0.8, -0.8, 1.6]), np.array([1.0, 0.95, 0.9]), 1.6)
        rim = Light(-loc * 1.2 + np.array([0, 0, 1.5]), np.array([0.85, 0.9, 1.0]), 1.1)
        sc = Scene([item], [key, rim], cam, np.array(bg), 1.0)
        rgb, _, alpha = render(sc, W, H, background=np.zeros(3), return_alpha=True)
        frames.append((rgb, alpha))
    return frames
