import blenderproc as bproc  # noqa: I001  (BlenderProc requires this to be the first import)

"""
CAD2COCO render job.

Runs *inside* Blender's Python via:

    blenderproc run cad2coco/blender/render_scene.py path/to/job.json

For every accepted frame it writes:
    images/<id>.jpg     RGB render
    masks/<id>.npz      uint16 instance map (0 = background, n = target instance n)
    frames.jsonl        one line per frame: instance -> category mapping, camera K and pose

The COCO conversion happens outside Blender (cad2coco/coco.py) so it can be unit-tested
without a Blender install.
"""

import argparse
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix

try:
    import cv2
except ImportError:  # BlenderProc normally ships opencv-contrib-python, but stay robust
    cv2 = None


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def log(msg):
    print(f"[cad2coco] {msg}", flush=True)


def principled(mat_bpy):
    for node in mat_bpy.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            return node
    raise RuntimeError(f"Material {mat_bpy.name} has no Principled BSDF node")


def set_input(node, name, value):
    """Set a Principled BSDF input if it exists (names differ slightly across Blender versions)."""
    if name in node.inputs:
        node.inputs[name].default_value = value


def normalize_mesh(obj):
    """Centre the mesh on its bounding box and scale its longest side to 1 m.

    CAD exports come in mm, cm, m or inches. Normalising at load time makes the unit
    irrelevant; the physical size in the scene is then controlled by `object_size`.
    """
    obj.persist_transformation_into_mesh()
    mesh = obj.get_mesh()
    n = len(mesh.vertices)
    if n == 0:
        raise RuntimeError(f"{obj.get_name()} has no vertices")
    co = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lo, hi = co.min(axis=0), co.max(axis=0)
    extent = float((hi - lo).max())
    if extent <= 0:
        raise RuntimeError(f"{obj.get_name()} has a degenerate bounding box")
    centre = (lo + hi) / 2.0
    s = 1.0 / extent
    mesh.transform(Matrix.Diagonal((s, s, s, 1.0)) @ Matrix.Translation(-centre))
    mesh.update()
    return extent


def load_template(path, category_id, name):
    parts = bproc.loader.load_obj(path)
    parts = [p for p in parts if isinstance(p, bproc.types.MeshObject)]
    if not parts:
        raise RuntimeError(f"No mesh found in {path}")
    # Detach from any importer-created parents (glTF/FBX) while keeping world transforms
    for p in parts:
        bo = p.blender_obj
        if bo.parent is not None:
            mw = bo.matrix_world.copy()
            bo.parent = None
            bo.matrix_world = mw
    obj = parts[0]
    if len(parts) > 1:
        obj.join_with_other_objects(parts[1:])
    obj.set_name(f"template_{name}")
    raw_extent = normalize_mesh(obj)

    # A mesh without material slots would silently ignore material randomisation
    if len(obj.blender_obj.data.materials) == 0:
        default = bproc.material.create(f"default_{name}")
        set_input(principled(default.blender_obj), "Base Color", (0.6, 0.6, 0.6, 1.0))
        obj.blender_obj.data.materials.append(default.blender_obj)

    obj.set_cp("category_id", category_id)
    obj.hide(True)
    obj.set_location([0, 0, -100])
    log(f"loaded {os.path.basename(path)} as '{name}' (raw extent {raw_extent:.4g} units, "
        f"{len(obj.get_mesh().vertices)} vertices)")
    return obj


def instance_copy(template):
    """Per-frame copy with its own mesh data.

    Physics simulation applies scale into the mesh, which Blender refuses on shared
    (multi-user) mesh data, so every instance owns its mesh and frees it afterwards.
    """
    bo = template.blender_obj.copy()
    bo.data = template.blender_obj.data.copy()
    bpy.context.scene.collection.objects.link(bo)
    copy = bproc.types.MeshObject(bo)
    copy.hide(False)
    return copy


def remove_objects(objs):
    for o in objs:
        mesh = o.blender_obj.data
        bpy.data.objects.remove(o.blender_obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


# --------------------------------------------------------------------------------------
# randomisers
# --------------------------------------------------------------------------------------

FINISHES = ("metal", "painted", "plastic", "rubber")


def randomize_finish(mat_bpy, rng):
    node = principled(mat_bpy)
    finish = FINISHES[rng.integers(len(FINISHES))]
    if finish == "metal":
        g = rng.uniform(0.35, 0.9)
        tint = rng.uniform(-0.05, 0.05, 3)
        color = np.clip(g + tint, 0, 1)
        metallic, rough = rng.uniform(0.75, 1.0), rng.uniform(0.12, 0.55)
    elif finish == "painted":
        hsv_h = rng.uniform(0, 1)
        color = np.array(hsv_to_rgb(hsv_h, rng.uniform(0.4, 0.95), rng.uniform(0.25, 0.95)))
        metallic, rough = rng.uniform(0.0, 0.2), rng.uniform(0.25, 0.7)
    elif finish == "plastic":
        color = rng.uniform(0.02, 0.95, 3)
        metallic, rough = 0.0, rng.uniform(0.3, 0.8)
    else:  # rubber
        color = np.full(3, rng.uniform(0.02, 0.15))
        metallic, rough = 0.0, rng.uniform(0.7, 1.0)
    set_input(node, "Base Color", (*color, 1.0))
    set_input(node, "Metallic", float(metallic))
    set_input(node, "Roughness", float(rough))
    return finish


def hsv_to_rgb(h, s, v):
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def make_floor_material():
    mat = bproc.material.create("floor_mat")
    nt = mat.blender_obj.node_tree
    bsdf = principled(mat.blender_obj)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    # Keep names, not node references: the physics step runs inside an undo block,
    # which invalidates raw bpy references (BlenderProc only refreshes its own wrappers).
    return mat, noise.name, ramp.name


def randomize_floor(floor_parts, rng):
    mat, noise_name, ramp_name = floor_parts
    nodes = mat.blender_obj.node_tree.nodes
    noise, ramp, bsdf = nodes[noise_name], nodes[ramp_name], principled(mat.blender_obj)
    base = rng.uniform(0.05, 0.85, 3)
    other = np.clip(base + rng.uniform(-0.25, 0.25, 3), 0, 1)
    ramp.color_ramp.elements[0].color = (*base, 1.0)
    ramp.color_ramp.elements[1].color = (*other, 1.0)
    noise.inputs["Scale"].default_value = float(rng.uniform(1.0, 60.0))
    noise.inputs["Detail"].default_value = float(rng.uniform(0.0, 12.0))
    set_input(bsdf, "Roughness", float(rng.uniform(0.2, 1.0)))
    set_input(bsdf, "Metallic", float(rng.uniform(0.0, 0.4)))


def random_rotation(rng):
    return rng.uniform(0, 2 * math.pi, 3)


def sample_positions(k, radius, min_dist, rng, tries=200):
    pts = []
    for _ in range(tries):
        if len(pts) == k:
            break
        r = radius * math.sqrt(rng.uniform())
        a = rng.uniform(0, 2 * math.pi)
        p = np.array([r * math.cos(a), r * math.sin(a)])
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
    while len(pts) < k:  # fall back to overlapping placement rather than fewer objects
        pts.append(rng.uniform(-radius, radius, 2))
    return pts


def resting_z(obj):
    """z offset so the object's lowest point touches the floor."""
    bb = obj.get_bound_box()
    return -float(bb[:, 2].min())


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", help="job.json written by cad2coco.pipeline")
    args = ap.parse_args()
    with open(args.job, encoding="utf-8") as f:
        job = json.load(f)

    ds, sc, cam, look = job["dataset"], job["scene"], job["camera"], job["appearance"]
    out = job["render_dir"]
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "masks"), exist_ok=True)
    rng = np.random.default_rng(int(ds["seed"]))

    bproc.init()

    # ---- models ------------------------------------------------------------------
    templates = []
    for m in job["models"]:
        try:
            templates.append((load_template(m["path"], int(m["category_id"]), m["name"]), int(m["category_id"])))
        except Exception as exc:  # report and keep going with the other models
            log(f"ERROR loading {m['path']}: {exc}")
    if not templates:
        log("ERROR no model could be loaded - nothing to render")
        sys.exit(2)

    # ---- static scene ------------------------------------------------------------
    floor = bproc.object.create_primitive("PLANE")
    floor.set_name("floor")
    floor.set_scale([30, 30, 1])
    floor_parts = make_floor_material()
    floor.replace_materials(floor_parts[0])

    distractor_pool = []
    shapes = ["CUBE", "CYLINDER", "CONE", "SPHERE"]
    for i in range(int(sc["distractors_max"])):
        d = bproc.object.create_primitive(shapes[i % len(shapes)])
        d.set_name(f"distractor_{i}")
        dm = bproc.material.create(f"distractor_mat_{i}")
        d.replace_materials(dm)
        d.hide(True)
        distractor_pool.append((d, dm))

    max_inst = int(sc["instances_max"])
    inst_materials = [bproc.material.create(f"instance_mat_{i}") for i in range(max_inst)]

    lights = [bproc.types.Light(light_type="POINT", name=f"light_{i}") for i in range(int(look["lights_max"]))]

    bproc.camera.set_resolution(int(ds["width"]), int(ds["height"]))
    bproc.renderer.set_max_amount_of_samples(int(look["samples"]))
    bproc.renderer.set_noise_threshold(0.05)
    bproc.renderer.enable_segmentation_output(map_by=["instance"], default_values={"category_id": 0})

    size = float(sc["object_size"])
    n_target = int(ds["num_images"])
    max_attempts = n_target * 4 + 10
    saved, attempts = 0, 0
    frames_path = os.path.join(out, "frames.jsonl")
    open(frames_path, "w").close()
    t0 = time.time()
    physics = sc["placement"] == "physics"

    while saved < n_target and attempts < max_attempts:
        attempts += 1
        bproc.utility.reset_keyframes()

        # ---- targets -------------------------------------------------------------
        k = int(rng.integers(int(sc["instances_min"]), max_inst + 1))
        spread = float(sc["spread_radius"])
        positions = sample_positions(k, spread, 0.8 * size, rng)
        instances = []  # (obj, category_id, pass_index)
        for j in range(k):
            tpl, cid = templates[int(rng.integers(len(templates)))]
            obj = instance_copy(tpl)
            s = size * float(rng.uniform(1 - sc["size_jitter"], 1 + sc["size_jitter"]))
            obj.set_scale([s, s, s])
            obj.set_rotation_euler(random_rotation(rng))
            x, y = positions[j]
            if physics:
                z = size * (1.0 + 1.2 * j)  # stagger drop heights
            else:
                z = 0.0
            obj.set_location([x, y, z])
            if not physics:
                bpy.context.view_layer.update()
                obj.set_location([x, y, resting_z(obj) + float(rng.uniform(0, 0.4)) * size])

            use_original = (look["material_mode"] == "original" or
                            (look["material_mode"] == "mixed" and rng.uniform() < 0.5))
            bo = obj.blender_obj
            if not use_original:
                mat = inst_materials[j]
                randomize_finish(mat.blender_obj, rng)
                for slot in bo.material_slots:
                    slot.link = "OBJECT"
                    slot.material = mat.blender_obj
            instances.append((obj, cid, j + 1))

        # ---- distractors ---------------------------------------------------------
        n_dis = int(rng.integers(0, len(distractor_pool) + 1)) if distractor_pool else 0
        active_dis = []
        for idx, (d, dm) in enumerate(distractor_pool):
            if idx < n_dis:
                d.hide(False)
                ds_ = size * float(rng.uniform(0.15, 0.6))
                d.set_scale([ds_ * rng.uniform(0.5, 1.5), ds_ * rng.uniform(0.5, 1.5), ds_ * rng.uniform(0.5, 1.5)])
                d.set_rotation_euler([0, 0, rng.uniform(0, 2 * math.pi)])
                # some land among the targets (occlusion), some around them (clutter)
                r = rng.uniform(0.2, 1.4) * (spread + size)
                a = rng.uniform(0, 2 * math.pi)
                d.set_location([r * math.cos(a), r * math.sin(a), 0])
                bpy.context.view_layer.update()
                d.set_location([r * math.cos(a), r * math.sin(a), resting_z(d)])
                randomize_finish(dm.blender_obj, rng)
                active_dis.append(d)
            else:
                d.hide(True)
                d.set_location([0, 0, -100])

        # ---- floor / background ----------------------------------------------------
        use_floor = physics or rng.uniform() < float(sc["floor_probability"])
        floor.hide(not use_floor)
        if use_floor:
            randomize_floor(floor_parts, rng)

        # ---- physics ---------------------------------------------------------------
        if physics:
            floor.enable_rigidbody(active=False, collision_shape="BOX")
            for d in active_dis:
                d.enable_rigidbody(active=False, collision_shape="CONVEX_HULL")
            for obj, _, _ in instances:
                obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL", friction=0.8)
            bproc.object.simulate_physics_and_fix_final_poses(
                min_simulation_time=1.0, max_simulation_time=4.0, check_object_interval=0.5)
            bproc.utility.reset_keyframes()

        # ---- lighting --------------------------------------------------------------
        centre = np.mean([o.get_location() for o, _, _ in instances], axis=0)
        bg = rng.uniform(0.05, 0.9, 3)
        bproc.renderer.set_world_background(list(map(float, bg)), strength=float(rng.uniform(0.15, 0.9)))
        n_lights = int(rng.integers(1, len(lights) + 1))
        for i, light in enumerate(lights):
            if i >= n_lights:
                light.set_energy(0)
                continue
            dist = rng.uniform(1.0, 3.0) * max(size, 0.05) * 4
            el = rng.uniform(math.radians(20), math.radians(85))
            az = rng.uniform(0, 2 * math.pi)
            light.set_location(centre + dist * np.array([math.cos(el) * math.cos(az),
                                                         math.cos(el) * math.sin(az), math.sin(el)]))
            light.set_energy(float(rng.uniform(look["light_min"], look["light_max"]) * dist ** 2))
            warm, cool = np.array([1.0, 0.82, 0.62]), np.array([0.78, 0.88, 1.0])
            t = rng.uniform()
            light.set_color(list(map(float, warm * (1 - t) + cool * t)))

        # ---- camera ----------------------------------------------------------------
        fov = math.radians(rng.uniform(cam["fov_min"], cam["fov_max"]))
        bproc.camera.set_intrinsics_from_blender_params(lens=fov, lens_unit="FOV", clip_start=0.005, clip_end=200)
        target = centre + rng.normal(0, 0.15 * size, 3)
        d = rng.uniform(cam["dist_min"], cam["dist_max"])
        el = math.radians(rng.uniform(cam["elev_min"], cam["elev_max"]))
        az = rng.uniform(0, 2 * math.pi)
        location = target + d * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        roll = math.radians(rng.uniform(-cam["roll_max"], cam["roll_max"]))
        rot = bproc.camera.rotation_from_forward_vec(target - location, inplane_rot=roll)
        cam2world = bproc.math.build_transformation_mat(location, rot)
        bproc.camera.add_camera_pose(cam2world)

        # ---- segmentation ids: only targets get a non-zero pass index --------------
        for bo in bpy.data.objects:
            if bo.type == "MESH":
                bo.pass_index = 0
        for obj, _, pid in instances:
            obj.blender_obj.pass_index = pid

        # ---- render ----------------------------------------------------------------
        data = bproc.renderer.render()
        rgb = np.asarray(data["colors"][0])[..., :3].astype(np.uint8)
        inst_map = np.asarray(data["instance_segmaps"][0]).astype(np.uint16)

        min_px = int(job["annotation"]["min_visible_px"])
        kept = []
        for obj, cid, pid in instances:
            px = int((inst_map == pid).sum())
            if px >= min_px:
                kept.append({"id": pid, "category_id": cid, "visible_px": px,
                             "object_to_world": obj.get_local2world_mat().round(6).tolist()})
            else:
                inst_map[inst_map == pid] = 0  # too small / hidden: treat as background

        if kept:
            rgb = augment(rgb, look, rng)
            stem = f"{saved:06d}"
            img_rel, mask_rel = f"images/{stem}.jpg", f"masks/{stem}.npz"
            save_jpg(os.path.join(out, img_rel), rgb, int(rng.integers(80, 97)))
            np.savez_compressed(os.path.join(out, mask_rel), instances=inst_map)
            record = {
                "image": img_rel, "mask": mask_rel,
                "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                "instances": kept,
                "camera_K": np.asarray(bproc.camera.get_intrinsics_as_K_matrix()).round(6).tolist(),
                "camera_to_world": np.asarray(cam2world).round(6).tolist(),
            }
            with open(frames_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            saved += 1
            print(f"PROGRESS {saved}/{n_target}", flush=True)
        else:
            log(f"attempt {attempts}: no instance visible enough, retrying")

        # ---- clean up per-frame objects --------------------------------------------
        remove_objects([o for o, _, _ in instances])
        if physics:
            for d in active_dis:
                if d.has_rigidbody_enabled():
                    d.disable_rigidbody()
            if floor.has_rigidbody_enabled():
                floor.disable_rigidbody()

    log(f"done: {saved} images in {attempts} attempts, {time.time() - t0:.1f}s")
    if saved == 0:
        log("ERROR zero usable images - check that the camera distance range matches object_size")
        sys.exit(3)


def augment(rgb, look, rng):
    img = rgb.astype(np.float32)
    sigma = rng.uniform(0, float(look["noise_max"]))
    if sigma > 0:
        img += rng.normal(0, sigma, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if cv2 is not None and rng.uniform() < float(look["blur_prob"]):
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


def save_jpg(path, rgb, quality):
    if cv2 is not None:
        cv2.imwrite(path, rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        from PIL import Image
        Image.fromarray(rgb).save(path, quality=quality)


main()
