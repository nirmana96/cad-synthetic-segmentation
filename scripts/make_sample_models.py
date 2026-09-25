"""Generate the small sample CAD models in examples/models (no CAD licence issues).

    python scripts/make_sample_models.py
"""

from pathlib import Path

import numpy as np
import trimesh
from trimesh.creation import annulus, box, cylinder

OUT = Path(__file__).resolve().parents[1] / "examples" / "models"


def l_bracket() -> trimesh.Trimesh:
    """80 x 60 x 50 mm angle bracket with two mounting holes' bosses (units: mm)."""
    base = box(extents=[80, 60, 5])
    wall = box(extents=[80, 5, 50])
    wall.apply_translation([0, -27.5, 27.5])
    rib = box(extents=[5, 30, 30])
    rib.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [1, 0, 0]))
    rib.apply_translation([0, -15, 12])
    return trimesh.util.concatenate([base, wall, rib])


def pipe_flange() -> trimesh.Trimesh:
    """DN50-style slip-on flange: ring plate + hub + 4 bolt bosses (units: mm)."""
    plate = annulus(r_min=30, r_max=82, height=16, sections=96)
    hub = annulus(r_min=30, r_max=42, height=30, sections=96)
    hub.apply_translation([0, 0, 15])
    bosses = []
    for a in np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4:
        b = cylinder(radius=8, height=18, sections=32)
        b.apply_translation([62 * np.cos(a), 62 * np.sin(a), 0])
        bosses.append(b)
    return trimesh.util.concatenate([plate, hub, *bosses])


def pipe_tee() -> trimesh.Trimesh:
    """Equal tee: run + branch tubes with end collars (units: mm)."""
    run = annulus(r_min=18, r_max=24, height=120, sections=64)
    run.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    branch = annulus(r_min=18, r_max=24, height=60, sections=64)
    branch.apply_translation([0, 0, 30])
    parts = [run, branch]
    for pos, axis in (([-58, 0, 0], [0, 1, 0]), ([58, 0, 0], [0, 1, 0]), ([0, 0, 58], None)):
        collar = annulus(r_min=18, r_max=28, height=8, sections=64)
        if axis:
            collar.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, axis))
        collar.apply_translation(pos)
        parts.append(collar)
    return trimesh.util.concatenate(parts)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples.mtl").write_text("newmtl steel\nKa 0.2 0.2 0.2\nKd 0.62 0.64 0.67\nKs 0.5 0.5 0.5\nNs 60\n")
    for name, mesh in (("l_bracket", l_bracket()), ("pipe_flange", pipe_flange()), ("pipe_tee", pipe_tee())):
        path = OUT / f"{name}.obj"
        mesh.export(path, include_normals=True)
        # point every sample at one shared steel material (shows .mtl handling in the app)
        body = path.read_text()
        path.write_text(f"mtllib samples.mtl\nusemtl steel\n{body}")
        print(f"{path.name}: {len(mesh.faces)} faces, extents {np.round(mesh.extents, 1)} mm")
