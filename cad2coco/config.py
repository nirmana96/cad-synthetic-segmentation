"""Generation settings: one dataclass tree, loadable from / savable to YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    num_images: int = 200
    width: int = 640
    height: int = 640
    seed: int = 42
    valid_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class SceneConfig:
    instances_min: int = 1
    instances_max: int = 4
    distractors_max: int = 4
    placement: str = "free"  # "free" (random 6-DoF pose) or "physics" (dropped onto the floor)
    object_size: float = 0.2  # longest side of every model after normalisation, metres
    size_jitter: float = 0.15
    spread_radius: float = 0.25  # objects are placed inside a disc of this radius, metres
    floor_probability: float = 0.85  # otherwise objects float over a plain world colour


@dataclass
class CameraConfig:
    dist_min: float = 0.35
    dist_max: float = 1.1
    elev_min: float = 10.0
    elev_max: float = 85.0
    fov_min: float = 45.0
    fov_max: float = 65.0
    roll_max: float = 15.0


@dataclass
class AppearanceConfig:
    material_mode: str = "random"  # "random" | "original" | "mixed"
    lights_max: int = 3
    light_min: float = 40.0  # point-light power in W per m^2 of light distance
    light_max: float = 400.0
    samples: int = 64
    noise_max: float = 6.0  # std of gaussian sensor noise, 0-255 scale
    blur_prob: float = 0.2


@dataclass
class AnnotationConfig:
    min_visible_px: int = 150  # instances with fewer visible pixels are dropped
    polygon_tolerance: float = 1.0  # Douglas-Peucker epsilon in pixels
    min_part_area: float = 12.0  # polygon parts smaller than this (px^2) are dropped


@dataclass
class GenerationConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)

    # ---- serialisation ------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GenerationConfig:
        data = data or {}
        cfg = cls()
        for f in fields(cls):
            section = getattr(cfg, f.name)
            for key, value in (data.get(f.name) or {}).items():
                if not hasattr(section, key):
                    raise ValueError(f"Unknown config key: {f.name}.{key}")
                current = getattr(section, key)
                setattr(section, key, type(current)(value))
        cfg.validate()
        return cfg

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_yaml(cls, path: str | Path) -> GenerationConfig:
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    # ---- dotted access (used by the UI to map widgets <-> fields) -------------------
    def get(self, dotted: str) -> Any:
        section, key = dotted.split(".")
        return getattr(getattr(self, section), key)

    def set(self, dotted: str, value: Any) -> None:
        section, key = dotted.split(".")
        obj = getattr(self, section)
        setattr(obj, key, type(getattr(obj, key))(value))

    # ---- sanity checks ----------------------------------------------------------------
    def validate(self) -> None:
        d, s, c, a = self.dataset, self.scene, self.camera, self.appearance
        errors = []
        if d.num_images < 1:
            errors.append("num_images must be >= 1")
        if not 0 <= d.valid_ratio + d.test_ratio < 1:
            errors.append("valid_ratio + test_ratio must be below 1")
        if s.instances_min < 1 or s.instances_min > s.instances_max:
            errors.append("need 1 <= instances_min <= instances_max")
        if s.placement not in ("free", "physics"):
            errors.append("placement must be 'free' or 'physics'")
        if s.object_size <= 0:
            errors.append("object_size must be positive")
        if c.dist_min <= 0 or c.dist_min >= c.dist_max:
            errors.append("need 0 < dist_min < dist_max")
        if c.elev_min > c.elev_max or c.fov_min > c.fov_max:
            errors.append("min values must not exceed max values (elevation, fov)")
        if a.material_mode not in ("random", "original", "mixed"):
            errors.append("material_mode must be random, original or mixed")
        if a.lights_max < 1:
            errors.append("lights_max must be >= 1")
        if a.light_min > a.light_max:
            errors.append("light_min must not exceed light_max")
        if errors:
            raise ValueError("; ".join(errors))


PRESETS: dict[str, dict[str, Any]] = {
    "Quick look": {
        "dataset.num_images": 12, "dataset.width": 512, "dataset.height": 512,
        "appearance.samples": 16, "scene.instances_max": 2, "scene.distractors_max": 2,
    },
    "Balanced": {
        "dataset.num_images": 300, "dataset.width": 640, "dataset.height": 640,
        "appearance.samples": 64, "scene.instances_max": 4, "scene.distractors_max": 4,
    },
    "Heavy randomisation": {
        "dataset.num_images": 1000, "dataset.width": 640, "dataset.height": 640,
        "appearance.samples": 64, "scene.instances_max": 6, "scene.distractors_max": 8,
        "camera.elev_min": 5.0, "camera.roll_max": 30.0, "appearance.noise_max": 10.0,
        "appearance.blur_prob": 0.35, "appearance.material_mode": "mixed",
    },
}


def apply_preset(cfg: GenerationConfig, name: str) -> GenerationConfig:
    for key, value in PRESETS[name].items():
        cfg.set(key, value)
    return cfg
