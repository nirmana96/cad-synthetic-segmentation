import pytest

from cad2coco.config import PRESETS, GenerationConfig, apply_preset


def test_yaml_roundtrip(tmp_path):
    cfg = GenerationConfig()
    cfg.scene.placement = "physics"
    cfg.camera.dist_max = 2.5
    path = tmp_path / "c.yaml"
    cfg.to_yaml(path)
    assert GenerationConfig.from_yaml(path) == cfg


def test_repo_default_config_matches_code():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    assert GenerationConfig.from_yaml(path) == GenerationConfig()


@pytest.mark.parametrize("key,value", [
    ("scene.instances_min", 9), ("camera.dist_min", 5.0), ("scene.placement", "orbit"),
    ("appearance.material_mode", "chrome"), ("dataset.valid_ratio", 0.95),
])
def test_invalid_values_rejected(key, value):
    cfg = GenerationConfig()
    cfg.set(key, value)
    with pytest.raises(ValueError):
        cfg.validate()


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="Unknown config key"):
        GenerationConfig.from_dict({"scene": {"gravity": 1}})


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_are_valid(name):
    apply_preset(GenerationConfig(), name).validate()
