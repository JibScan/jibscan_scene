from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from jibscan_scene import ManifestSceneComposer


def _instance(tmp_path: Path, *, alpha: int = 255, vehicle_depth_m: float = 900.0) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image_path = tmp_path / "instance.png"
    Image.new("RGBA", (8, 8), (220, 30, 40, alpha)).save(image_path)
    manifest_path = tmp_path / "instance.json"
    manifest_path.write_text(
        json.dumps(
            {
                "instance_id": "source:shared",
                "image": {"rgba_path": image_path.name, "width_px": 8, "height_px": 8},
                "geometry": {"projected_width_px": 8, "projected_height_px": 8},
                "environment": {"vehicle_depth_m": vehicle_depth_m},
                "range": {"subject_distance_m": 3.0, "method": "synthetic"},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _scene(tmp_path: Path, instance: Path, *, version: str = "jibscan.scene/v0.1", policy: str | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    background = {"rgba": [48, 52, 58, 255]}
    if policy is not None:
        background["policy"] = policy
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(
        json.dumps(
            {
                "schema_version": version,
                "scene_id": "scene:geometry:test",
                "camera": {"fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 8.0, "width_px": 16, "height_px": 16},
                "background": background,
                "objects": [
                    {"object_id": "far", "instance_manifest": str(instance.resolve()), "target": {"subject_distance_m": 6.0, "center_px": [8.0, 8.0]}, "z_index": 10},
                    {"object_id": "near", "instance_manifest": str(instance.resolve()), "target": {"subject_distance_m": 2.0, "center_px": [8.0, 8.0]}, "z_index": 0},
                ],
            }
        ),
        encoding="utf-8",
    )
    return scene_path


def _schema(name: str) -> dict:
    return json.loads((Path(__file__).parents[1] / "schemas" / name).read_text(encoding="utf-8"))


def test_v01_geometry_defaults_to_invalid_background(tmp_path: Path) -> None:
    instance = _instance(tmp_path)
    scene = _scene(tmp_path, instance)
    metadata = ManifestSceneComposer().compose_with_geometry(scene, tmp_path / "out")
    sidecar = json.loads((tmp_path / "out" / metadata["geometry_manifest_path"]).read_text(encoding="utf-8"))
    range_map = np.load(tmp_path / "out" / sidecar["maps"]["range_m"]["path"])
    assert np.isnan(range_map[0, 0])
    assert Image.open(tmp_path / "out" / sidecar["maps"]["validity"]["path"]).getpixel((0, 0)) == 0
    assert sidecar["background_range"]["range_mode"] == "invalid"
    assert sidecar["representation"]["exact_for_current_scene"] is False


def test_geometry_maps_are_pixel_aligned_lossless_and_occurrence_specific(tmp_path: Path) -> None:
    instance = _instance(tmp_path)
    scene = _scene(tmp_path, instance)
    output = tmp_path / "out"
    metadata = ManifestSceneComposer().compose_with_geometry(
        scene, output, background_policy="constant", background_range_m=10.0
    )
    sidecar = json.loads((output / metadata["geometry_manifest_path"]).read_text(encoding="utf-8"))
    range_map = np.load(output / sidecar["maps"]["range_m"]["path"])
    visibility = np.load(output / sidecar["maps"]["visibility"]["path"])
    assert range_map.dtype == np.dtype("<f4")
    assert visibility.dtype == np.dtype("<i4")
    assert range_map.shape == visibility.shape == (16, 16)
    assert range_map[8, 8] == 6.0
    assert visibility[8, 8] == 1
    assert sidecar["visibility_labels"]["1"]["source_instance_id"] == "source:shared"
    assert sidecar["visibility_labels"]["2"]["occurrence_index"] == 1
    with Image.open(output / sidecar["maps"]["validity"]["path"]) as validity, Image.open(
        output / sidecar["maps"]["coverage"]["path"]
    ) as coverage:
        assert validity.mode == coverage.mode == "L"
        assert validity.size == coverage.size == (16, 16)
        assert validity.getpixel((8, 8)) == 255
        assert coverage.getpixel((8, 8)) == 255
        assert validity.getpixel((0, 0)) == 255
        assert coverage.getpixel((0, 0)) == 0
    assert range_map[0, 0] == 10.0
    assert sidecar["representation"]["exact_for_current_scene"] is True
    Draft202012Validator(_schema("scene_geometry.schema.json")).validate(sidecar)
    Draft202012Validator(_schema("rendered_scene_manifest_v0.2.schema.json")).validate(metadata)


def test_partial_alpha_multi_range_limit_is_reported(tmp_path: Path) -> None:
    instance = _instance(tmp_path, alpha=128)
    scene = _scene(tmp_path, instance)
    output = tmp_path / "out"
    metadata = ManifestSceneComposer().compose_with_geometry(
        scene, output, background_policy="constant", background_range_m=10.0
    )
    sidecar = json.loads((output / metadata["geometry_manifest_path"]).read_text(encoding="utf-8"))
    assert sidecar["exactness"]["partial_alpha_multi_range_pixels"] > 0
    assert sidecar["representation"]["exact_for_current_scene"] is False
    assert "partial_alpha_multiple_ranges" in sidecar["representation"]["limitations"]
    assert "not recoverable" in sidecar["exactness"]["range_map_limitation"]


def test_v02_manifest_carries_policy_without_changing_v01_loader(tmp_path: Path) -> None:
    instance = _instance(tmp_path)
    scene = _scene(tmp_path, instance, version="jibscan.scene/v0.2", policy="constant")
    data = json.loads(scene.read_text(encoding="utf-8"))
    data["background"]["range_m"] = 10.0
    scene.write_text(json.dumps(data), encoding="utf-8")
    Draft202012Validator(_schema("scene_manifest_v0.2.schema.json")).validate(
        json.loads(scene.read_text(encoding="utf-8"))
    )
    metadata = ManifestSceneComposer().compose_with_geometry(scene, tmp_path / "out")
    assert metadata["schema_version"] == "jibscan.rendered_scene/v0.2"


def test_invalid_background_is_not_fabricated(tmp_path: Path) -> None:
    instance = _instance(tmp_path)
    scene = _scene(tmp_path, instance)
    output = tmp_path / "out"
    metadata = ManifestSceneComposer().compose_with_geometry(scene, output)
    sidecar = json.loads((output / metadata["geometry_manifest_path"]).read_text(encoding="utf-8"))
    range_map = np.load(output / sidecar["maps"]["range_m"]["path"])
    validity = Image.open(output / sidecar["maps"]["validity"]["path"])
    visibility = np.load(output / sidecar["maps"]["visibility"]["path"])
    assert np.isnan(range_map[0, 0])
    assert validity.getpixel((0, 0)) == 0
    assert visibility[0, 0] == 0


def test_vehicle_depth_does_not_change_geometry_maps(tmp_path: Path) -> None:
    first = _instance(tmp_path / "a", vehicle_depth_m=900.0)
    second = _instance(tmp_path / "b", vehicle_depth_m=901.0)
    first_scene = _scene(first.parent, first)
    second_scene = _scene(second.parent, second)
    first_out = tmp_path / "first_out"
    second_out = tmp_path / "second_out"
    first_meta = ManifestSceneComposer().compose_with_geometry(first_scene, first_out, background_policy="constant", background_range_m=10.0)
    second_meta = ManifestSceneComposer().compose_with_geometry(second_scene, second_out, background_policy="constant", background_range_m=10.0)
    first_sidecar = json.loads((first_out / first_meta["geometry_manifest_path"]).read_text(encoding="utf-8"))
    second_sidecar = json.loads((second_out / second_meta["geometry_manifest_path"]).read_text(encoding="utf-8"))
    for key in ("range_m", "visibility", "validity", "coverage"):
        first_path = first_out / first_sidecar["maps"][key]["path"]
        second_path = second_out / second_sidecar["maps"][key]["path"]
        assert first_path.read_bytes() == second_path.read_bytes()


def test_controlled_alpha_and_deterministic_export(tmp_path: Path) -> None:
    instance = _instance(tmp_path / "source", alpha=128)
    scene = _scene(tmp_path / "scene", instance)
    data = json.loads(scene.read_text(encoding="utf-8"))
    data["objects"] = data["objects"][:1]
    scene.write_text(json.dumps(data), encoding="utf-8")
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    first_meta = ManifestSceneComposer().compose_with_geometry(scene, first_out)
    second_meta = ManifestSceneComposer().compose_with_geometry(scene, second_out)
    first_sidecar = json.loads((first_out / first_meta["geometry_manifest_path"]).read_text(encoding="utf-8"))
    second_sidecar = json.loads((second_out / second_meta["geometry_manifest_path"]).read_text(encoding="utf-8"))
    assert first_meta == second_meta
    for key in ("range_m", "visibility", "validity", "coverage"):
        assert (first_out / first_sidecar["maps"][key]["path"]).read_bytes() == (second_out / second_sidecar["maps"][key]["path"]).read_bytes()
    with Image.open(first_out / first_sidecar["maps"]["coverage"]["path"]) as coverage:
        assert coverage.getpixel((8, 8)) == 128
        assert coverage.getpixel((0, 0)) == 0


def test_generated_manifest_and_sidecar_schema_validation_constant_and_invalid_policies(tmp_path: Path) -> None:
    rendered_schema = _schema("rendered_scene_manifest_v0.2.schema.json")
    geometry_schema = _schema("scene_geometry.schema.json")
    rendered_validator = Draft202012Validator(rendered_schema)
    geometry_validator = Draft202012Validator(geometry_schema)

    instance = _instance(tmp_path)

    # 1. Constant policy with positive range_m generated from v0.2 SceneManifest
    const_scene = _scene(tmp_path / "scene_const", instance, version="jibscan.scene/v0.2", policy="constant")
    const_data = json.loads(const_scene.read_text(encoding="utf-8"))
    const_data["background"]["range_m"] = 10.0
    const_scene.write_text(json.dumps(const_data), encoding="utf-8")

    out_const = tmp_path / "out_const"
    meta_const = ManifestSceneComposer().compose_with_geometry(const_scene, out_const)
    sidecar_const = json.loads((out_const / meta_const["geometry_manifest_path"]).read_text(encoding="utf-8"))

    # Validate emitted artifacts directly against schemas
    rendered_validator.validate(meta_const)
    geometry_validator.validate(sidecar_const)
    assert meta_const["background"]["policy"] == "constant"
    assert meta_const["background"]["range_m"] == 10.0
    assert sidecar_const["background_range"]["range_mode"] == "constant"
    assert sidecar_const["background_range"]["range_m"] == 10.0

    # 2. Invalid policy with no range_m generated from v0.2 SceneManifest
    inv_scene = _scene(tmp_path / "scene_inv", instance, version="jibscan.scene/v0.2", policy="invalid")
    out_inv = tmp_path / "out_inv"
    meta_inv = ManifestSceneComposer().compose_with_geometry(inv_scene, out_inv)
    sidecar_inv = json.loads((out_inv / meta_inv["geometry_manifest_path"]).read_text(encoding="utf-8"))

    # Validate emitted artifacts directly against schemas
    rendered_validator.validate(meta_inv)
    geometry_validator.validate(sidecar_inv)
    assert meta_inv["background"]["policy"] == "invalid"
    assert "range_m" not in meta_inv["background"]
    assert sidecar_inv["background_range"]["range_mode"] == "invalid"
    assert "range_m" not in sidecar_inv["background_range"]

    # 3. Negative schema validation tests for RenderedSceneManifest
    # Constant policy missing range_m
    broken_meta_1 = json.loads(json.dumps(meta_const))
    del broken_meta_1["background"]["range_m"]
    assert list(rendered_validator.iter_errors(broken_meta_1))

    # Invalid policy with supplied range_m
    broken_meta_2 = json.loads(json.dumps(meta_inv))
    broken_meta_2["background"]["range_m"] = 10.0
    assert list(rendered_validator.iter_errors(broken_meta_2))

    # Constant policy with non-positive range_m
    broken_meta_3 = json.loads(json.dumps(meta_const))
    broken_meta_3["background"]["range_m"] = 0.0
    assert list(rendered_validator.iter_errors(broken_meta_3))

    # 4. Negative schema validation tests for SceneGeometrySidecar
    # Constant range_mode missing range_m
    broken_sidecar_1 = json.loads(json.dumps(sidecar_const))
    del broken_sidecar_1["background_range"]["range_m"]
    assert list(geometry_validator.iter_errors(broken_sidecar_1))

    # Invalid range_mode with supplied range_m
    broken_sidecar_2 = json.loads(json.dumps(sidecar_inv))
    broken_sidecar_2["background_range"]["range_m"] = 10.0
    assert list(geometry_validator.iter_errors(broken_sidecar_2))

    # Constant range_mode with non-positive range_m
    broken_sidecar_3 = json.loads(json.dumps(sidecar_const))
    broken_sidecar_3["background_range"]["range_m"] = 0.0
    assert list(geometry_validator.iter_errors(broken_sidecar_3))
