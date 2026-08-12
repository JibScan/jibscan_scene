from __future__ import annotations

import json
import inspect
import os
from dataclasses import is_dataclass
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

import jibscan_scene.renderer as renderer_module
from jibscan_scene import (
    ManifestSceneComposer,
    MultiObjectSceneState,
    SceneComposer,
    SceneObject,
    SceneState,
    load_scene_manifest,
    render_scene,
)


def _manifest(path: Path, instance: Path, *, objects: list[dict] | None = None) -> Path:
    payload = {
        "schema_version": "jibscan.scene/v0.1",
        "scene_id": "scene:synthetic:001",
        "camera": {"fx": 900.0, "fy": 900.0, "cx": 512.0, "cy": 450.0, "width_px": 1024, "height_px": 900},
        "background": {"rgba": [48, 52, 58, 255]},
        "objects": objects
        if objects is not None
        else [
            {
                "object_id": "scene:squid:left",
                "instance_manifest": os.path.relpath(instance, path.parent),
                "target": {"subject_distance_m": 3.0, "center_px": [300.0, 450.0], "orientation_deg": -20.0},
                "z_index": 0,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _two_objects(instance: Path, base: Path) -> list[dict]:
    reference = os.path.relpath(instance, base.parent)
    return [
        {"object_id": "near", "instance_manifest": reference, "target": {"subject_distance_m": 1.0, "center_px": [300.0, 450.0], "orientation_deg": -15.0}, "z_index": 0},
        {"object_id": "far", "instance_manifest": reference, "target": {"subject_distance_m": 6.0, "center_px": [700.0, 450.0], "orientation_deg": 40.0}, "z_index": 3},
    ]


def _rejects(payload: dict, path: Path, match: str) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_scene_manifest(path)


def _assert_json_native(value: object) -> None:
    assert not isinstance(value, (tuple, Path, Image.Image))
    assert not is_dataclass(value)
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_json_native(key)
            _assert_json_native(item)
    elif isinstance(value, list):
        for item in value:
            _assert_json_native(item)


def _schema(name: str) -> dict:
    return json.loads((Path(__file__).parents[1] / "schemas" / name).read_text(encoding="utf-8"))


def _schema_errors(schema: dict, payload: dict) -> list:
    return list(Draft202012Validator(schema).iter_errors(payload))


def _assert_schema_and_runtime_reject(payload: dict, path: Path, schema: dict) -> None:
    assert _schema_errors(schema, payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        load_scene_manifest(path)


def test_valid_input_agrees_with_draft_2020_12_schema(synthetic_manifest: Path, tmp_path: Path) -> None:
    scene_path = _manifest(tmp_path / "scene.json", synthetic_manifest)
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    Draft202012Validator(_schema("scene_manifest.schema.json")).validate(payload)
    load_scene_manifest(scene_path)


def test_invalid_input_is_rejected_by_schema_and_runtime(synthetic_manifest: Path, tmp_path: Path) -> None:
    scene_path = _manifest(tmp_path / "scene.json", synthetic_manifest)
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    schema = _schema("scene_manifest.schema.json")
    cases = []

    missing_scene_id = json.loads(json.dumps(payload))
    missing_scene_id.pop("scene_id")
    cases.append(missing_scene_id)

    missing_object_id = json.loads(json.dumps(payload))
    missing_object_id["objects"][0].pop("object_id")
    cases.append(missing_object_id)

    missing_instance_manifest = json.loads(json.dumps(payload))
    missing_instance_manifest["objects"][0].pop("instance_manifest")
    cases.append(missing_instance_manifest)

    invalid_distance = json.loads(json.dumps(payload))
    invalid_distance["objects"][0]["target"]["subject_distance_m"] = 0
    cases.append(invalid_distance)

    invalid_center = json.loads(json.dumps(payload))
    invalid_center["objects"][0]["target"]["center_px"] = [1]
    cases.append(invalid_center)

    invalid_rgba = json.loads(json.dumps(payload))
    invalid_rgba["background"]["rgba"] = [0, 0, 0, 256]
    cases.append(invalid_rgba)

    invalid_z_index = json.loads(json.dumps(payload))
    invalid_z_index["objects"][0]["z_index"] = "front"
    cases.append(invalid_z_index)

    for index, case in enumerate(cases):
        _assert_schema_and_runtime_reject(case, tmp_path / f"invalid_{index}.json", schema)


def test_input_schema_allows_runtime_preserved_extension_fields(synthetic_manifest: Path, tmp_path: Path) -> None:
    scene_path = _manifest(tmp_path / "scene.json", synthetic_manifest)
    payload = json.loads(scene_path.read_text(encoding="utf-8"))
    payload["future_scene_field"] = {"preserved": True}
    payload["camera"]["future_camera_field"] = "allowed"
    payload["objects"][0]["future_object_field"] = 1
    Draft202012Validator(_schema("scene_manifest.schema.json")).validate(payload)
    scene_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_scene_manifest(scene_path).scene_id == "scene:synthetic:001"


def test_one_object_manifest_preserves_scene_contract(synthetic_manifest: Path, tmp_path: Path) -> None:
    scene_path = tmp_path / "scenes" / "one.json"
    scene_path.parent.mkdir()
    _manifest(scene_path, synthetic_manifest)
    scene = load_scene_manifest(scene_path)
    assert scene.scene_id == "scene:synthetic:001"
    assert scene.camera.width_px == 1024
    assert scene.background_rgba == (48, 52, 58, 255)
    obj = scene.objects[0]
    assert (obj.object_id, obj.instance.instance_id, obj.state.target_distance_m) == (
        "scene:squid:left", "synthetic:squid:001", 3.0
    )
    assert obj.state.center_px == (300.0, 450.0)
    assert obj.state.orientation_deg == -20.0
    assert obj.z_index == 0


def test_multi_object_duplicate_source_and_cwd_independent(synthetic_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene_path = tmp_path / "scenes" / "multi.json"
    scene_path.parent.mkdir()
    _manifest(scene_path, synthetic_manifest, objects=_two_objects(synthetic_manifest, scene_path))
    monkeypatch.chdir(tmp_path)
    scene = load_scene_manifest(scene_path)
    assert [obj.object_id for obj in scene.objects] == ["near", "far"]
    assert [obj.instance.instance_id for obj in scene.objects] == ["synthetic:squid:001"] * 2
    assert scene.objects[0].state.center_px != scene.objects[1].state.center_px
    assert scene.objects[0].state.orientation_deg != scene.objects[1].state.orientation_deg


def test_compose_writes_rgba_and_json_native_provenance(synthetic_manifest: Path, tmp_path: Path) -> None:
    scene_path = _manifest(tmp_path / "scene.json", synthetic_manifest)
    output = ManifestSceneComposer().compose(scene_path, tmp_path / "out")
    _assert_json_native(output)
    assert json.dumps(output)
    assert output["image"]["rgba_path"] == "scene_synthetic_001.png"
    with Image.open(tmp_path / "out" / output["image"]["rgba_path"]) as image:
        assert image.mode == "RGBA"
        assert image.size == (1024, 900)
    assert json.loads((tmp_path / "out" / "scene_synthetic_001.json").read_text()) == output


def test_output_metadata_and_disk_manifest_match_output_schema(synthetic_manifest: Path, tmp_path: Path) -> None:
    scene_path = _manifest(tmp_path / "scene.json", synthetic_manifest)
    metadata = ManifestSceneComposer().compose(scene_path, tmp_path / "out")
    schema = _schema("rendered_scene_manifest.schema.json")
    Draft202012Validator(schema).validate(metadata)
    written = json.loads((tmp_path / "out" / "scene_synthetic_001.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(written)
    assert written == metadata


def test_output_matches_existing_render_scene_and_preserves_order(synthetic_manifest: Path, tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    _manifest(scene_path, synthetic_manifest, objects=_two_objects(synthetic_manifest, scene_path))
    scene = load_scene_manifest(scene_path)
    expected = render_scene(scene).composite_rgba
    metadata = ManifestSceneComposer().compose(scene_path, tmp_path / "out")
    with Image.open(tmp_path / "out" / metadata["image"]["rgba_path"]) as image:
        assert image.tobytes() == expected.tobytes()
    assert [(o["object_id"], o["z_index"], o["original_input_index"]) for o in metadata["objects"]] == [
        ("near", 0, 0), ("far", 3, 1)
    ]
    assert metadata["objects"][0]["source_instance_id"] != metadata["objects"][0]["object_id"]
    assert metadata["objects"][0]["transform"]["target_distance_m"] == 1.0


def test_identical_inputs_produce_identical_outputs(synthetic_manifest: Path, tmp_path: Path) -> None:
    scene_path = _manifest(tmp_path / "scene.json", synthetic_manifest)
    composer = ManifestSceneComposer()
    first = composer.compose(scene_path, tmp_path / "run-a")
    second = composer.compose(scene_path, tmp_path / "run-b")
    assert first == second
    assert (tmp_path / "run-a" / first["image"]["rgba_path"]).read_bytes() == (
        tmp_path / "run-b" / second["image"]["rgba_path"]
    ).read_bytes()


def test_empty_scene_is_background_and_deterministic(tmp_path: Path) -> None:
    scene_path = tmp_path / "empty.json"
    _manifest(scene_path, tmp_path / "unused.json", objects=[])
    scene_path.write_text(scene_path.read_text().replace("unused.json", "unused.json"), encoding="utf-8")
    metadata = ManifestSceneComposer().compose(scene_path, tmp_path / "out")
    with Image.open(tmp_path / "out" / metadata["image"]["rgba_path"]) as image:
        assert image.getpixel((0, 0)) == (48, 52, 58, 255)
        assert image.size == (1024, 900)


def test_invalid_scene_fields_are_rejected(synthetic_manifest: Path, tmp_path: Path) -> None:
    scene_path = _manifest(tmp_path / "scene.json", synthetic_manifest)
    payload = json.loads(scene_path.read_text())
    for field, value, match in [
        ("schema_version", None, "schema_version"),
        ("scene_id", None, "scene_id"),
        ("camera", None, "camera"),
    ]:
        broken = dict(payload)
        if value is None:
            broken.pop(field)
        _rejects(broken, tmp_path / f"{field}.json", match)
    for key, value, match in [
        ("object_id", None, "object_id"),
        ("instance_manifest", None, "instance_manifest"),
        ("z_index", "bad", "z_index"),
    ]:
        broken = json.loads(scene_path.read_text())
        if value is None:
            broken["objects"][0].pop(key)
        else:
            broken["objects"][0][key] = value
        _rejects(broken, tmp_path / f"{key}.json", match)
    broken = json.loads(scene_path.read_text())
    broken["objects"][0]["target"].pop("subject_distance_m")
    _rejects(broken, tmp_path / "distance.json", "subject_distance_m")


def test_composer_delegates_to_existing_renderer(synthetic_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene_path = _manifest(tmp_path / "scene.json", synthetic_manifest)
    called = False
    original = renderer_module.render_scene

    def spy(scene: MultiObjectSceneState):
        nonlocal called
        called = True
        return original(scene)

    monkeypatch.setattr(renderer_module, "render_scene", spy)
    ManifestSceneComposer().compose(scene_path, tmp_path / "out")
    assert called


def test_scene_loader_reuses_normalized_instance_loader(
    synthetic_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene_path = _manifest(tmp_path / "scene.json", synthetic_manifest)
    import jibscan_scene.scene_manifest as scene_manifest_module

    called = False
    original = scene_manifest_module.loader.load_normalized_squid_instance

    def spy(path: Path):
        nonlocal called
        called = True
        return original(path)

    monkeypatch.setattr(scene_manifest_module.loader, "load_normalized_squid_instance", spy)
    load_scene_manifest(scene_path)
    assert called


def test_legacy_protocol_and_manifest_entrypoint_surfaces_are_distinct() -> None:
    legacy_parameters = list(inspect.signature(SceneComposer.compose).parameters)
    manifest_parameters = list(inspect.signature(ManifestSceneComposer.compose).parameters)
    assert legacy_parameters == ["self", "organism", "scene_condition", "output_dir"]
    assert manifest_parameters == ["self", "scene_manifest_path", "output_dir"]
