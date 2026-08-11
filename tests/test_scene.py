from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from jibscan_scene import CameraModel, SceneState, load_normalized_squid_instance, pinhole_range_scale, render_observation


def camera() -> CameraModel:
    return CameraModel(fx=900.0, fy=900.0, cx=512.0, cy=450.0, width_px=1024, height_px=900)


def test_fixture_loads_and_keeps_depth_separate_from_range(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    assert instance.range.subject_distance_m == 3.0
    assert instance.range.method == "synthetic"
    assert instance.environment.vehicle_depth_m == 500.0
    assert instance.provenance.synthetic_fixture is True


def test_loader_preserves_restoration_and_unknown_top_level_metadata(synthetic_manifest: Path) -> None:
    payload = json.loads(synthetic_manifest.read_text(encoding="utf-8"))
    payload["restoration"] = {"implementation": "identity", "quality_flags": []}
    payload["contract_revision"] = "fixture-extension"
    synthetic_manifest.write_text(json.dumps(payload), encoding="utf-8")
    instance = load_normalized_squid_instance(synthetic_manifest)
    assert instance.restoration["implementation"] == "identity"
    assert instance.extra["contract_revision"] == "fixture-extension"


def test_pinhole_range_scale_matches_expected_ratios() -> None:
    assert pinhole_range_scale(3.0, 1.0) == 3.0
    assert pinhole_range_scale(3.0, 3.0) == 1.0
    assert pinhole_range_scale(3.0, 6.0) == 0.5


def test_rendered_dimensions_follow_uniform_scale(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    expected = {1.0: (384, 768), 3.0: (128, 256), 6.0: (64, 128)}
    for distance, dimensions in expected.items():
        result = render_observation(instance, SceneState(camera=camera(), target_distance_m=distance))
        assert (result.transform.target_projected_width_px, result.transform.target_projected_height_px) == dimensions


def test_default_placement_centers_on_camera_principal_point(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    result = render_observation(instance, SceneState(camera=camera(), target_distance_m=3.0))
    assert result.transform.target_position_px == (448, 322)


def test_alpha_is_preserved_on_transformed_layer(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    result = render_observation(instance, SceneState(camera=camera(), target_distance_m=3.0))
    alpha = result.squid_layer_rgba.getchannel("A")
    assert alpha.getextrema() == (0, 255)


def test_render_is_deterministic_and_does_not_mutate_source_state(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    scene = SceneState(camera=camera(), target_distance_m=6.0)
    first = render_observation(instance, scene)
    second = render_observation(instance, scene)
    assert first.composite_rgba.tobytes() == second.composite_rgba.tobytes()
    assert first.transform == second.transform
    assert instance.range.subject_distance_m == 3.0
    assert instance.environment.vehicle_depth_m == 500.0


def test_source_rgba_dimensions_match_contract(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    with Image.open(instance.image.rgba_path) as image:
        assert image.mode == "RGBA"
        assert image.size == (128, 256)


def test_bioflow_generated_asset_can_use_same_boundary(synthetic_manifest: Path) -> None:
    payload = json.loads(synthetic_manifest.read_text(encoding="utf-8"))
    payload["instance_id"] = "bioflow:sample:001"
    payload["provenance"] = {"synthetic_fixture": False, "generator": "bioflow"}
    synthetic_manifest.write_text(json.dumps(payload), encoding="utf-8")
    instance = load_normalized_squid_instance(synthetic_manifest)
    result = render_observation(instance, SceneState(camera=camera(), target_distance_m=3.0))
    assert result.transform.source_instance_id == "bioflow:sample:001"
    assert instance.provenance.extra["generator"] == "bioflow"
