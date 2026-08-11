from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from jibscan_scene.contracts import TransformationMetadata
from jibscan_scene import (
    CameraModel,
    SceneState,
    load_normalized_squid_instance,
    pinhole_range_scale,
    render_observation,
)


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


def test_orientation_accepts_finite_values_and_rejects_nonfinite() -> None:
    scene_camera = camera()
    assert SceneState(camera=scene_camera, target_distance_m=3.0, orientation_deg=45.0).orientation_deg == 45.0
    for orientation in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="orientation_deg"):
            SceneState(camera=scene_camera, target_distance_m=3.0, orientation_deg=orientation)


def test_scene_state_preserves_existing_positional_background_argument() -> None:
    background = (1, 2, 3, 4)
    scene = SceneState(camera(), 3.0, None, background)
    assert scene.background_rgba == background
    assert scene.orientation_deg == 0.0


def test_transformation_metadata_preserves_existing_positional_fields() -> None:
    metadata = TransformationMetadata(
        "instance",
        8,
        16,
        8,
        16,
        3.0,
        3.0,
        1.0,
        8,
        16,
        (10.0, 20.0),
        (6, 12),
        None,
        camera(),
    )
    assert metadata.target_center_px == (10.0, 20.0)
    assert metadata.orientation_deg == 0.0


def test_rendered_dimensions_follow_uniform_scale(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    expected = {1.0: (384, 768), 3.0: (128, 256), 6.0: (64, 128)}
    for distance, dimensions in expected.items():
        result = render_observation(instance, SceneState(camera=camera(), target_distance_m=distance))
        assert (result.transform.target_projected_width_px, result.transform.target_projected_height_px) == dimensions


def test_default_placement_centers_on_camera_principal_point(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    result = render_observation(instance, SceneState(camera=camera(), target_distance_m=3.0))
    assert result.transform.target_center_px == (512.0, 450.0)
    assert result.transform.target_top_left_px == (448, 322)


def test_zero_rotation_preserves_prior_raster_and_projected_geometry(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    result = render_observation(instance, SceneState(camera=camera(), target_distance_m=3.0))
    with Image.open(instance.image.rgba_path) as source:
        expected = source.resize((128, 256), resample=Image.Resampling.LANCZOS)
    x, y = result.transform.target_top_left_px
    assert result.squid_layer_rgba.crop((x, y, x + 128, y + 256)).tobytes() == expected.tobytes()
    assert (result.transform.rotated_raster_width_px, result.transform.rotated_raster_height_px) == (128, 256)


def test_explicit_center_is_invariant_across_ranges(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    center = (700.25, 120.75)
    results = [
        render_observation(
            instance,
            SceneState(camera=camera(), target_distance_m=distance, center_px=center),
        )
        for distance in (1.0, 3.0, 6.0)
    ]
    assert [result.transform.target_center_px for result in results] == [center] * 3


def test_explicit_center_is_invariant_across_orientations(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    center = (700.25, 120.75)
    results = [
        render_observation(
            instance,
            SceneState(
                camera=camera(),
                target_distance_m=3.0,
                center_px=center,
                orientation_deg=orientation,
            ),
        )
        for orientation in (0.0, 45.0, 90.0, -30.0)
    ]
    assert [result.transform.target_center_px for result in results] == [center] * 4


def test_explicit_center_controls_target_top_left(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    result = render_observation(
        instance,
        SceneState(camera=camera(), target_distance_m=3.0, center_px=(700.25, 120.75)),
    )
    assert result.transform.target_top_left_px == (636, -7)


def test_rotation_uses_pillow_expanded_footprint_after_resize(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    result = render_observation(
        instance,
        SceneState(camera=camera(), target_distance_m=3.0, orientation_deg=45.0),
    )
    with Image.open(instance.image.rgba_path) as source:
        expected = source.resize((128, 256), resample=Image.Resampling.LANCZOS).rotate(
            45.0, expand=True, fillcolor=(0, 0, 0, 0)
        )
    x, y = result.transform.target_top_left_px
    assert (result.transform.target_projected_width_px, result.transform.target_projected_height_px) == (128, 256)
    assert (result.transform.rotated_raster_width_px, result.transform.rotated_raster_height_px) == expected.size
    assert expected.getchannel("A").getbbox() is not None
    assert expected.size[0] > 128 and expected.size[1] > 256
    assert result.squid_layer_rgba.crop((x, y, x + expected.width, y + expected.height)).tobytes() == expected.tobytes()


def test_ninety_degree_rotation_swaps_rectangular_raster_footprint(
    synthetic_manifest: Path,
) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    result = render_observation(
        instance,
        SceneState(camera=camera(), target_distance_m=3.0, orientation_deg=90.0),
    )
    assert (result.transform.target_projected_width_px, result.transform.target_projected_height_px) == (128, 256)
    assert (result.transform.rotated_raster_width_px, result.transform.rotated_raster_height_px) == (256, 128)


def test_positive_rotation_is_counter_clockwise_in_rendered_image(
    synthetic_manifest: Path, tmp_path: Path
) -> None:
    base = load_normalized_squid_instance(synthetic_manifest)
    path = tmp_path / "asymmetric.png"
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    image.putpixel((6, 1), (255, 0, 0, 255))
    image.putpixel((1, 6), (0, 255, 0, 255))
    image.save(path)
    instance = replace(
        base,
        image=replace(base.image, rgba_path=path, width_px=8, height_px=8),
        geometry=replace(base.geometry, projected_width_px=8, projected_height_px=8),
    )
    result = render_observation(
        instance,
        SceneState(camera=camera(), target_distance_m=3.0, orientation_deg=90.0),
    )
    x, y = result.transform.target_top_left_px
    rotated = image.rotate(90.0, expand=True, fillcolor=(0, 0, 0, 0))
    crop = result.squid_layer_rgba.crop((x, y, x + rotated.width, y + rotated.height))
    assert crop.tobytes() == rotated.tobytes()
    assert crop.getpixel((1, 1)) == (255, 0, 0, 255)
    assert crop.getpixel((6, 6)) == (0, 255, 0, 255)


def test_fractional_center_and_orientation_are_deterministic(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    scene = SceneState(
        camera=camera(), center_px=(10.25, 10.5), target_distance_m=3.0, orientation_deg=45.0
    )
    first = render_observation(instance, scene)
    second = render_observation(instance, scene)
    assert first.transform.target_center_px == (10.25, 10.5)
    assert first.transform.target_top_left_px == (-126, -126)
    assert first.transform == second.transform
    assert first.squid_layer_rgba.tobytes() == second.squid_layer_rgba.tobytes()


def test_explicit_center_is_independent_of_camera_raster_resolution(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    raster_path = synthetic_manifest.parent / "higher_resolution_squid.png"
    with Image.open(instance.image.rgba_path) as source:
        source.resize((256, 512), resample=Image.Resampling.NEAREST).save(raster_path)
    higher_resolution = replace(
        instance,
        image=replace(instance.image, rgba_path=raster_path, width_px=256, height_px=512),
    )
    center = (300.25, 220.75)
    scene = SceneState(camera=camera(), target_distance_m=3.0, center_px=center, orientation_deg=45.0)
    first = render_observation(instance, scene)
    second = render_observation(higher_resolution, scene)
    assert first.transform.target_projected_width_px == second.transform.target_projected_width_px
    assert first.transform.target_projected_height_px == second.transform.target_projected_height_px
    assert first.transform.target_center_px == second.transform.target_center_px
    assert first.transform.target_top_left_px == second.transform.target_top_left_px
    assert (first.transform.rotated_raster_width_px, first.transform.rotated_raster_height_px) == (272, 272)
    assert (second.transform.rotated_raster_width_px, second.transform.rotated_raster_height_px) == (272, 272)
    assert first.transform.source_raster_width_px == 128
    assert second.transform.source_raster_width_px == 256


def test_vehicle_depth_does_not_change_render_geometry(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    changed = instance.__class__(
        instance_id=instance.instance_id,
        image=instance.image,
        geometry=instance.geometry,
        environment=instance.environment.__class__(vehicle_depth_m=12.0),
        range=instance.range,
        provenance=instance.provenance,
    )
    scene = SceneState(camera=camera(), target_distance_m=3.0, center_px=(500.5, 300.5))
    first = render_observation(instance, scene)
    second = render_observation(changed, scene)
    assert first.squid_layer_rgba.tobytes() == second.squid_layer_rgba.tobytes()
    assert first.transform.target_top_left_px == second.transform.target_top_left_px
    assert first.transform.vehicle_depth_m == 500.0
    assert second.transform.vehicle_depth_m == 12.0


def test_fractional_odd_size_uses_python_round_for_rasterization(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    scene = SceneState(camera=camera(), target_distance_m=384 / 127, center_px=(10.25, 10.5))
    first = render_observation(instance, scene)
    second = render_observation(instance, scene)
    assert (first.transform.target_projected_width_px, first.transform.target_projected_height_px) == (127, 254)
    assert first.transform.target_top_left_px == (round(10.25 - 127 / 2), round(10.5 - 254 / 2))
    assert first.transform == second.transform
    assert first.squid_layer_rgba.tobytes() == second.squid_layer_rgba.tobytes()


def test_negative_placement_is_clipped_without_repositioning(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    result = render_observation(
        instance,
        SceneState(camera=camera(), target_distance_m=3.0, center_px=(-20.0, -30.0), orientation_deg=45.0),
    )
    assert result.transform.target_top_left_px == (-156, -166)
    assert result.transform.target_center_px == (-20.0, -30.0)
    assert result.squid_layer_rgba.getbbox() is not None


def test_rotated_alpha_extends_beyond_unrotated_rectangle(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    center = (512.0, 450.0)
    result = render_observation(
        instance,
        SceneState(camera=camera(), target_distance_m=3.0, center_px=center, orientation_deg=45.0),
    )
    unrotated_left = round(center[0] - result.transform.target_projected_width_px / 2)
    unrotated_top = round(center[1] - result.transform.target_projected_height_px / 2)
    alpha_bbox = result.squid_layer_rgba.getchannel("A").getbbox()
    assert alpha_bbox is not None
    assert alpha_bbox[0] < unrotated_left or alpha_bbox[1] < unrotated_top


def test_rotation_does_not_change_scale_or_projected_dimensions(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    unrotated = render_observation(instance, SceneState(camera=camera(), target_distance_m=1.0))
    rotated = render_observation(
        instance, SceneState(camera=camera(), target_distance_m=1.0, orientation_deg=90.0)
    )
    assert rotated.transform.scale == unrotated.transform.scale
    assert (
        rotated.transform.target_projected_width_px,
        rotated.transform.target_projected_height_px,
    ) == (384, 768)
    assert (
        rotated.transform.rotated_raster_width_px,
        rotated.transform.rotated_raster_height_px,
    ) == (768, 384)
    assert rotated.transform.orientation_deg == 90.0


def test_alpha_is_preserved_on_transformed_layer(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    result = render_observation(instance, SceneState(camera=camera(), target_distance_m=3.0))
    assert result.squid_layer_rgba.mode == "RGBA"
    assert result.composite_rgba.mode == "RGBA"
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
