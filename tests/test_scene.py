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
    MultiObjectSceneState,
    NormalizedSquidInstance,
    RenderedObservation,
    RenderedScene,
    RenderedSceneObject,
    SceneObject,
    SceneState,
    load_normalized_squid_instance,
    pinhole_range_scale,
    render_observation,
    render_scene,
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


def create_color_squid_instance(
    base: NormalizedSquidInstance,
    tmp_path: Path,
    name: str,
    color: tuple[int, int, int, int],
    size: tuple[int, int] = (16, 16),
) -> NormalizedSquidInstance:
    path = tmp_path / f"{name}.png"
    img = Image.new("RGBA", size, color)
    img.save(path)
    return replace(
        base,
        instance_id=f"test:{name}",
        image=replace(base.image, rgba_path=path, width_px=size[0], height_px=size[1]),
        geometry=replace(base.geometry, projected_width_px=size[0], projected_height_px=size[1]),
    )


def test_scene_object_validation(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    state = SceneState(camera=camera(), target_distance_m=3.0)

    # Valid instantiation
    obj = SceneObject(object_id="squid_1", instance=instance, state=state, z_index=5)
    assert obj.object_id == "squid_1"
    assert obj.z_index == 5

    # Empty / whitespace object_id rejection
    for invalid_id in ("", "   "):
        with pytest.raises(ValueError, match="object_id"):
            SceneObject(object_id=invalid_id, instance=instance, state=state)

    # Non-finite or non-numeric z_index rejection
    for invalid_z in (math.nan, math.inf, -math.inf, True, False, "0"):
        with pytest.raises(ValueError, match="z_index"):
            SceneObject(object_id="valid_id", instance=instance, state=state, z_index=invalid_z)


def test_multi_object_scene_state_coerces_sequence_to_tuple(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    state = SceneState(camera=camera(), target_distance_m=3.0)
    obj1 = SceneObject(object_id="s1", instance=instance, state=state, z_index=0)
    obj2 = SceneObject(object_id="s2", instance=instance, state=state, z_index=1)

    scene = MultiObjectSceneState(camera=camera(), objects=[obj1, obj2])
    assert isinstance(scene.objects, tuple)
    assert len(scene.objects) == 2


def test_rendered_scene_coerces_objects_to_tuple() -> None:
    canvas = Image.new("RGBA", (100, 100), (0, 0, 0, 255))
    ro = RenderedSceneObject(
        object_id="s1",
        source_instance_id="inst_1",
        z_index=0,
        original_input_index=0,
        transform=TransformationMetadata(
            "inst_1", 8, 8, 8, 8, 3.0, 3.0, 1.0, 8, 8, (10.0, 10.0), (6, 6), None, camera()
        ),
        squid_layer_rgba=canvas,
    )
    rendered = RenderedScene(composite_rgba=canvas, objects=[ro])
    assert isinstance(rendered.objects, tuple)
    assert len(rendered.objects) == 1


def test_render_scene_rejects_duplicate_object_id(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    state = SceneState(camera=camera(), target_distance_m=3.0)
    obj1 = SceneObject(object_id="duplicate_id", instance=instance, state=state, z_index=0)
    obj2 = SceneObject(object_id="duplicate_id", instance=instance, state=state, z_index=1)

    with pytest.raises(ValueError, match="Duplicate object_id"):
        render_scene([obj1, obj2])


def test_render_scene_two_non_overlapping_squids(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    cam = camera()
    state1 = SceneState(camera=cam, target_distance_m=3.0, center_px=(200.0, 200.0))
    state2 = SceneState(camera=cam, target_distance_m=3.0, center_px=(800.0, 700.0))

    obj1 = SceneObject(object_id="squid_top_left", instance=instance, state=state1, z_index=0)
    obj2 = SceneObject(object_id="squid_bottom_right", instance=instance, state=state2, z_index=1)

    scene = MultiObjectSceneState(camera=cam, objects=(obj1, obj2))
    rendered = render_scene(scene)

    assert len(rendered.objects) == 2
    assert rendered.composite_rgba.size == (cam.width_px, cam.height_px)
    assert rendered.objects[0].object_id == "squid_top_left"
    assert rendered.objects[1].object_id == "squid_bottom_right"

    # Verify both regions contain non-background rendered pixels
    bg = scene.background_rgba
    assert rendered.composite_rgba.getpixel((200, 200)) != bg
    assert rendered.composite_rgba.getpixel((800, 700)) != bg


def test_render_scene_explicit_z_order_with_synthetic_overlapping_colored_pixels(
    synthetic_manifest: Path, tmp_path: Path
) -> None:
    base = load_normalized_squid_instance(synthetic_manifest)
    red_inst = create_color_squid_instance(base, tmp_path, "red", (255, 0, 0, 255), size=(32, 32))
    blue_inst = create_color_squid_instance(base, tmp_path, "blue", (0, 0, 255, 255), size=(32, 32))

    cam = camera()
    center = (512.0, 450.0)
    state_red = SceneState(camera=cam, target_distance_m=3.0, center_px=center)
    state_blue = SceneState(camera=cam, target_distance_m=3.0, center_px=center)

    # Red z=0, Blue z=1 -> Blue on top
    obj_red_lower = SceneObject(object_id="red_sq", instance=red_inst, state=state_red, z_index=0)
    obj_blue_higher = SceneObject(object_id="blue_sq", instance=blue_inst, state=state_blue, z_index=1)

    rendered_blue_top = render_scene([obj_red_lower, obj_blue_higher])
    assert rendered_blue_top.composite_rgba.getpixel((512, 450)) == (0, 0, 255, 255)

    # Blue z=0, Red z=1 -> Red on top
    obj_blue_lower = SceneObject(object_id="blue_sq", instance=blue_inst, state=state_blue, z_index=0)
    obj_red_higher = SceneObject(object_id="red_sq", instance=red_inst, state=state_red, z_index=1)

    rendered_red_top = render_scene([obj_blue_lower, obj_red_higher])
    assert rendered_red_top.composite_rgba.getpixel((512, 450)) == (255, 0, 0, 255)


def test_range_does_not_define_z_order(synthetic_manifest: Path, tmp_path: Path) -> None:
    base = load_normalized_squid_instance(synthetic_manifest)
    red_inst = create_color_squid_instance(base, tmp_path, "red", (255, 0, 0, 255), size=(64, 64))
    blue_inst = create_color_squid_instance(base, tmp_path, "blue", (0, 0, 255, 255), size=(64, 64))

    cam = camera()
    center = (512.0, 450.0)
    # Red is physically closer (1.0m) but has lower z_index (0)
    state_near_red = SceneState(camera=cam, target_distance_m=1.0, center_px=center)
    # Blue is physically farther (6.0m) but has higher z_index (1)
    state_far_blue = SceneState(camera=cam, target_distance_m=6.0, center_px=center)

    obj_near_red = SceneObject(object_id="near_red", instance=red_inst, state=state_near_red, z_index=0)
    obj_far_blue = SceneObject(object_id="far_blue", instance=blue_inst, state=state_far_blue, z_index=1)

    rendered = render_scene([obj_near_red, obj_far_blue])
    # Far squid with z=1 must win overlap against near squid with z=0
    assert rendered.composite_rgba.getpixel((512, 450)) == (0, 0, 255, 255)


def test_stable_tie_breaking_for_equal_z_index(synthetic_manifest: Path, tmp_path: Path) -> None:
    base = load_normalized_squid_instance(synthetic_manifest)
    red_inst = create_color_squid_instance(base, tmp_path, "red", (255, 0, 0, 255), size=(32, 32))
    blue_inst = create_color_squid_instance(base, tmp_path, "blue", (0, 0, 255, 255), size=(32, 32))

    cam = camera()
    center = (512.0, 450.0)
    state_red = SceneState(camera=cam, target_distance_m=3.0, center_px=center)
    state_blue = SceneState(camera=cam, target_distance_m=3.0, center_px=center)

    obj_red = SceneObject(object_id="red_obj", instance=red_inst, state=state_red, z_index=0)
    obj_blue = SceneObject(object_id="blue_obj", instance=blue_inst, state=state_blue, z_index=0)

    # Red first (index 0), Blue second (index 1) -> Blue composited on top
    res1 = render_scene([obj_red, obj_blue])
    res1_repeat = render_scene([obj_red, obj_blue])
    assert res1.composite_rgba.getpixel((512, 450)) == (0, 0, 255, 255)
    assert res1.composite_rgba.tobytes() == res1_repeat.composite_rgba.tobytes()
    assert res1.objects == res1_repeat.objects
    assert [ro.original_input_index for ro in res1.objects] == [0, 1]

    # Blue first (index 0), Red second (index 1) -> Red composited on top
    res2 = render_scene([obj_blue, obj_red])
    assert res2.composite_rgba.getpixel((512, 450)) == (255, 0, 0, 255)
    assert [ro.original_input_index for ro in res2.objects] == [0, 1]
    assert res2.objects[0].object_id == "blue_obj"
    assert res2.objects[1].object_id == "red_obj"


def test_duplicate_source_instance_with_different_states(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    cam = camera()

    state1 = SceneState(camera=cam, target_distance_m=2.0, center_px=(300.0, 300.0), orientation_deg=0.0)
    state2 = SceneState(camera=cam, target_distance_m=4.0, center_px=(700.0, 600.0), orientation_deg=45.0)

    obj1 = SceneObject(object_id="squid_occ_1", instance=instance, state=state1, z_index=0)
    obj2 = SceneObject(object_id="squid_occ_2", instance=instance, state=state2, z_index=1)

    rendered = render_scene([obj1, obj2])
    assert len(rendered.objects) == 2
    assert rendered.objects[0].source_instance_id == instance.instance_id
    assert rendered.objects[1].source_instance_id == instance.instance_id
    assert rendered.objects[0].object_id == "squid_occ_1"
    assert rendered.objects[1].object_id == "squid_occ_2"
    assert rendered.objects[0].transform.target_center_px == (300.0, 300.0)
    assert rendered.objects[1].transform.target_center_px == (700.0, 600.0)
    assert rendered.objects[0].transform.orientation_deg == 0.0
    assert rendered.objects[1].transform.orientation_deg == 45.0


def test_camera_mismatch_rejection(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    base_cam = camera()
    base_state = SceneState(camera=base_cam, target_distance_m=3.0)
    obj_base = SceneObject(object_id="obj_base", instance=instance, state=base_state)

    mismatched_cams = [
        CameraModel(fx=800.0, fy=900.0, cx=512.0, cy=450.0, width_px=1024, height_px=900),
        CameraModel(fx=900.0, fy=800.0, cx=512.0, cy=450.0, width_px=1024, height_px=900),
        CameraModel(fx=900.0, fy=900.0, cx=500.0, cy=450.0, width_px=1024, height_px=900),
        CameraModel(fx=900.0, fy=900.0, cx=512.0, cy=400.0, width_px=1024, height_px=900),
        CameraModel(fx=900.0, fy=900.0, cx=512.0, cy=450.0, width_px=800, height_px=900),
        CameraModel(fx=900.0, fy=900.0, cx=512.0, cy=450.0, width_px=1024, height_px=800),
    ]

    for mismatch in mismatched_cams:
        mismatch_state = SceneState(camera=mismatch, target_distance_m=3.0)
        obj_mismatch = SceneObject(object_id="obj_mismatch", instance=instance, state=mismatch_state)
        with pytest.raises(ValueError, match="Camera mismatch"):
            render_scene([obj_base, obj_mismatch])

    # Also check scene camera parameter mismatch
    scene_state = MultiObjectSceneState(camera=base_cam, objects=(obj_base,))
    with pytest.raises(ValueError, match="Explicit camera does not match"):
        render_scene(scene_state, camera=mismatched_cams[0])


def test_individual_transform_preservation(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    cam = camera()

    objects = [
        SceneObject(
            object_id="s1",
            instance=instance,
            state=SceneState(camera=cam, target_distance_m=1.5, center_px=(200.0, 300.0), orientation_deg=-15.0),
            z_index=2,
        ),
        SceneObject(
            object_id="s2",
            instance=instance,
            state=SceneState(camera=cam, target_distance_m=4.5, center_px=(600.0, 500.0), orientation_deg=60.0),
            z_index=0,
        ),
        SceneObject(
            object_id="s3",
            instance=instance,
            state=SceneState(camera=cam, target_distance_m=3.0, center_px=(400.0, 400.0), orientation_deg=0.0),
            z_index=1,
        ),
    ]

    rendered_scene = render_scene(objects)
    assert len(rendered_scene.objects) == 3

    for idx, (ro, obj) in enumerate(zip(rendered_scene.objects, objects)):
        standalone = render_observation(obj.instance, obj.state)
        assert ro.object_id == obj.object_id
        assert ro.source_instance_id == obj.instance.instance_id
        assert ro.z_index == obj.z_index
        assert ro.original_input_index == idx
        assert ro.transform == standalone.transform
        assert ro.squid_layer_rgba.tobytes() == standalone.squid_layer_rgba.tobytes()


def test_clipping_partially_outside_and_inside(synthetic_manifest: Path) -> None:
    instance = load_normalized_squid_instance(synthetic_manifest)
    cam = camera()

    obj_outside = SceneObject(
        object_id="outside",
        instance=instance,
        state=SceneState(camera=cam, target_distance_m=3.0, center_px=(-20.0, -30.0), orientation_deg=45.0),
        z_index=0,
    )
    obj_inside = SceneObject(
        object_id="inside",
        instance=instance,
        state=SceneState(camera=cam, target_distance_m=3.0, center_px=(512.0, 450.0)),
        z_index=1,
    )

    rendered = render_scene([obj_outside, obj_inside])
    assert rendered.composite_rgba.size == (cam.width_px, cam.height_px)
    assert rendered.objects[0].transform.target_top_left_px == (-156, -166)
    assert rendered.objects[0].squid_layer_rgba.getbbox() is not None
    assert rendered.objects[1].squid_layer_rgba.getbbox() is not None


def test_alpha_semantics_transparent_layer_does_not_erase_lower_layer(
    synthetic_manifest: Path, tmp_path: Path
) -> None:
    base = load_normalized_squid_instance(synthetic_manifest)

    # Bottom solid red 64x64
    red_inst = create_color_squid_instance(base, tmp_path, "red_solid", (255, 0, 0, 255), size=(64, 64))

    # Top layer: 64x64 with blue center (16x16) and transparent boundary
    top_path = tmp_path / "top_transparent.png"
    top_img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for x in range(24, 40):
        for y in range(24, 40):
            top_img.putpixel((x, y), (0, 0, 255, 255))
    top_img.save(top_path)
    top_inst = replace(
        base,
        instance_id="test:top_transparent",
        image=replace(base.image, rgba_path=top_path, width_px=64, height_px=64),
        geometry=replace(base.geometry, projected_width_px=64, projected_height_px=64),
    )

    cam = camera()
    center = (512.0, 450.0)
    state_bottom = SceneState(camera=cam, target_distance_m=3.0, center_px=center)
    state_top = SceneState(camera=cam, target_distance_m=3.0, center_px=center)

    obj_bottom = SceneObject(object_id="bottom_red", instance=red_inst, state=state_bottom, z_index=0)
    obj_top = SceneObject(object_id="top_trans", instance=top_inst, state=state_top, z_index=1)

    rendered = render_scene([obj_bottom, obj_top])

    # Center is blue from top layer
    assert rendered.composite_rgba.getpixel((512, 450)) == (0, 0, 255, 255)
    # Region at (500, 430) is inside red bottom layer footprint but top layer is transparent:
    # It must remain pure red and NOT be erased to background!
    assert rendered.composite_rgba.getpixel((500, 430)) == (255, 0, 0, 255)


def test_empty_scene_handling() -> None:
    cam = camera()
    # 0 objects with MultiObjectSceneState
    scene = MultiObjectSceneState(camera=cam, objects=(), background_rgba=(48, 52, 58, 255))
    rendered = render_scene(scene)
    assert rendered.objects == ()
    assert rendered.composite_rgba.size == (cam.width_px, cam.height_px)
    assert rendered.composite_rgba.getpixel((0, 0)) == (48, 52, 58, 255)
    assert rendered.composite_rgba.getpixel((512, 450)) == (48, 52, 58, 255)

    # 0 objects with empty sequence and explicit camera
    custom_bg = (10, 20, 30, 255)
    rendered_empty_seq = render_scene([], camera=cam, background_rgba=custom_bg)
    assert rendered_empty_seq.objects == ()
    assert rendered_empty_seq.composite_rgba.getpixel((0, 0)) == custom_bg

    # 0 objects with empty sequence without camera raises ValueError
    with pytest.raises(ValueError, match="camera must be provided"):
        render_scene([])
