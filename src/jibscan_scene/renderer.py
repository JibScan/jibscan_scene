from __future__ import annotations

from PIL import Image

from .contracts import NormalizedSquidInstance, RenderedObservation, SceneState, TransformationMetadata
from .projection import pinhole_range_scale


def _scaled_size(instance: NormalizedSquidInstance, scale: float) -> tuple[int, int]:
    width = max(1, round(instance.geometry.projected_width_px * scale))
    height = max(1, round(instance.geometry.projected_height_px * scale))
    return width, height


def _placement(scene: SceneState, size: tuple[int, int]) -> tuple[int, int]:
    if scene.position_px is not None:
        return scene.position_px
    width, height = size
    return (round(scene.camera.cx - width / 2), round(scene.camera.cy - height / 2))


def render_observation(instance: NormalizedSquidInstance, scene: SceneState) -> RenderedObservation:
    source_distance = instance.range.subject_distance_m
    scale = pinhole_range_scale(source_distance, scene.target_distance_m)
    target_size = _scaled_size(instance, scale)
    position = _placement(scene, target_size)

    source_rgba = Image.open(instance.image.rgba_path).convert("RGBA")
    if source_rgba.size != (
        instance.geometry.projected_width_px,
        instance.geometry.projected_height_px,
    ):
        raise ValueError(
            "RGBA pixel dimensions must match geometry.projected_width_px/projected_height_px in v0"
        )

    transformed = source_rgba.resize(target_size, resample=Image.Resampling.LANCZOS)

    layer = Image.new("RGBA", (scene.camera.width_px, scene.camera.height_px), (0, 0, 0, 0))
    layer.alpha_composite(transformed, dest=position)

    composite = Image.new(
        "RGBA", (scene.camera.width_px, scene.camera.height_px), scene.background_rgba
    )
    composite.alpha_composite(layer)

    metadata = TransformationMetadata(
        source_instance_id=instance.instance_id,
        source_projected_width_px=instance.geometry.projected_width_px,
        source_projected_height_px=instance.geometry.projected_height_px,
        source_distance_m=source_distance,
        target_distance_m=scene.target_distance_m,
        scale=scale,
        target_projected_width_px=target_size[0],
        target_projected_height_px=target_size[1],
        target_position_px=position,
        vehicle_depth_m=instance.environment.vehicle_depth_m,
        camera=scene.camera,
    )
    return RenderedObservation(layer, composite, metadata)
