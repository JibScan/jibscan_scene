from __future__ import annotations

from PIL import Image

from .contracts import NormalizedSquidInstance, RenderedObservation, SceneState, TransformationMetadata
from .projection import pinhole_range_scale


def _scaled_size(instance: NormalizedSquidInstance, scale: float) -> tuple[int, int]:
    width = max(1, round(instance.geometry.projected_width_px * scale))
    height = max(1, round(instance.geometry.projected_height_px * scale))
    return width, height


def _placement(scene: SceneState, size: tuple[int, int]) -> tuple[tuple[float, float], tuple[int, int]]:
    center = scene.center_px if scene.center_px is not None else (scene.camera.cx, scene.camera.cy)
    width, height = size
    return center, (round(center[0] - width / 2), round(center[1] - height / 2))


def render_observation(instance: NormalizedSquidInstance, scene: SceneState) -> RenderedObservation:
    source_distance = instance.range.subject_distance_m
    scale = pinhole_range_scale(source_distance, scene.target_distance_m)
    target_size = _scaled_size(instance, scale)

    with Image.open(instance.image.rgba_path) as source_image:
        if source_image.mode != "RGBA":
            raise ValueError("source raster must be RGBA in v0")
        source_rgba = source_image.copy()
    declared_size = (instance.image.width_px, instance.image.height_px)
    if all(value is not None for value in declared_size) and source_rgba.size != declared_size:
        raise ValueError("RGBA pixel dimensions must match image.width_px/height_px in v0")

    center, top_left = _placement(scene, target_size)
    transformed = source_rgba.resize(target_size, resample=Image.Resampling.LANCZOS)

    layer = Image.new("RGBA", (scene.camera.width_px, scene.camera.height_px), (0, 0, 0, 0))
    layer.alpha_composite(transformed, dest=top_left)

    composite = Image.new(
        "RGBA", (scene.camera.width_px, scene.camera.height_px), scene.background_rgba
    )
    composite.alpha_composite(layer)

    metadata = TransformationMetadata(
        source_instance_id=instance.instance_id,
        source_raster_width_px=source_rgba.width,
        source_raster_height_px=source_rgba.height,
        source_projected_width_px=instance.geometry.projected_width_px,
        source_projected_height_px=instance.geometry.projected_height_px,
        source_distance_m=source_distance,
        target_distance_m=scene.target_distance_m,
        scale=scale,
        target_projected_width_px=target_size[0],
        target_projected_height_px=target_size[1],
        target_center_px=center,
        target_top_left_px=top_left,
        vehicle_depth_m=instance.environment.vehicle_depth_m,
        camera=scene.camera,
    )
    return RenderedObservation(layer, composite, metadata)
