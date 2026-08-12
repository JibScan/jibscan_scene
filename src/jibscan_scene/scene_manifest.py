from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from . import loader, renderer
from .contracts import CameraModel, MultiObjectSceneState, SceneObject, SceneState

SCENE_SCHEMA_VERSION = "jibscan.scene/v0.1"
RENDERED_SCENE_SCHEMA_VERSION = "jibscan.rendered_scene/v0.1"
RENDERER_VERSION = "jibscan-scene/0.1.0"


def _error(path: str, message: str) -> ValueError:
    return ValueError(f"SceneManifest {path}: {message}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _error(path, "must be an object")
    return value


def _required(data: dict[str, Any], key: str, path: str) -> Any:
    if key not in data:
        raise _error(f"{path}.{key}", "is required")
    return data[key]


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(path, "must be a non-empty string")
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _error(path, "must be a finite number")
    if positive and value <= 0:
        raise _error(path, "must be > 0")
    return float(value)


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(path, "must be an integer")
    return value


def _rgba(value: Any, path: str) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise _error(path, "must be a four-item array")
    channels = tuple(_integer(channel, f"{path}[{index}]") for index, channel in enumerate(value))
    if any(channel < 0 or channel > 255 for channel in channels):
        raise _error(path, "channels must be between 0 and 255")
    return channels


def _camera(data: Any) -> CameraModel:
    camera = _object(data, "camera")
    return CameraModel(
        fx=_number(_required(camera, "fx", "camera"), "camera.fx", positive=True),
        fy=_number(_required(camera, "fy", "camera"), "camera.fy", positive=True),
        cx=_number(_required(camera, "cx", "camera"), "camera.cx"),
        cy=_number(_required(camera, "cy", "camera"), "camera.cy"),
        width_px=_integer(_required(camera, "width_px", "camera"), "camera.width_px"),
        height_px=_integer(_required(camera, "height_px", "camera"), "camera.height_px"),
    )


def _scene_object(data: Any, index: int, camera: CameraModel, manifest_path: Path) -> SceneObject:
    path = f"objects[{index}]"
    item = _object(data, path)
    object_id = _string(_required(item, "object_id", path), f"{path}.object_id")
    instance_ref = _string(_required(item, "instance_manifest", path), f"{path}.instance_manifest")
    target = _object(_required(item, "target", path), f"{path}.target")
    distance = _number(
        _required(target, "subject_distance_m", f"{path}.target"),
        f"{path}.target.subject_distance_m",
        positive=True,
    )
    center = _required(target, "center_px", f"{path}.target")
    if not isinstance(center, list) or len(center) != 2:
        raise _error(f"{path}.target.center_px", "must be a two-item array")
    center_px = (
        _number(center[0], f"{path}.target.center_px[0]"),
        _number(center[1], f"{path}.target.center_px[1]"),
    )
    orientation = _number(target.get("orientation_deg", 0.0), f"{path}.target.orientation_deg")
    z_index = _number(_required(item, "z_index", path), f"{path}.z_index")
    if z_index.is_integer():
        z_index = int(z_index)
    instance_path = Path(instance_ref)
    if not instance_path.is_absolute():
        instance_path = (manifest_path.parent / instance_path).resolve()
    instance = loader.load_normalized_squid_instance(instance_path)
    return SceneObject(
        object_id=object_id,
        instance=instance,
        state=SceneState(
            camera=camera,
            target_distance_m=distance,
            center_px=center_px,
            orientation_deg=orientation,
        ),
        z_index=z_index,
    )


def load_scene_manifest(path: str | Path) -> MultiObjectSceneState:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = _object(payload, "$")
    version = _string(_required(data, "schema_version", "$"), "$.schema_version")
    if version != SCENE_SCHEMA_VERSION:
        raise _error("$.schema_version", f"must equal {SCENE_SCHEMA_VERSION!r}")
    scene_id = _string(_required(data, "scene_id", "$"), "$.scene_id")
    camera = _camera(_required(data, "camera", "$"))
    background = _object(_required(data, "background", "$"), "background")
    background_rgba = _rgba(_required(background, "rgba", "background"), "background.rgba")
    objects = _required(data, "objects", "$")
    if not isinstance(objects, list):
        raise _error("$.objects", "must be an array")
    scene_objects = tuple(
        _scene_object(item, index, camera, manifest_path) for index, item in enumerate(objects)
    )
    return MultiObjectSceneState(
        camera=camera,
        objects=scene_objects,
        background_rgba=background_rgba,
        scene_id=scene_id,
        manifest_path=manifest_path,
    )


def _camera_metadata(camera: CameraModel) -> dict[str, Any]:
    return {
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "width_px": camera.width_px,
        "height_px": camera.height_px,
    }


def _transform_metadata(transform: Any) -> dict[str, Any]:
    return {
        "source_raster_width_px": transform.source_raster_width_px,
        "source_raster_height_px": transform.source_raster_height_px,
        "source_projected_width_px": transform.source_projected_width_px,
        "source_projected_height_px": transform.source_projected_height_px,
        "source_distance_m": transform.source_distance_m,
        "target_distance_m": transform.target_distance_m,
        "scale": transform.scale,
        "target_projected_width_px": transform.target_projected_width_px,
        "target_projected_height_px": transform.target_projected_height_px,
        "orientation_deg": transform.orientation_deg,
        "rotated_raster_width_px": transform.rotated_raster_width_px,
        "rotated_raster_height_px": transform.rotated_raster_height_px,
        "target_center_px": list(transform.target_center_px),
        "target_top_left_px": list(transform.target_top_left_px),
        "vehicle_depth_m": transform.vehicle_depth_m,
        "camera": _camera_metadata(transform.camera),
        "projection_model": transform.projection_model,
        "renderer_version": transform.renderer_version,
    }


def _safe_scene_name(scene_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", scene_id).strip("_")
    return name or "scene"


class ManifestSceneComposer:
    """File boundary for composing a SceneManifest with the existing renderer."""

    def compose(self, scene_manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        scene = load_scene_manifest(scene_manifest_path)
        rendered = renderer.render_scene(scene)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stem = _safe_scene_name(scene.scene_id or "scene")
        image_name = f"{stem}.png"
        manifest_name = f"{stem}.json"
        image_path = output_path / image_name
        rendered.composite_rgba.save(image_path, format="PNG")

        metadata: dict[str, Any] = {
            "schema_version": RENDERED_SCENE_SCHEMA_VERSION,
            "scene_id": scene.scene_id,
            "image": {
                "rgba_path": image_name,
                "width_px": rendered.composite_rgba.width,
                "height_px": rendered.composite_rgba.height,
            },
            "camera": _camera_metadata(scene.camera),
            "background": {"rgba": list(scene.background_rgba)},
            "objects": [
                {
                    "object_id": obj.object_id,
                    "source_instance_id": obj.source_instance_id,
                    "z_index": obj.z_index,
                    "original_input_index": obj.original_input_index,
                    "transform": _transform_metadata(obj.transform),
                }
                for obj in rendered.objects
            ],
            "renderer": {"implementation": "jibscan_scene", "version": RENDERER_VERSION},
        }
        (output_path / manifest_name).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return metadata


__all__ = [
    "ManifestSceneComposer",
    "RENDERED_SCENE_SCHEMA_VERSION",
    "SCENE_SCHEMA_VERSION",
    "load_scene_manifest",
]
