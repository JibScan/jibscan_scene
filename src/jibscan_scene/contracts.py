from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageSpec:
    rgba_path: Path
    width_px: int | None = None
    height_px: int | None = None


@dataclass(frozen=True)
class GeometrySpec:
    projected_width_px: int
    projected_height_px: int
    source_bbox: tuple[int, int, int, int] | None = None
    source_image_size: tuple[int, int] | None = None
    roi_box: tuple[int, int, int, int] | None = None
    crop_transform: dict[str, Any] | None = None


@dataclass(frozen=True)
class EnvironmentSpec:
    vehicle_depth_m: float | None = None
    temperature_c: float | None = None
    salinity_psu: float | None = None
    oxygen: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RangeSpec:
    subject_distance_m: float
    method: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.subject_distance_m <= 0:
            raise ValueError("subject_distance_m must be > 0")


@dataclass(frozen=True)
class ProvenanceSpec:
    source_sample_id: str | None = None
    annotation_id: str | None = None
    synthetic_fixture: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedSquidInstance:
    instance_id: str
    image: ImageSpec
    geometry: GeometrySpec
    environment: EnvironmentSpec
    range: RangeSpec
    provenance: ProvenanceSpec
    restoration: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    width_px: int
    height_px: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx and fy must be > 0")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("camera dimensions must be > 0")


@dataclass(frozen=True)
class SceneState:
    camera: CameraModel
    target_distance_m: float
    center_px: tuple[float, float] | None = None
    background_rgba: tuple[int, int, int, int] = (48, 52, 58, 255)
    orientation_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.target_distance_m <= 0:
            raise ValueError("target_distance_m must be > 0")
        if not math.isfinite(self.orientation_deg):
            raise ValueError("orientation_deg must be finite")


@dataclass(frozen=True)
class TransformationMetadata:
    source_instance_id: str
    source_raster_width_px: int
    source_raster_height_px: int
    source_projected_width_px: int
    source_projected_height_px: int
    source_distance_m: float
    target_distance_m: float
    scale: float
    target_projected_width_px: int
    target_projected_height_px: int
    target_center_px: tuple[float, float]
    target_top_left_px: tuple[int, int]
    vehicle_depth_m: float | None
    camera: CameraModel
    projection_model: str = "pinhole_fixed_intrinsics_fixed_physical_size"
    renderer_version: str = "jibscan-scene/0.1.0"
    orientation_deg: float = 0.0
    rotated_raster_width_px: int = 0
    rotated_raster_height_px: int = 0


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    instance: NormalizedSquidInstance
    state: SceneState
    z_index: int | float = 0

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id or not self.object_id.strip():
            raise ValueError("object_id must be a non-empty string")
        if (
            not isinstance(self.z_index, (int, float))
            or isinstance(self.z_index, bool)
            or not math.isfinite(self.z_index)
        ):
            raise ValueError("z_index must be a finite number")


@dataclass(frozen=True)
class MultiObjectSceneState:
    camera: CameraModel
    objects: tuple[SceneObject, ...] = ()
    background_rgba: tuple[int, int, int, int] = (48, 52, 58, 255)
    scene_id: str | None = None
    manifest_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.objects, tuple):
            object.__setattr__(self, "objects", tuple(self.objects))


@dataclass(frozen=True)
class RenderedSceneObject:
    object_id: str
    source_instance_id: str
    z_index: int | float
    original_input_index: int
    transform: TransformationMetadata
    squid_layer_rgba: Any


@dataclass(frozen=True)
class RenderedScene:
    composite_rgba: Any
    objects: tuple[RenderedSceneObject, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.objects, tuple):
            object.__setattr__(self, "objects", tuple(self.objects))


@dataclass(frozen=True)
class RenderedObservation:
    squid_layer_rgba: Any
    composite_rgba: Any
    transform: TransformationMetadata
