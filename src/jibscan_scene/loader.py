from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import (
    EnvironmentSpec,
    GeometrySpec,
    ImageSpec,
    NormalizedSquidInstance,
    ProvenanceSpec,
    RangeSpec,
)


def _optional_tuple(data: dict[str, Any], key: str) -> tuple[int, ...] | None:
    value = data.get(key)
    if value is None:
        return None
    return tuple(int(x) for x in value)


def load_normalized_squid_instance(path: str | Path) -> NormalizedSquidInstance:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    image_data = payload["image"]
    geometry_data = payload["geometry"]
    environment_data = payload.get("environment", {})
    range_data = payload["range"]
    provenance_data = payload.get("provenance", {})
    known_top_level = {"instance_id", "image", "geometry", "environment", "range", "provenance", "restoration"}

    rgba_path = Path(image_data["rgba_path"])
    if not rgba_path.is_absolute():
        rgba_path = (manifest_path.parent / rgba_path).resolve()

    known_environment = {"vehicle_depth_m", "temperature_c", "salinity_psu", "oxygen"}
    known_provenance = {"source_sample_id", "annotation_id", "synthetic_fixture"}

    return NormalizedSquidInstance(
        instance_id=str(payload["instance_id"]),
        image=ImageSpec(
            rgba_path=rgba_path,
            width_px=image_data.get("width_px"),
            height_px=image_data.get("height_px"),
        ),
        geometry=GeometrySpec(
            projected_width_px=int(geometry_data["projected_width_px"]),
            projected_height_px=int(geometry_data["projected_height_px"]),
            source_bbox=_optional_tuple(geometry_data, "source_bbox"),
            source_image_size=_optional_tuple(geometry_data, "source_image_size"),
            roi_box=_optional_tuple(geometry_data, "roi_box"),
            crop_transform=geometry_data.get("crop_transform"),
        ),
        environment=EnvironmentSpec(
            vehicle_depth_m=environment_data.get("vehicle_depth_m"),
            temperature_c=environment_data.get("temperature_c"),
            salinity_psu=environment_data.get("salinity_psu"),
            oxygen=environment_data.get("oxygen"),
            extra={k: v for k, v in environment_data.items() if k not in known_environment},
        ),
        range=RangeSpec(
            subject_distance_m=float(range_data["subject_distance_m"]),
            method=str(range_data["method"]),
            confidence=range_data.get("confidence"),
        ),
        provenance=ProvenanceSpec(
            source_sample_id=provenance_data.get("source_sample_id"),
            annotation_id=provenance_data.get("annotation_id"),
            synthetic_fixture=bool(provenance_data.get("synthetic_fixture", False)),
            extra={k: v for k, v in provenance_data.items() if k not in known_provenance},
        ),
        restoration=dict(payload.get("restoration", {})),
        extra={k: v for k, v in payload.items() if k not in known_top_level},
    )
