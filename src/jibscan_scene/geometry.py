from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

from PIL import Image

from .contracts import RenderedScene


GEOMETRY_SCHEMA_VERSION = "jibscan.scene_geometry/v0.1"


def _write_npy(path: Path, values: list[float | int], shape: tuple[int, int], dtype: str) -> None:
    format_code = {"<f4": "f", "<i4": "i"}[dtype]
    header = f"{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape}, }}"
    header_bytes = header.encode("ascii")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"
    item = struct.Struct("<" + format_code)
    data = bytearray(len(values) * item.size)
    for index, value in enumerate(values):
        item.pack_into(data, index * item.size, value)
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + data)


def _sorted_objects(rendered: RenderedScene):
    return sorted(rendered.objects, key=lambda obj: (obj.z_index, obj.original_input_index))


def write_geometry_sidecar(
    rendered: RenderedScene,
    output_dir: Path,
    stem: str,
    *,
    scene_id: str | None,
    background_rgba: tuple[int, int, int, int],
    background_policy: str,
    background_range_m: float | None,
    rendered_manifest_name: str,
) -> dict[str, Any]:
    width, height = rendered.composite_rgba.size
    pixel_count = width * height
    if background_policy == "constant":
        if background_range_m is None or not math.isfinite(background_range_m) or background_range_m <= 0:
            raise ValueError("constant background policy requires background_range_m > 0")
        range_values = [float(background_range_m)] * pixel_count
        validity_values = bytearray([255] * pixel_count)
    elif background_policy == "invalid":
        range_values = [math.nan] * pixel_count
        validity_values = bytearray(pixel_count)
    else:
        raise ValueError("background_policy must be 'constant' or 'invalid'")
    visibility_values = [0] * pixel_count
    coverage_values = bytearray(pixel_count)
    object_alphas: dict[int, bytes] = {}

    labels: dict[str, Any] = {
        "0": {"label": "background", "object_id": None, "occurrence_index": None}
    }
    sorted_objects = _sorted_objects(rendered)
    for obj in rendered.objects:
        label_id = obj.original_input_index + 1
        labels[str(label_id)] = {
            "label": f"object:{obj.object_id}:occurrence:{obj.original_input_index}",
            "object_id": obj.object_id,
            "occurrence_index": obj.original_input_index,
            "source_instance_id": obj.source_instance_id,
            "target_distance_m": obj.transform.target_distance_m,
        }

    for obj in sorted_objects:
        alpha = obj.squid_layer_rgba.getchannel("A").tobytes()
        label_id = obj.original_input_index + 1
        object_alphas[label_id] = alpha
        for index, value in enumerate(alpha):
            if not value:
                continue
            visibility_values[index] = label_id
            range_values[index] = obj.transform.target_distance_m

    limitations: set[str] = set()
    mixed_partial_count = 0
    exact_count = 0
    for index in range(pixel_count):
        transmission = 1.0
        contributors: list[tuple[float, float]] = []
        for obj in reversed(sorted_objects):
            alpha = object_alphas[obj.original_input_index + 1][index] / 255.0
            if alpha and transmission:
                contributors.append((obj.transform.target_distance_m, alpha * transmission))
                transmission *= 1.0 - alpha
        coverage = round((1.0 - transmission) * 255.0)
        coverage_values[index] = coverage
        if visibility_values[index]:
            validity_values[index] = 255
        if transmission and background_policy == "invalid":
            limitations.add("background_range_invalid")
        ranges = {distance for distance, weight in contributors if weight > 0}
        if transmission and background_policy == "constant":
            ranges.add(float(background_range_m))
        if len(ranges) > 1 and coverage:
            mixed_partial_count += 1
            limitations.add("partial_alpha_multiple_ranges")
        if coverage and not (transmission and background_policy == "invalid") and len(ranges) <= 1:
            exact_count += 1

    range_path = output_dir / f"{stem}.range_m.npy"
    visibility_path = output_dir / f"{stem}.visibility.npy"
    validity_path = output_dir / f"{stem}.validity.png"
    coverage_path = output_dir / f"{stem}.coverage.png"
    _write_npy(range_path, range_values, (height, width), "<f4")
    _write_npy(visibility_path, visibility_values, (height, width), "<i4")
    Image.frombytes("L", (width, height), bytes(validity_values)).save(validity_path, format="PNG")
    Image.frombytes("L", (width, height), bytes(coverage_values)).save(coverage_path, format="PNG")

    sidecar_name = f"{stem}.geometry.json"
    sidecar: dict[str, Any] = {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "scene_id": scene_id,
        "rendered_scene_manifest": rendered_manifest_name,
        "image": {
            "format": "PNG",
            "dtype": "uint8",
            "color_space": "sRGB",
            "alpha_mode": "straight",
            "channel_range": [0, 255],
            "width_px": width,
            "height_px": height,
        },
        "background": {"policy": background_policy, "rgba": list(background_rgba)},
        "maps": {
            "range_m": {
                "path": range_path.name,
                "format": "npy",
                "dtype": "<f4",
                "shape": [height, width],
                "units": "m",
                "representation": "metric_range",
                "semantics": "target_distance_m of the frontmost visible occurrence",
                "invalid_value": "NaN",
            },
            "visibility": {
                "path": visibility_path.name,
                "format": "npy",
                "dtype": "<i4",
                "shape": [height, width],
                "semantics": "frontmost alpha-contributing occurrence label; zero is background",
            },
            "validity": {
                "path": validity_path.name,
                "format": "PNG",
                "dtype": "uint8",
                "shape": [height, width],
                "semantics": "255 where range contains a declared metric value, otherwise 0",
            },
            "coverage": {
                "path": coverage_path.name,
                "format": "PNG",
                "dtype": "uint8",
                "shape": [height, width],
                "semantics": "alpha union of rendered object layers",
            },
        },
        "visibility_labels": labels,
        "exactness": {
            "range_map_exact_pixels": exact_count,
            "partial_alpha_multi_range_pixels": mixed_partial_count,
            "range_map_limitation": (
                "At pixels with partial alpha from multiple distinct target ranges, the map stores "
                "the frontmost alpha-contributing occurrence; the physical range mixture is not "
                "recoverable from the composited RGBA image alone."
            ),
        },
        "representation": {
            "surface_model": "single_visible_surface",
            "alpha_policy": "frontmost_nonzero_alpha_range; coverage_is_alpha_union",
            "exact_for_current_scene": not limitations,
            "limitations": sorted(limitations),
        },
        "raster": {"width_px": width, "height_px": height},
        "background_range": {
            "range_mode": background_policy,
            **({"range_m": float(background_range_m)} if background_policy == "constant" else {}),
        },
        "path": sidecar_name,
    }
    (output_dir / sidecar_name).write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sidecar
