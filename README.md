# jibscan-scene

Controlled geometry and scene compositor for JibScan.

## v0 scope

This first vertical slice answers one question only: can the same normalized squid asset be moved between explicit camera-to-subject distances while preserving anatomy, alpha, source state, and provenance?

```text
NormalizedSquidInstance
        +
    SceneState
        |
        v
fixed-intrinsics pinhole range scaling
        |
        v
RGBA squid layer + neutral composite + transformation metadata
```

Scene does **not** segment squid, infer biological state, estimate range, equate vehicle depth with range, run BioFlow, or implement AquaPhys. Real normalized squid and future BioFlow-generated squid are expected to enter through the same `NormalizedSquidInstance` boundary.

## Input contract

The v0 input is one independently traceable squid instance. The checked-in JSON Schema is `schemas/normalized_squid_instance.schema.json` and the synthetic example is `examples/fixtures/synthetic_squid_001.json`.

Two quantities are intentionally independent:

- `environment.vehicle_depth_m`: camera/vehicle depth in the water column.
- `range.subject_distance_m`: camera-to-squid distance.

Synthetic range values are test inputs, not measurements.

## Camera model and equation

For a pinhole camera,

```text
projected_size ∝ focal_length * physical_size / distance
```

For the same camera intrinsics, physical squid dimensions, and orientation:

```text
scale_target / scale_source = source_distance / target_distance
```

v0 therefore uses:

```text
scale = source_subject_distance_m / target_subject_distance_m
```

Assumptions: fixed intrinsics, fixed physical dimensions, no perspective shape change beyond uniform scale, no occlusion change, and no water-optics transformation. Scene Wave 4 adds only deterministic in-plane 2D image rotation; it does not add 3D pose or perspective changes.

Placement uses `SceneState.center_px` as the requested projected squid center in camera pixels. If omitted, it defaults to the camera principal point. Rasterization uses Python's built-in `round` (ties to even) for `round(center_x - width / 2)` and the corresponding y expression, so the rasterized center is within 0.5 px per axis of the requested center. Placement is never clamped or repositioned; out-of-frame pixels are clipped.

`SceneState.orientation_deg > 0` means counter-clockwise in the rendered image, using Pillow's `Image.rotate` convention. Rendering applies the exact order source RGBA raster -> resize to target projected dimensions -> expanded RGBA rotation with transparent fill -> placement by rotated-raster center -> alpha composite on the camera canvas. `target_projected_width_px` and `target_projected_height_px` remain the scaled projected geometry; `rotated_raster_width_px` and `rotated_raster_height_px` record the actual expanded raster footprint. Rotation never changes scale or projected dimensions, and zero degrees preserves the prior pixels and geometry.

## Run

```bash
python -m pip install -e '.[dev]'
pytest
python scripts/generate_synthetic_fixture.py
python scripts/render_range_comparison.py
python scripts/render_orientation_comparison.py
python scripts/compose_scene.py examples/scenes/synthetic_multi_squid.json --output-dir artifacts/example_scene
```

## File-based scene boundary

`ManifestSceneComposer().compose(scene_manifest_path, output_dir)` is the public
file-based entrypoint. A SceneManifest uses `jibscan.scene/v0.1`, keeps the
camera and background global, and references upstream normalized-instance JSON
manifests by path. Relative references resolve from the SceneManifest's
directory, never from the caller's working directory.

The composer writes deterministic `<safe-scene-id>.png` and
`<safe-scene-id>.json` files. The safe stem replaces every run of characters
outside `[A-Za-z0-9_-]` with `_`, strips leading/trailing `_`, and uses
`scene` when empty. Output paths in the rendered manifest are relative to the
output directory. The output contract is `jibscan.rendered_scene/v0.1` and
contains JSON-native camera, object ordering, z-index, and full transform
provenance only; it does not expose internal PIL images.

`ManifestSceneComposer().compose_with_geometry(...)` is the explicit Wave 8
adapter. It preserves the v0.1 call and writes a
`jibscan.rendered_scene/v0.2` manifest plus a `jibscan.scene_geometry/v0.1`
sidecar. The sidecar stores pixel-aligned float32 metric ranges in meters,
int32 occurrence labels, uint8 validity, and uint8 alpha-union coverage.
Visible object ranges use each occurrence's `target_distance_m`; `z_index`
only selects the composited visible occurrence and `vehicle_depth_m` never
enters a range map. Legacy v0.1 scenes default to an invalid background
range. A v0.2 scene may declare a positive constant background range with
`background.policy: "constant"` and `background.range_m`, or explicitly use
`background.policy: "invalid"`.

The single-visible-surface representation reports
`representation.exact_for_current_scene` and limitations. Partial-alpha
mixtures at different declared ranges are retained as an approximation and
are not treated as physical radiance.

`RenderedScene` still retains one full-canvas RGBA layer per object internally
for the existing renderer. This is a known O(objects × canvas) memory limit;
the file boundary exports only the final composite and provenance.

The comparison is generated as:

```text
FAR 6 m | SOURCE 3 m | NEAR 1 m
  0.5x  |      1x    |     3x
```

and written to `artifacts/range_comparison.png`.

The orientation comparison uses the same squid, distance, and center for panels at -45, 0, +45, and +90 degrees. It writes `artifacts/orientation_comparison.png`; its center marker is diagnostic-only and is not part of `RenderedObservation`.

## Provenance

Every render records source instance ID, source raster and projected dimensions, source range, target range, scale, target projected dimensions, orientation, rotated raster dimensions, requested `target_center_px`, post-rotation rasterized `target_top_left_px`, vehicle depth, camera, projection model, and renderer version. The source instance is immutable and is never overwritten by a rendered observation.

## Known limitations

- scale + translation + deterministic in-plane 2D rotation only;
- no 3D rotation or pose normalization;
- no physical squid-size inference;
- no range estimation;
- no occlusion model;
- no AquaPhys/environment rendering;
- canvas clipping is allowed if a requested target projection exceeds the camera frame.

Those are explicit extension points, not hidden behavior in v0.
