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

Assumptions: fixed intrinsics, fixed physical dimensions, fixed orientation, no perspective shape change beyond uniform scale, no occlusion change, and no water-optics transformation.

## Run

```bash
python -m pip install -e '.[dev]'
pytest
python scripts/generate_synthetic_fixture.py
python scripts/render_range_comparison.py
```

The comparison is generated as:

```text
FAR 6 m | SOURCE 3 m | NEAR 1 m
  0.5x  |      1x    |     3x
```

and written to `artifacts/range_comparison.png`.

## Provenance

Every render records source instance ID, source projected dimensions, source range, target range, scale, target projected dimensions, placement, vehicle depth, projection model, and renderer version. The source instance is immutable and is never overwritten by a rendered observation.

## Known limitations

- scale + translation only;
- no rotation or pose normalization;
- no physical squid-size inference;
- no range estimation;
- no occlusion model;
- no AquaPhys/environment rendering;
- canvas clipping is allowed if a requested target projection exceeds the camera frame.

Those are explicit extension points, not hidden behavior in v0.
