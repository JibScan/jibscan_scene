from __future__ import annotations


def pinhole_range_scale(source_distance_m: float, target_distance_m: float) -> float:
    """Return target/source projected-size ratio under the v0 pinhole assumptions.

    Assumptions: fixed camera intrinsics, fixed squid physical dimensions, fixed
    orientation, no occlusion change, and no water-optics effect. Under these
    assumptions projected size is proportional to 1 / subject distance, so the
    intrinsics and physical size cancel in the ratio.
    """
    if source_distance_m <= 0 or target_distance_m <= 0:
        raise ValueError("source and target subject distances must be > 0")
    return source_distance_m / target_distance_m
