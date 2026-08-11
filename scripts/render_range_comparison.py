from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from jibscan_scene import CameraModel, SceneState, load_normalized_squid_instance, render_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=Path("examples/fixtures/synthetic_squid_001.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/range_comparison.png"))
    args = parser.parse_args()

    instance = load_normalized_squid_instance(args.fixture)
    camera = CameraModel(fx=900.0, fy=900.0, cx=512.0, cy=450.0, width_px=1024, height_px=900)
    targets = [("FAR 6.0 m", 6.0), ("SOURCE 3.0 m", 3.0), ("NEAR 1.0 m", 1.0)]

    panels = []
    for label, distance in targets:
        result = render_observation(instance, SceneState(camera=camera, target_distance_m=distance))
        panel = result.composite_rgba.copy()
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 260, 38), fill=(0, 0, 0, 180))
        draw.text((12, 11), f"{label}  scale={result.transform.scale:g}x", fill="white")
        panels.append(panel)

    comparison = Image.new("RGBA", (camera.width_px * len(panels), camera.height_px))
    for index, panel in enumerate(panels):
        comparison.alpha_composite(panel, dest=(index * camera.width_px, 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(args.output)


if __name__ == "__main__":
    main()
