from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from jibscan_scene import (
    CameraModel,
    MultiObjectSceneState,
    SceneObject,
    SceneState,
    load_normalized_squid_instance,
    render_scene,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=Path("examples/fixtures/synthetic_squid_001.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/multi_squid_comparison.png"))
    args = parser.parse_args()

    camera = CameraModel(fx=900.0, fy=900.0, cx=512.0, cy=450.0, width_px=1024, height_px=900)
    instance = load_normalized_squid_instance(args.fixture)

    # 3 squid occurrences with different centers, orientations, and explicit z-indices.
    # At least two overlap (squid_bg_overlap and squid_fg_overlap).
    # All 3 use the same source asset (satisfies >= 2 using same source asset).
    obj1 = SceneObject(
        object_id="squid_bg_overlap",
        instance=instance,
        state=SceneState(
            camera=camera,
            target_distance_m=4.0,
            center_px=(450.0, 430.0),
            orientation_deg=-25.0,
        ),
        z_index=0,
    )
    obj2 = SceneObject(
        object_id="squid_fg_overlap",
        instance=instance,
        state=SceneState(
            camera=camera,
            target_distance_m=2.5,
            center_px=(520.0, 480.0),
            orientation_deg=35.0,
        ),
        z_index=2,
    )
    obj3 = SceneObject(
        object_id="squid_mid_isolated",
        instance=instance,
        state=SceneState(
            camera=camera,
            target_distance_m=3.0,
            center_px=(780.0, 350.0),
            orientation_deg=90.0,
        ),
        z_index=1,
    )

    scene = MultiObjectSceneState(camera=camera, objects=(obj1, obj2, obj3))
    rendered = render_scene(scene)

    # Core composite remains completely clean (no baked annotations).
    # We create a 2-panel comparison image:
    # Left: Clean Rendered Composite
    # Right: Diagnostic Overlay (center crosshairs, object_id, z_index, orientation, distance)
    left_panel = rendered.composite_rgba.copy()
    left_draw = ImageDraw.Draw(left_panel)
    left_draw.rectangle((0, 0, 300, 38), fill=(0, 0, 0, 180))
    left_draw.text((12, 11), "Clean Multi-Squid Composite", fill="white")

    right_panel = rendered.composite_rgba.copy()
    right_draw = ImageDraw.Draw(right_panel)
    right_draw.rectangle((0, 0, 420, 38), fill=(0, 0, 0, 180))
    right_draw.text((12, 11), "Diagnostic Overlay (IDs, z-order, crosshairs)", fill="white")

    colors = [(255, 220, 0, 255), (0, 230, 255, 255), (255, 100, 200, 255)]
    for i, ro in enumerate(rendered.objects):
        cx, cy = ro.transform.target_center_px
        color = colors[i % len(colors)]

        # Center crosshair
        right_draw.line((cx - 12, cy, cx + 12, cy), fill=color, width=2)
        right_draw.line((cx, cy - 12, cx, cy + 12), fill=color, width=2)

        # Object label box
        label_text = f"ID: {ro.object_id}\nz={ro.z_index} | rot={ro.transform.orientation_deg:g} deg | dist={ro.transform.target_distance_m:g}m"
        tl_x, tl_y = int(cx + 15), int(cy - 20)
        right_draw.rectangle((tl_x - 4, tl_y - 4, tl_x + 220, tl_y + 36), fill=(0, 0, 0, 190), outline=color)
        right_draw.text((tl_x, tl_y), label_text, fill="white")

    comparison = Image.new("RGBA", (camera.width_px * 2, camera.height_px))
    comparison.alpha_composite(left_panel, dest=(0, 0))
    comparison.alpha_composite(right_panel, dest=(camera.width_px, 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(args.output)


if __name__ == "__main__":
    main()
