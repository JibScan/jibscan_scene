from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from jibscan_scene import ManifestSceneComposer


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose a scene and write dense geometry sidecars")
    parser.add_argument("scene_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--background-policy", choices=["constant", "invalid"], default=None)
    parser.add_argument("--background-range-m", type=float, default=None)
    parser.add_argument("--diagnostic", type=Path, default=Path("artifacts/geometry_sidecar_comparison.png"))
    args = parser.parse_args()
    metadata = ManifestSceneComposer().compose_with_geometry(
        args.scene_manifest,
        args.output_dir,
        background_policy=args.background_policy,
        background_range_m=args.background_range_m,
    )
    sidecar = json.loads((args.output_dir / metadata["geometry_manifest_path"]).read_text(encoding="utf-8"))
    image = Image.open(args.output_dir / metadata["image"]["rgba_path"]).convert("RGB")
    range_map = np.load(args.output_dir / sidecar["maps"]["range_m"]["path"])
    visibility = np.load(args.output_dir / sidecar["maps"]["visibility"]["path"])
    validity = Image.open(args.output_dir / sidecar["maps"]["validity"]["path"]).convert("L")
    coverage = Image.open(args.output_dir / sidecar["maps"]["coverage"]["path"]).convert("L")
    finite = np.isfinite(range_map)
    range_panel = np.zeros((*range_map.shape, 3), dtype=np.uint8)
    if finite.any():
        lo, hi = range_map[finite].min(), range_map[finite].max()
        scaled = np.zeros_like(range_map, dtype=np.float32) if hi == lo else (range_map - lo) / (hi - lo)
        range_panel[finite] = np.repeat((255 * scaled[finite, None]).astype(np.uint8), 3, axis=1)
    visibility_panel = np.zeros((*visibility.shape, 3), dtype=np.uint8)
    for label in np.unique(visibility):
        if label:
            color = ((int(label) * 73) % 256, (int(label) * 151) % 256, (int(label) * 211) % 256)
            visibility_panel[visibility == label] = color
    panels = [image, Image.fromarray(range_panel), Image.fromarray(visibility_panel), validity, coverage]
    diagnostic = Image.new("RGB", (image.width * len(panels), image.height))
    for index, panel in enumerate(panels):
        diagnostic.paste(panel.convert("RGB"), (index * image.width, 0))
    args.diagnostic.parent.mkdir(parents=True, exist_ok=True)
    diagnostic.save(args.diagnostic, format="PNG")


if __name__ == "__main__":
    main()
