from __future__ import annotations

import argparse
from pathlib import Path

from jibscan_scene import ManifestSceneComposer


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose a SceneManifest into a PNG and manifest")
    parser.add_argument("scene_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ManifestSceneComposer().compose(args.scene_manifest, args.output_dir)


if __name__ == "__main__":
    main()
