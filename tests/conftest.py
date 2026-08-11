from __future__ import annotations

import json
from pathlib import Path

import pytest

from jibscan_scene.fixtures import generate_synthetic_squid_rgba


@pytest.fixture
def synthetic_manifest(tmp_path: Path) -> Path:
    rgba_path = generate_synthetic_squid_rgba(tmp_path / "synthetic_squid_001.png")
    payload = json.loads(Path("examples/fixtures/synthetic_squid_001.json").read_text(encoding="utf-8"))
    payload["image"]["rgba_path"] = rgba_path.name
    manifest = tmp_path / "synthetic_squid_001.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest
