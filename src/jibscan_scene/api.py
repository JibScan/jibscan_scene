from pathlib import Path
from typing import Protocol

class SceneComposer(Protocol):
    def compose(self, organism: dict, scene_condition: dict, output_dir: Path) -> dict:
        """Return a CleanSceneSample-compatible mapping."""
        ...
