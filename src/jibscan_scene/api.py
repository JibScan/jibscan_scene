from pathlib import Path
from typing import Protocol


class SceneComposer(Protocol):
    """Stable outer adapter boundary retained from the repository scaffold."""

    def compose(self, organism: dict, scene_condition: dict, output_dir: Path) -> dict:
        """Return a rendered scene mapping without importing sibling JibScan modules."""
        ...
