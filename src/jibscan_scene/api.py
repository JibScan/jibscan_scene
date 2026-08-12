from pathlib import Path
from typing import Protocol


class SceneComposer(Protocol):
    """Legacy scaffold boundary retained unchanged for existing callers.

    ``ManifestSceneComposer`` is the separate file-based Wave 6 entrypoint;
    it intentionally does not replace this positional protocol.
    """

    def compose(self, organism: dict, scene_condition: dict, output_dir: Path) -> dict:
        """Return a rendered scene mapping without importing sibling JibScan modules."""
        ...
