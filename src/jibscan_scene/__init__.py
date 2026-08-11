from .api import SceneComposer
from .contracts import CameraModel, NormalizedSquidInstance, RenderedObservation, SceneState
from .fixtures import generate_synthetic_squid_rgba
from .loader import load_normalized_squid_instance
from .projection import pinhole_range_scale
from .renderer import render_observation

__all__ = [
    "CameraModel",
    "NormalizedSquidInstance",
    "RenderedObservation",
    "SceneComposer",
    "SceneState",
    "generate_synthetic_squid_rgba",
    "load_normalized_squid_instance",
    "pinhole_range_scale",
    "render_observation",
]
