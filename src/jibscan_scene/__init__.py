from .api import SceneComposer
from .contracts import (
    CameraModel,
    MultiObjectSceneState,
    NormalizedSquidInstance,
    RenderedObservation,
    RenderedScene,
    RenderedSceneObject,
    SceneObject,
    SceneState,
    TransformationMetadata,
)
from .fixtures import generate_synthetic_squid_rgba
from .loader import load_normalized_squid_instance
from .projection import pinhole_range_scale
from .renderer import render_observation, render_scene
from .scene_manifest import (
    GEOMETRY_SCHEMA_VERSION,
    ManifestSceneComposer,
    RENDERED_SCENE_SCHEMA_VERSION,
    RENDERED_SCENE_SCHEMA_VERSION_V2,
    SCENE_SCHEMA_VERSION,
    SCENE_SCHEMA_VERSION_V2,
    compose_with_geometry,
    load_scene_manifest,
    load_scene_manifest_v2,
)

__all__ = [
    "CameraModel",
    "MultiObjectSceneState",
    "NormalizedSquidInstance",
    "RenderedObservation",
    "RenderedScene",
    "RenderedSceneObject",
    "SceneComposer",
    "ManifestSceneComposer",
    "compose_with_geometry",
    "GEOMETRY_SCHEMA_VERSION",
    "SceneObject",
    "SceneState",
    "TransformationMetadata",
    "generate_synthetic_squid_rgba",
    "load_normalized_squid_instance",
    "load_scene_manifest",
    "load_scene_manifest_v2",
    "SCENE_SCHEMA_VERSION",
    "SCENE_SCHEMA_VERSION_V2",
    "RENDERED_SCENE_SCHEMA_VERSION",
    "RENDERED_SCENE_SCHEMA_VERSION_V2",
    "pinhole_range_scale",
    "render_observation",
    "render_scene",
]
