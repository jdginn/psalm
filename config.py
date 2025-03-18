from dataclasses import dataclass, field
import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Material:
    absorption: float


@dataclass
class Materials:
    inline: Dict[str, Material] = field(default_factory=dict)
    from_file: str = ""

    def has_material(self, name: str) -> bool:
        return name in self.inline


@dataclass
class SurfaceAssignments:
    inline: Dict[str, str] = field(default_factory=dict)
    from_file: str = ""


@dataclass
class SpeakerDimensions:
    x: float  # Width in meters
    y: float  # Height in meters
    z: float  # Depth in meters


@dataclass
class SpeakerOffset:
    y: float  # Vertical offset in meters
    z: float  # Depth offset in meters


@dataclass
class Directivity:
    horizontal: Dict[float, float] = field(default_factory=dict)  # angle -> attenuation
    vertical: Dict[float, float] = field(default_factory=dict)  # angle -> attenuation


@dataclass
class Speaker:
    model: str
    dimensions: SpeakerDimensions
    offset: SpeakerOffset
    directivity: Directivity


@dataclass
class ListeningTriangle:
    reference_position: Tuple[float, float, float] = field(
        default_factory=lambda: (0, 0, 0)
    )
    reference_normal: Tuple[float, float, float] = field(
        default_factory=lambda: (0, 0, 0)
    )
    distance_from_front: float = 0
    distance_from_center: float = 0
    source_height: float = 0
    listen_height: float = 0


@dataclass
class Simulation:
    rfz_radius: float
    shot_count: int
    shot_angle_range: float
    order: int
    gain_threshold_db: float
    time_threshold_ms: float


@dataclass
class Flags:
    skip_speaker_in_room_check: bool = False
    skip_add_speaker_wall: bool = False


@dataclass
class CeilingPanelCenter:
    thickness: float
    height: float
    width: float
    xmin: float
    xmax: float


@dataclass
class CeilingPanelSides:
    thickness: float
    height: float
    width: float
    spacing: float
    xmin: float
    xmax: float


@dataclass
class CeilingPanels:
    center: Optional[CeilingPanelCenter] = None
    sides: Optional[CeilingPanelSides] = None


@dataclass
class Input:
    class Mesh:
        path: str = ""

    mesh: Mesh = field(default_factory=Mesh)


@dataclass
class Metadata:
    timestamp: str = ""  # YYYY-MM-DD HH:MM:SS in UTC
    git_commit: str = ""


@dataclass
class ExperimentConfig:
    metadata: Metadata = field(default_factory=Metadata)
    input: Input = field(default_factory=Input)
    materials: Materials = field(default_factory=Materials)
    surface_assignments: SurfaceAssignments = field(default_factory=SurfaceAssignments)
    speaker: Speaker = None
    listening_triangle: ListeningTriangle = field(default_factory=ListeningTriangle)
    simulation: Simulation = None
    flags: Flags = field(default_factory=Flags)
    ceiling_panels: CeilingPanels = field(default_factory=CeilingPanels)


class PathResolver:
    """Handles resolution of relative paths in the config."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def resolve_path(self, path: str) -> str:
        """Resolves a potentially relative path to an absolute path."""
        if os.path.isabs(path):
            return path
        return str(Path(self.base_dir) / path)

    def file_exists(self, path: str) -> bool:
        """Checks if a file exists and is readable."""
        abs_path = self.resolve_path(path)
        return os.path.exists(abs_path)


@dataclass
class LoadOptions:
    validate_immediately: bool = False
    resolve_paths: bool = False
    merge_files: bool = False


def load_from_file(path: str) -> ExperimentConfig:
    """Loads an ExperimentConfig from a YAML file."""

    try:
        with open(path, "r") as f:
            data = f.read()
    except Exception as e:
        raise RuntimeError(f"Error reading config file: {str(e)}")

    try:
        config_dict = yaml.safe_load(data)
        config = ExperimentConfig(**config_dict)
    except Exception as e:
        raise RuntimeError(f"Error parsing config file: {str(e)}")

    return config


def save_to_file(config: ExperimentConfig, path: str) -> None:
    """Saves an ExperimentConfig to a YAML file."""
    # try:
    #     collector = MetadataCollector()
    #     collector.populate_metadata(config)
    # except Exception as e:
    #     raise RuntimeError(f"creating metadata collector: {str(e)}")

    try:
        data = yaml.dump(config)
    except Exception as e:
        raise RuntimeError(f"marshaling config: {str(e)}")

    try:
        with open(path, "w") as f:
            f.write(data)
    except Exception as e:
        raise RuntimeError(f"writing config file: {str(e)}")


def resolve_paths(config: ExperimentConfig, resolver: PathResolver) -> None:
    """Resolves all relative paths in the config to absolute paths."""
    config.input.mesh.path = resolver.resolve_path(config.input.mesh.path)

    if config.materials.from_file:
        config.materials.from_file = resolver.resolve_path(config.materials.from_file)

    if config.surface_assignments.from_file:
        config.surface_assignments.from_file = resolver.resolve_path(
            config.surface_assignments.from_file
        )


# def load_and_merge(config: ExperimentConfig) -> None:
#     """Loads all external files and merges their contents."""
#     merge_materials(config.materials)
#     merge_surface_assignments(config.surface_assignments)
