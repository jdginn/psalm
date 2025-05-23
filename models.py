import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Point:
    """A point in 3D space with optional display properties."""

    x: float
    y: float
    z: float
    name: Optional[str] = None
    color: str = "#000000"  # Default color

    @classmethod
    def from_dict(cls, data: dict) -> "Point":
        """Create Point from dictionary representation."""
        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            name=data.get("name"),
            color=data.get("color", "#000000"),  # Default color
        )

    def to_array(self) -> List[float]:
        """Convert point to [x, y, z] array."""
        return [self.x, self.y, self.z]

    def to_dict(self) -> dict:
        """Convert Point to dictionary representation."""
        result = {"x": self.x, "y": self.y, "z": self.z}
        if self.name is not None:
            result["name"] = self.name
        if self.color != "#000000":  # Default color check
            result["color"] = self.color
        return result


@dataclass
class Vector:
    """A 3D vector representing a direction."""

    x: float
    y: float
    z: float

    @classmethod
    def from_dict(cls, data: dict) -> "Vector":
        """Create Vector from dictionary representation."""
        return cls(data["x"], data["y"], data["z"])

    def to_array(self) -> List[float]:
        """Convert vector to [x, y, z] array."""
        return [self.x, self.y, self.z]

    def to_dict(self) -> dict:
        """Convert Vector to dictionary representation."""
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class Ray:
    """A ray defined by origin point and direction vector."""

    origin: Point
    direction: Vector

    @classmethod
    def from_dict(cls, data: dict) -> "Ray":
        """Create Ray from dictionary representation."""
        return cls(
            origin=Point.from_dict(data["origin"]),
            direction=Vector.from_dict(data["direction"]),
        )

    def to_dict(self) -> dict:
        """Convert Ray to dictionary representation."""
        return {"origin": self.origin.to_dict(), "direction": self.direction.to_dict()}


@dataclass
class Shot:
    """An acoustic ray with initial gain."""

    ray: Ray
    gain: float  # stored in dB

    @classmethod
    def from_dict(cls, data: dict) -> "Shot":
        """Create Shot from dictionary representation."""
        return cls(ray=Ray.from_dict(data["ray"]), gain=data["gain"])

    def to_dict(self) -> dict:
        """Convert Shot to dictionary representation."""
        return {"ray": self.ray.to_dict(), "gain": self.gain}


@dataclass
class NearestApproach:
    """Information about closest point to listener."""

    position: Point
    distance: float

    @classmethod
    def from_dict(cls, data: dict) -> "NearestApproach":
        """Create NearestApproach from dictionary representation."""
        return cls(
            position=Point.from_dict(data["position"]), distance=data["distance"]
        )

    def to_dict(self) -> dict:
        """Convert NearestApproach to dictionary representation."""
        return {"position": self.position.to_dict(), "distance": self.distance}


@dataclass
class Path:
    """A sequence of 3D points with display properties."""

    points: List[Point]
    name: Optional[str] = None
    color: str = "#0000FF"
    thickness: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> "Path":
        """Create Path from dictionary representation."""
        return cls(
            points=[Point.from_dict(p) for p in data["points"]],
            name=data.get("name"),
            color=data.get("color", "#0000FF"),
            thickness=data.get("thickness", 1.0),
        )

    def to_dict(self) -> dict:
        """Convert Path to dictionary representation."""
        result = {"points": [p.to_dict() for p in self.points]}
        if self.name is not None:
            result["name"] = self.name
        if self.color != "#0000FF":
            result["color"] = self.color
        if self.thickness != 1.0:
            result["thickness"] = self.thickness
        return result


@dataclass
class Material:
    """Information about a material that can absorb or reflect sound."""

    absorption: float  # between 0 and 1.0

    def __post_init__(self):
        if not 0 <= self.absorption <= 1:
            raise ValueError("absorption must be between 0 and 1")

    @classmethod
    def from_dict(cls, data: dict) -> "Material":
        """Create Material from dictionary representation."""
        return cls(absorption=0)

    def to_dict(self) -> dict:
        """Convert Material to dictionary representation."""
        return {"absorption": self.absorption}


@dataclass
class Surface:
    """Information about a surface that can absorb or reflect sound."""

    material: Material
    name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Surface":
        """Create Surface from dictionary representation."""
        return cls(
            material=Material.from_dict(data["material"]),
            name=data.get("name"),
        )

    def to_dict(self) -> dict:
        """Convert Surface to dictionary representation."""
        result = {"material": self.material.to_dict()}
        if self.name is not None:
            result["name"] = self.name
        return result


@dataclass
class Reflection:
    """A reflection point with position and normal direction."""

    position: Point
    normal: Optional[Vector] = None
    surface: Optional[Surface] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Reflection":
        """Create Reflection from dictionary representation."""
        return cls(
            position=Point.from_dict(data["position"]),
            normal=Vector.from_dict(data["normal"]) if "normal" in data else None,
            surface=Surface.from_dict(data["surface"]) if "surface" in data else None,
        )

    def to_dict(self) -> dict:
        """Convert Reflection to dictionary representation."""
        result = {"position": self.position.to_dict()}
        if self.normal is not None:
            result["normal"] = self.normal.to_dict()
        if self.surface is not None:
            result["surface"] = self.surface.to_dict()
        return result


@dataclass
class AcousticPath:
    """A path representing acoustic ray reflections with associated properties."""

    reflections: List[Reflection]
    shot: Shot
    gain: float  # stored in dB
    distance: float
    nearest_approach: NearestApproach
    name: Optional[str] = None
    color: str = "#FF0000"
    thickness: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> "AcousticPath":
        """Create AcousticPath from dictionary representation."""
        return cls(
            reflections=[Reflection.from_dict(r) for r in data["reflections"]],
            shot=Shot.from_dict(data["shot"]),
            gain=data["gain"],
            distance=data["distance"],
            nearest_approach=NearestApproach.from_dict(data["nearestApproach"]),
            name=data.get("name"),
            color=data.get("color", "#FF0000"),
            thickness=data.get("thickness", 1.0),
        )

    def to_dict(self) -> dict:
        """Convert AcousticPath to dictionary representation."""
        result = {
            "reflections": [r.to_dict() for r in self.reflections],
            "shot": self.shot.to_dict(),
            "gain": self.gain,
            "distance": self.distance,
            "nearestApproach": self.nearest_approach.to_dict(),
        }
        if self.name is not None:
            result["name"] = self.name
        if self.color != "#FF0000":
            result["color"] = self.color
        if self.thickness != 1.0:
            result["thickness"] = self.thickness
        return result


@dataclass
class Zone:
    """A spherical region in 3D space."""

    x: float
    y: float
    z: float
    radius: float
    color: Optional[str] = None
    name: Optional[str] = None
    transparency: float = 0.8

    @classmethod
    def from_dict(cls, data: dict) -> "Zone":
        """Create Zone from dictionary representation."""
        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            radius=data["radius"],
            name=data.get("name"),
            color=data.get("color"),
            transparency=data.get("transparency", 0.8),
        )

    def to_dict(self) -> dict:
        """Convert Zone to dictionary representation."""
        result = {"x": self.x, "y": self.y, "z": self.z, "radius": self.radius}
        if self.name is not None:
            result["name"] = self.name
        if self.color is not None:
            result["color"] = self.color
        if self.transparency != 0.8:
            result["transparency"] = self.transparency
        return result


def serialize_scene(
    points: List[Point] = None,
    paths: List[Path] = None,
    acoustic_paths: List[AcousticPath] = None,
    zones: List[Zone] = None,
) -> dict:
    """
    Serialize a complete scene to a dictionary following the schema specification.

    Args:
        points: List of Point objects
        paths: List of Path objects
        acoustic_paths: List of AcousticPath objects
        zones: List of Zone objects

    Returns:
        Dictionary representation of the scene
    """
    result = {}

    if points:
        result["points"] = [p.to_dict() for p in points]
    if paths:
        result["paths"] = [p.to_dict() for p in paths]
    if acoustic_paths:
        result["acousticPaths"] = [p.to_dict() for p in acoustic_paths]
    if zones:
        result["zones"] = [z.to_dict() for z in zones]

    return result


@dataclass
class AnalysisResults:
    """Analysis results for binaural analysis."""

    ITD: Optional[float] = None
    ITD_2: Optional[float] = None
    avg_gain_5ms: Optional[float] = None
    listen_pos_dist: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisResults":
        """Create AnalysisResults from dictionary representation."""
        return cls(
            ITD=data.get("ITD"),
            ITD_2=data.get("ITD_2"),
            avg_gain_5ms=data.get("avg_gain_5ms"),
            listen_pos_dist=data.get("listen_pos_dist"),
        )

    def to_dict(self) -> dict:
        """Convert AnalysisResults to dictionary representation."""
        if self.ITD is not None:
            result = {"ITD": self.ITD}
        if self.ITD_2 is not None:
            result["ITD_2"] = self.ITD_2
        if self.avg_gain_5ms is not None:
            result["avg_gain_5ms"] = self.avg_gain_5ms
        if self.listen_pos_dist is not None:
            result["listen_pos_dist"] = self.listen_pos_dist
        return result


@dataclass
class SummaryResults:
    """Binaural analysis results including validation errors and status."""

    status: str
    validation_errors: Optional[List[str]] = None
    results: Optional[AnalysisResults] = None

    @classmethod
    def from_dict(cls, data: dict) -> "SummaryResults":
        """Create BinauralAnalysisResults from dictionary representation."""
        return cls(
            status=data["status"],
            validation_errors=data.get("validation_errors"),
            results=AnalysisResults.from_dict(data["results"]),
        )

    def to_dict(self) -> dict:
        """Convert BinauralAnalysisResults to dictionary representation."""
        result = {
            "status": self.status,
            "results": self.results.to_dict(),
        }
        if self.validation_errors is not None:
            result["validation_errors"] = self.validation_errors
        return result


def serialize_analysis_results(results: SummaryResults) -> dict:
    """
    Serialize binaural analysis results to a dictionary following the schema specification.

    Args:
        results: BinauralAnalysisResults object

    Returns:
        Dictionary representation of the analysis results
    """
    return results.to_dict()


def deserialize_analysis_results(data: dict) -> SummaryResults:
    """
    Deserialize binaural analysis results from a dictionary following the schema specification.

    Args:
        data: Dictionary representation of the analysis results

    Returns:
        BinauralAnalysisResults object
    """
    return SummaryResults.from_dict(data)
