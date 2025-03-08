import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Point:
    """A point in 3D space with optional display properties."""

    x: float
    y: float
    z: float
    size: float = 1.0
    name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Point":
        """Create Point from dictionary representation."""
        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            size=data.get("size", 1.0),
            name=data.get("name"),
        )

    def to_array(self) -> List[float]:
        """Convert point to [x, y, z] array."""
        return [self.x, self.y, self.z]


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


@dataclass
class Shot:
    """An acoustic ray with initial gain."""

    ray: Ray
    gain: float  # stored in dB

    @classmethod
    def from_dict(cls, data: dict) -> "Shot":
        """Create Shot from dictionary representation."""
        return cls(ray=Ray.from_dict(data["ray"]), gain=data["gain"])


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


@dataclass
class AcousticPath:
    """A path representing acoustic ray reflections with associated properties."""

    points: List[Point]
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
            points=[Point.from_dict(p) for p in data["points"]],
            shot=Shot.from_dict(data["shot"]),
            gain=data["gain"],
            distance=data["distance"],
            nearest_approach=NearestApproach.from_dict(data["nearestApproach"]),
            name=data.get("name"),
            color=data.get("color", "#FF0000"),
            thickness=data.get("thickness", 1.0),
        )
