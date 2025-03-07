import argparse
import json
import os

import trimesh
import numpy as np

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any, Union


class Point(BaseModel):
    x: float
    y: float
    z: float
    size: float = Field(default=1.0, gt=0)
    name: Optional[str] = None

    @field_validator("name")
    def set_default_name(cls, name: Optional[str], info: Dict[str, Any]) -> str:
        """Set a default name based on coordinates if name is None"""
        if name is None:
            values = info.data
            x = values.get("x")
            y = values.get("y")
            z = values.get("z")
            if all(coord is not None for coord in (x, y, z)):
                return f"Point({x:.1f},{y:.1f},{z:.1f})"
        return name if name is not None else ""

    model_config = {
        "extra": "ignore",
        "validate_assignment": True,
    }


class Path(BaseModel):
    points: List[Point]
    name: Optional[str] = None
    color: str = Field(default="#0000FF")  # Default blue color
    thickness: float = Field(default=1.0, gt=0)

    @field_validator("name")
    def set_default_path_name(cls, name: Optional[str], info: Dict[str, Any]) -> str:
        """Set a default name for the path if none provided"""
        if name is None:
            # Count points for default name
            points = info.data.get("points", [])
            point_count = len(points)
            return f"Path-{point_count}pts"
        return name

    @model_validator(mode="after")
    def validate_path(self) -> "Path":
        """Validate that the path has at least one point"""
        if not self.points:
            raise ValueError("A path must contain at least one point")
        return self

    model_config = {
        "extra": "ignore",
        "validate_assignment": True,
    }


def load_from_file(
    file_path: str,
) -> Union[List[Point], List[Path], Dict[str, Union[List[Point], List[Path]]]]:
    """
    Read and deserialize Points and/or Paths from a JSON file.

    Args:
        file_path: Path to the JSON file containing data

    Returns:
        Dictionary containing lists of Points and/or Paths, or a single list

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        ValueError: If the JSON data doesn't match expected models
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read and parse the JSON file
    with open(file_path, "r") as f:
        data = json.load(f)

    result = {}

    # Handle different possible structures in the JSON
    if isinstance(data, dict):
        # Look for points key
        if "points" in data:
            result["points"] = [
                Point.model_validate(point_data) for point_data in data["points"]
            ]

        # Look for paths key
        if "paths" in data:
            result["paths"] = [
                Path.model_validate(path_data) for path_data in data["paths"]
            ]

        # If neither points nor paths found, try to parse the whole dict as a single path
        if not result and "points" in data and isinstance(data["points"], list):
            return Path.model_validate(data)

    elif isinstance(data, list):
        # Try to determine if this is a list of points or paths
        if data and all(isinstance(item, dict) for item in data):
            # If first item has 'points' key, assume list of paths
            if "points" in data[0] and isinstance(data[0]["points"], list):
                return [Path.model_validate(path_data) for path_data in data]
            # Otherwise assume list of points
            else:
                return [Point.model_validate(point_data) for point_data in data]

    # Return appropriate result
    if len(result) == 1:
        # If only one type found, return just that list
        return next(iter(result.values()))
    elif result:
        # If multiple types found, return the dictionary
        return result
    else:
        # If no valid data found
        raise ValueError("No valid points or paths found in the file")


def save_to_file(
    file_path: str,
    data: Union[Point, Path, List[Point], List[Path]],
    format_type: str = "auto",
):
    """
    Save Points and/or Paths to a JSON file.

    Args:
        file_path: Path where to save the JSON file
        data: Point, Path, list of Points, or list of Paths to save
        format_type: How to format the output ('auto', 'points', 'paths', 'mixed')
    """
    # Determine what we're saving and format appropriately
    output_data = {}

    # Single Point
    if isinstance(data, Point):
        if format_type in ("auto", "points"):
            output_data = [data.model_dump(exclude_none=True)]
        else:
            output_data = {"points": [data.model_dump(exclude_none=True)]}

    # Single Path
    elif isinstance(data, Path):
        if format_type in ("auto", "paths"):
            output_data = data.model_dump(exclude_none=True)
        else:
            output_data = {"paths": [data.model_dump(exclude_none=True)]}

    # List of objects
    elif isinstance(data, list) and data:
        # List of Points
        if all(isinstance(item, Point) for item in data):
            if format_type in ("auto", "points"):
                output_data = [item.model_dump(exclude_none=True) for item in data]
            else:
                output_data = {
                    "points": [item.model_dump(exclude_none=True) for item in data]
                }

        # List of Paths
        elif all(isinstance(item, Path) for item in data):
            if format_type in ("auto", "paths"):
                output_data = [item.model_dump(exclude_none=True) for item in data]
            else:
                output_data = {
                    "paths": [item.model_dump(exclude_none=True) for item in data]
                }

        # Mixed list - not supported
        else:
            raise ValueError(
                "Mixed lists of Points and Paths are not supported, use a dictionary instead"
            )

    # Dictionary of lists
    elif isinstance(data, dict):
        output_data = {}
        if "points" in data and isinstance(data["points"], list):
            output_data["points"] = [
                item.model_dump(exclude_none=True) if isinstance(item, Point) else item
                for item in data["points"]
            ]
        if "paths" in data and isinstance(data["paths"], list):
            output_data["paths"] = [
                item.model_dump(exclude_none=True) if isinstance(item, Path) else item
                for item in data["paths"]
            ]

    # Write to file
    with open(file_path, "w") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", help="Path to the mesh file")
    parser.add_argument(
        "--annotations", help="Annotations for the mesh (optional)", default=None
    )
    args = parser.parse_args()

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.load(args.mesh))

    points = []
    if args.annotations is not None:
        points = load_from_file(args.annotations)
        pc = trimesh.PointCloud([[point.x, point.y, point.z] for point in points])
        scene.add_geometry(pc)

    scene.show()
