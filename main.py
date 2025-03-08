import argparse
import json
import trimesh
import typing
import random
import numpy as np
from models import Point, Path, AcousticPath


def load_json_data(file_path: str) -> dict:
    """Load and parse JSON data from file."""
    with open(file_path, "r") as f:
        return json.load(f)


def create_point_cloud(points: list[Point]) -> typing.Union[trimesh.PointCloud, None]:
    """Create trimesh PointCloud from list of Points."""
    if not points:
        return None
    coords = [p.to_array() for p in points]
    return trimesh.PointCloud(coords)


def create_path_geometry(path: Path) -> trimesh.path.Path3D:
    """Create trimesh Path3D from Path or AcousticPath."""
    vertices = [p.to_array() for p in path.points]

    # If no color specified, generate a random one
    if not hasattr(path, "color") or path.color is None:
        path_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    else:
        path_color = path.color

    # Convert hex color to RGBA with 50% transparency
    rgba = list(trimesh.visual.color.hex_to_rgba(path_color))
    rgba[3] = 127  # Set alpha to 127 for 50% transparency

    # Create a single entity for the whole path
    entities = [trimesh.path.entities.Line(points=np.arange(len(vertices)), color=rgba)]

    return trimesh.path.Path3D(entities=entities, vertices=vertices)


def main():
    """Main function to visualize 3D mesh with annotations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", help="Path to the mesh file")
    parser.add_argument(
        "--annotations", help="Annotations for the mesh (optional)", default=None
    )
    args = parser.parse_args()

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.load(args.mesh))

    if args.annotations:
        data = load_json_data(args.annotations)

        # Handle standalone points
        if "points" in data:
            points = [Point.from_dict(p) for p in data["points"]]
            pc = create_point_cloud(points)
            if pc:
                scene.add_geometry(pc)

        # Handle regular paths
        if "paths" in data:
            for path_data in data["paths"]:
                path = Path.from_dict(path_data)
                scene.add_geometry(create_path_geometry(path))

        # Handle acoustic paths
        if "acousticPaths" in data:
            for path_data in data["acousticPaths"]:
                path = AcousticPath.from_dict(path_data)
                scene.add_geometry(create_path_geometry(path))

    scene.show()


if __name__ == "__main__":
    main()
