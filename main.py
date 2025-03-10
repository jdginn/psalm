import argparse
import json
import trimesh
import typing
import random
import numpy as np
from models import Point, Path, AcousticPath, Zone


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


def create_path_geometry(path: typing.Union[Path, AcousticPath]) -> trimesh.path.Path3D:
    """Create trimesh Path3D from Path or AcousticPath."""
    vertices = [p.to_array() for p in path.points]
    if isinstance(path, AcousticPath):
        vertices.append(path.nearest_approach.position.to_array())

    # If no color specified, generate a random one
    path_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

    # Convert hex color to RGBA with 50% transparency
    rgba = list(trimesh.visual.color.hex_to_rgba(path_color))
    rgba[3] = 127  # Set alpha to 127 for 50% transparency

    # Create a single entity for the whole path
    entities = [trimesh.path.entities.Line(points=np.arange(len(vertices)), color=rgba)]

    return trimesh.path.Path3D(entities=entities, vertices=vertices)


def create_zone_geometry(zone: Zone) -> trimesh.Trimesh:
    """Create trimesh Trimesh from Zone."""
    # Create a unit sphere and scale/translate it
    sphere = trimesh.primitives.Sphere(radius=1.0)

    # Scale by radius
    sphere.apply_scale(zone.radius)

    # Translate to center position
    translation = np.array([zone.x, zone.y, zone.z])
    sphere.apply_translation(translation)

    # If no color specified, generate a random one
    if not hasattr(zone, "color") or zone.color is None:
        random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    else:
        random_color = zone.color

    # Get transparency value (default to 0.8 or 80%)
    transparency = getattr(zone, "transparency", 0.8)

    # Convert hex color to RGBA with specified transparency
    rgba = list(trimesh.visual.color.hex_to_rgba(random_color))
    rgba[3] = int((1 - transparency) * 255)  # Convert transparency to alpha

    # Apply the color to the sphere
    sphere.visual.face_colors = rgba

    return sphere


def main():
    """Main function to visualize 3D mesh with annotations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", help="Path to the mesh file")
    parser.add_argument(
        "--annotations", help="Annotations for the mesh (optional)", default=None
    )
    parser.add_argument(
        "--path",
        type=int,
        help="Index of the acoustic path to visualize (0-based). If not provided, shows all paths.",
        default=None,
    )
    args = parser.parse_args()

    scene = trimesh.Scene()

    room_mesh = trimesh.load(args.mesh)
    room_mesh.fix_normals()
    n_faces = len(room_mesh.faces)
    print("Number of faces:", n_faces)
    face_colors = np.ones((n_faces, 4), dtype=np.uint8) * [255, 255, 255, 100]
    room_mesh.visual.face_colors = face_colors

    # Verify the colors were set
    print(f"Updated face colors shape: {room_mesh.visual.face_colors.shape}")
    print(f"Sample of face colors: {room_mesh.visual.face_colors[0]}")

    scene.add_geometry(room_mesh)

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
            acoustic_paths = [AcousticPath.from_dict(p) for p in data["acousticPaths"]]

            if args.path is not None:
                # Validate path index
                if 0 <= args.path < len(acoustic_paths):
                    # Show only the specified path
                    path = acoustic_paths[args.path]
                    print(f"\nShowing acoustic path {args.path}")
                    print(f"Number of points: {len(path.points)}")
                    print(f"Nearest approach: {path.nearest_approach.position}")
                    scene.add_geometry(create_path_geometry(path))
                    scene.add_geometry(
                        trimesh.PointCloud([path.nearest_approach.position.to_array()]),
                    )
                else:
                    print(
                        f"\nError: Path index {args.path} is out of range. "
                        f"Must be between 0 and {len(acoustic_paths)-1}"
                    )
                    return
            else:
                # Show all paths
                print(f"\nShowing all {len(acoustic_paths)} acoustic paths")
                for path in acoustic_paths:
                    scene.add_geometry(create_path_geometry(path))
                    scene.add_geometry(
                        trimesh.PointCloud([path.nearest_approach.position.to_array()]),
                    )

        # Add zones
        if "zones" in data:
            for i, zone_data in enumerate(data["zones"]):
                zone = Zone.from_dict(zone_data)
                zone_geom = create_zone_geometry(zone)
                scene.add_geometry(zone_geom, node_name=f"zone_{i}")

    scene.show()


if __name__ == "__main__":
    main()
