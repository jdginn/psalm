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


def visualize_reflections(
    room_mesh: trimesh.Trimesh,
    acoustic_paths: list[AcousticPath],
    points: list[Point] = None,
    paths: list[Path] = None,
    zones: list[Zone] = None,
) -> None:
    """Interactive visualization of acoustic reflections with additional geometries."""
    current_index = 0
    total_paths = len(acoustic_paths)

    while 0 <= current_index < total_paths:
        # Create fresh scene for this reflection
        scene = trimesh.Scene()

        # Add room mesh
        scene.add_geometry(room_mesh)

        # Add static geometries (points, regular paths, zones)
        if points:
            pc = create_point_cloud(points)
            if pc:
                scene.add_geometry(pc)

        if paths:
            for path in paths:
                scene.add_geometry(create_path_geometry(path))

        if zones:
            for i, zone in enumerate(zones):
                scene.add_geometry(create_zone_geometry(zone), node_name=f"zone_{i}")

        # Add current acoustic path
        current_path = acoustic_paths[current_index]
        scene.add_geometry(create_path_geometry(current_path))
        scene.add_geometry(
            trimesh.PointCloud([current_path.nearest_approach.position.to_array()]),
        )

        print(f"\nViewing acoustic path {current_index + 1} of {total_paths}")
        print("Controls:")
        print("  'q'         : show next path")
        print("  'esc'       : exit step-through mode")
        print("  'w'         : toggle wireframe mode")

        try:
            # Show scene - this blocks until user closes window
            scene.show()

            # After window closes with 'q', move to next path
            current_index += 1

        except (KeyboardInterrupt, SystemExit):
            # Handle escape key or window close button
            print("\nExiting step-through mode...")
            return

    # After showing all paths or if user exits
    print("\nFinished showing all acoustic paths")


def main():
    """Main function to visualize 3D mesh with annotations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", help="Path to the mesh file")
    parser.add_argument(
        "--annotations", help="Annotations for the mesh (optional)", default=None
    )
    parser.add_argument(
        "--step-through",
        action="store_true",
        help="Enable step-by-step visualization of acoustic paths",
    )
    args = parser.parse_args()

    # Load and prepare room mesh
    room_mesh = trimesh.load(args.mesh)
    room_mesh.fix_normals()
    n_faces = len(room_mesh.faces)
    print("Number of faces:", n_faces)
    face_colors = np.ones((n_faces, 4), dtype=np.uint8) * [255, 255, 255, 100]
    room_mesh.visual.face_colors = face_colors

    if args.annotations:
        data = load_json_data(args.annotations)

        # Parse all geometries
        points = [Point.from_dict(p) for p in data.get("points", [])]
        paths = [Path.from_dict(p) for p in data.get("paths", [])]
        acoustic_paths = [
            AcousticPath.from_dict(p) for p in data.get("acousticPaths", [])
        ]
        zones = [Zone.from_dict(z) for z in data.get("zones", [])]

        if args.step_through and acoustic_paths:
            # Use interactive visualization mode
            print("\nEntering step-through mode...")
            visualize_reflections(
                room_mesh=room_mesh,
                acoustic_paths=acoustic_paths,
                points=points,
                paths=paths,
                zones=zones,
            )

            # After step-through mode ends, show all paths
            print("\nShowing all paths together...")
            scene = trimesh.Scene()
            scene.add_geometry(room_mesh)

            # Add all geometries
            if points:
                pc = create_point_cloud(points)
                if pc:
                    scene.add_geometry(pc)

            for path in paths:
                scene.add_geometry(create_path_geometry(path))

            for path in acoustic_paths:
                scene.add_geometry(create_path_geometry(path))
                scene.add_geometry(
                    trimesh.PointCloud([path.nearest_approach.position.to_array()]),
                )

            for i, zone in enumerate(zones):
                scene.add_geometry(create_zone_geometry(zone), node_name=f"zone_{i}")

            scene.show()
        else:
            # Use standard visualization mode
            scene = trimesh.Scene()
            scene.add_geometry(room_mesh)

            # Add all geometries
            if points:
                pc = create_point_cloud(points)
                if pc:
                    scene.add_geometry(pc)

            for path in paths:
                scene.add_geometry(create_path_geometry(path))

            for path in acoustic_paths:
                scene.add_geometry(create_path_geometry(path))
                scene.add_geometry(
                    trimesh.PointCloud([path.nearest_approach.position.to_array()]),
                )

            for i, zone in enumerate(zones):
                scene.add_geometry(create_zone_geometry(zone), node_name=f"zone_{i}")

            scene.show()
    else:
        # Just show the room mesh if no annotations
        scene = trimesh.Scene()
        scene.add_geometry(room_mesh)
        scene.show()


if __name__ == "__main__":
    main()
