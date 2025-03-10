import argparse
import json
import trimesh
import typing
import random
import time
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


import multiprocessing
from multiprocessing import Process, Queue
from typing import Optional
import trimesh.viewer


def show_scene_and_wait(scene: trimesh.Scene, key_queue: Queue) -> None:
    """Helper function to show scene and capture key press in separate process."""

    def key_callback(scene, callback_queue: Queue) -> Optional[bool]:
        """Callback that puts pressed key into queue."""
        key = scene.last_key
        if key == "q":
            callback_queue.put("q")
            return True  # This signals to close the window
        return False

    scene.show(flags={"wireframe": True})

    # # Create viewer with custom callback
    # viewer = trimesh.viewer.SceneViewer(
    #     scene=scene,
    #     callback=lambda s: key_callback(s, key_queue),
    #     flags={"background": True},  # This prevents the viewer from blocking
    # )
    #
    # # Start the viewer loop
    # viewer.run()


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

        # Add static geometries
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
        print("Press 'q' to advance to next path")
        print("Press Ctrl+C to exit program")

        try:
            # # Create queue for key press communication
            key_queue = Queue()

            # Create and start visualization process
            viz_process = Process(target=show_scene_and_wait, args=(scene, key_queue))
            viz_process.start()

            # # Block until we receive 'q' in the queue
            # key = key_queue.get()  # This blocks until a key is put in the queue
            # print("unblocked")

            # Get keyboard input from main process
            while True:
                key = input().lower()
                if key == "n":
                    current_index += 1
                    break
                elif key == "p":
                    current_index = max(0, current_index - 1)
                    break
                elif key == "q":
                    return
                else:
                    print(
                        "Invalid input. Use 'n' for next, 'p' for previous, 'q' to quit"
                    )
                    continue

            # Clean up the visualization process
            if viz_process.is_alive():
                print("Terminating visualization process")
                viz_process.terminate()
            print("Joining process")
            viz_process.join()

            # Move to next path
            if key == "n":
                current_index += 1

        except KeyboardInterrupt:
            print("\nExiting program...")
            if viz_process.is_alive():
                viz_process.terminate()
                viz_process.join()
            sys.exit(0)

        except Exception as e:
            print(f"\nError: {e}")
            if viz_process.is_alive():
                viz_process.terminate()
                viz_process.join()
            return

    print("\nCompleted viewing all paths")


def main():
    """Main function to visualize 3D mesh with annotations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", help="Path to the mesh file")
    parser.add_argument(
        "--annotations",
        type=str,
        help="Annotations for the mesh (optional)",
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

        points = []
        paths = []
        acoustic_paths = []
        zones = []

        # Handle standalone points
        if "points" in data:
            points = [Point.from_dict(p) for p in data["points"]]

        # Handle regular paths
        if "paths" in data:
            paths = [Path.from_dict(p) for p in data["paths"]]

        # Handle acoustic paths
        if "acousticPaths" in data:
            acoustic_paths = [AcousticPath.from_dict(p) for p in data["acousticPaths"]]

        # Add zones
        if "zones" in data:
            zones = [Zone.from_dict(p) for p in data["zones"]]

    visualize_reflections(room_mesh, acoustic_paths, points, paths, zones)

    scene.show()


if __name__ == "__main__":
    # For macOS support
    multiprocessing.set_start_method("spawn")
    main()
