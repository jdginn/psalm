import argparse
import json
import trimesh
import typing
import random
import math
import time
import numpy as np
from models import Point, Path, AcousticPath, Zone, Reflection
import sys
import multiprocessing
from multiprocessing import Process, Queue
from typing import Optional, List
import trimesh.viewer
import culling


def load_json_data(file_path: str) -> dict:
    """Load and parse JSON data from file."""
    with open(file_path, "r") as f:
        return json.load(f)


def create_point_cloud(points: list[Point]) -> typing.Union[trimesh.PointCloud, None]:
    """Create trimesh PointCloud from list of Points."""
    if not points:
        return None
    coords = [p.to_array() for p in points]
    colors = [trimesh.visual.color.hex_to_rgba(p.color) for p in points]
    return trimesh.PointCloud(vertices=coords, colors=colors)


def create_path_geometry(path: Path) -> trimesh.path.Path3D:
    """Create trimesh Path3D from Path or AcousticPath."""
    vertices = [p.to_array() for p in path.points]

    # If no color specified, generate a random one
    path_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

    # Convert hex color to RGBA with 50% transparency
    rgba = list(trimesh.visual.color.hex_to_rgba(path_color))
    rgba[3] = 127  # Set alpha to 127 for 50% transparency

    # Create a single entity for the whole path
    entities = [trimesh.path.entities.Line(points=np.arange(len(vertices)), color=rgba)]

    return trimesh.path.Path3D(entities=entities, vertices=vertices)


def create_acoustic_path_geometry(
    path: AcousticPath,
) -> trimesh.path.Path3D:
    """Create trimesh Path3D from Path or AcousticPath."""
    vertices = [p.position.to_array() for p in path.reflections]
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


def create_normal_paths(reflections: List[Reflection], length: float) -> List[Path]:
    normal_paths = []
    for reflection in reflections:
        start = reflection.position
        if (
            reflection.normal.x == 0
            and reflection.normal.y == 0
            and reflection.normal.z == 0
        ):
            print("No normal")
            continue
        end = Point(
            x=reflection.position.x + reflection.normal.x * length,
            y=reflection.position.y + reflection.normal.y * length,
            z=reflection.position.z + reflection.normal.z * length,
            # color=PastelRed,
        )
        normal_paths.append(Path(points=[reflection.position, end]))
    return normal_paths


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


def visualize_reflections(
    room_mesh: trimesh.Trimesh,
    acoustic_paths: list[AcousticPath],
    points: list[Point] = None,
    paths: list[Path] = None,
    zones: list[Zone] = None,
) -> None:
    """Interactive visualization of acoustic reflections with additional geometries."""

    acoustic_paths.sort(key=lambda x: x.distance)
    current_index = 0
    total_paths = len(acoustic_paths)

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

    if not acoustic_paths:
        scene.show(flags={"wireframe": True})
        return

    for path in acoustic_paths:
        scene.add_geometry(create_acoustic_path_geometry(path))
        scene.add_geometry(
            trimesh.PointCloud(
                [
                    path.nearest_approach.position.to_array(),
                    path.shot.ray.origin.to_array(),
                ]
            ),
        )
    scene.show(flags={"wireframe": True})


def visualize_reflections_step(
    room_mesh: trimesh.Trimesh,
    acoustic_paths: list[AcousticPath],
    points: list[Point] = None,
    paths: list[Path] = None,
    zones: list[Zone] = None,
) -> None:
    """Interactive visualization of acoustic reflections with additional geometries."""

    acoustic_paths.sort(key=lambda x: x.distance)
    current_index = 0
    total_paths = len(acoustic_paths)

    while True:
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

        if not acoustic_paths:
            scene.show(flags={"wireframe": True})
            return

        # Add only current acoustic path in step mode
        current_path = acoustic_paths[current_index]
        scene.add_geometry(create_acoustic_path_geometry(current_path))
        scene.add_geometry(
            trimesh.PointCloud(
                [
                    current_path.nearest_approach.position.to_array(),
                    current_path.shot.ray.origin.to_array(),
                ]
            ),
        )

        scene.add_geometry(
            [
                create_path_geometry(p)
                for p in create_normal_paths(current_path.reflections, 0.4)
            ]
        )

        # for reflection in current_path.reflections:
        #     p1 = reflection.position
        #     p2 = Point(
        #         x=reflection.position.x + 10 * reflection.normal.x,
        #         y=reflection.position.y + 10 * reflection.normal.y,
        #         z=reflection.position.z + 10 * reflection.normal.z,
        #     )
        #     print(f"p1:{p1} p2:{p2}\n")
        #     scene.add_geometry(create_path_geometry(Path([p1, p2])))

        print(f"\nViewing acoustic path {current_index + 1} of {total_paths}")
        print("\n")

        direct_dist = math.sqrt(
            ((zones[0].x - current_path.shot.ray.origin.x) ** 2)
            + ((zones[0].y - current_path.shot.ray.origin.y) ** 2)
            + ((zones[0].z - current_path.shot.ray.origin.z) ** 2)
        )

        print(f"direct_dist:{direct_dist}")
        print(f"path dist:{current_path.distance}")

        itd = (current_path.distance - direct_dist) / 343 * 1000
        print(f"ITD:{itd}ms")
        print(f"gain:{current_path.gain}dB")
        print(f"shot gain:{current_path.shot.gain}dB")
        print("\n")
        print("Press 'n' for next, 'p' for previous, 'q' to quit")

        try:
            # Create queue for key press communication
            key_queue = Queue()

            # Create and start visualization process
            viz_process = Process(target=show_scene_and_wait, args=(scene, key_queue))
            viz_process.start()

            # In step mode, handle navigation
            while True:
                key = input().lower()
                if key == "n":
                    current_index = (current_index + 1) % total_paths
                    break
                elif key == "p":
                    current_index = (current_index - 1) % total_paths
                    break
                elif key == "q":
                    if viz_process.is_alive():
                        viz_process.terminate()
                    viz_process.join()
                    return
                else:
                    print(
                        "Invalid input. Use 'n' for next, 'p' for previous, 'q' to quit"
                    )
                    continue

            # Clean up the visualization process
            if viz_process.is_alive():
                viz_process.terminate()
            viz_process.join()

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
    parser.add_argument(
        "--step",
        action="store_true",
        help="Step through reflections one at a time",
        default=False,
    )
    parser.add_argument(
        "--cull",
        type=float,
        help="Cull very similar paths from the render",
        default=0.0,
    )
    args = parser.parse_args()

    scene = trimesh.Scene()

    room_mesh = trimesh.load(args.mesh)
    room_mesh.fix_normals()
    n_faces = len(room_mesh.faces)
    face_colors = np.ones((n_faces, 4), dtype=np.uint8) * [255, 255, 255, 100]
    room_mesh.visual.face_colors = face_colors

    scene.add_geometry(room_mesh)

    points = []
    paths = []
    acoustic_paths = []
    zones = []

    if args.annotations:
        data = load_json_data(args.annotations)

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

    if args.cull > 0.0:
        acoustic_paths = culling.cull_acoustic_paths(acoustic_paths, args.cull)

    if args.step:
        visualize_reflections_step(room_mesh, acoustic_paths, points, paths, zones)
    else:
        visualize_reflections(room_mesh, acoustic_paths, points, paths, zones)


if __name__ == "__main__":
    # For macOS support
    multiprocessing.set_start_method("spawn")
    main()
