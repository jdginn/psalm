import argparse
import json
import trimesh
import typing
import random
import math
import os.path
import time
import numpy as np
from models import Point, Path, AcousticPath, Zone, Reflection, SummaryResults, Ray
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
    path_color = getattr(path, "color", "#{:06x}".format(random.randint(0, 0xFFFFFF)))

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


def create_normal_paths_from_reflections(
    reflections: List[Reflection], length: float
) -> List[Path]:
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


def create_normal_path(ray: Ray, length: float):
    return create_path_geometry(
        Path(
            points=[
                ray.origin,
                Point(
                    x=ray.origin.x + ray.direction.x * length,
                    y=ray.origin.y + ray.direction.y * length,
                    z=ray.origin.z + ray.direction.z * length,
                ),
            ]
        )
    )


def direct_distance(path: AcousticPath) -> float:
    return math.sqrt(
        ((path.nearest_approach.position.x - path.shot.ray.origin.x) ** 2)
        + ((path.nearest_approach.position.y - path.shot.ray.origin.y) ** 2)
        + ((path.nearest_approach.position.z - path.shot.ray.origin.z) ** 2)
    )


def itd(path: AcousticPath) -> float:
    return (path.distance - direct_distance(path)) / 343 * 1000


def filter_by_source_name(
    acoustic_paths: list[AcousticPath],
    name: str,
) -> list[AcousticPath]:
    """
    Filter paths to only include those with the specified source name.

    Args:
        acoustic_paths: List of acoustic paths to filter
        name: Name of the source to filter by
    """
    return [path for path in acoustic_paths if path.shot.source_name == name]


def filter_by_minimum_gain(
    acoustic_paths: list[AcousticPath], min_gain_db: float
) -> list[AcousticPath]:
    """
    Filter paths to only include those with gain above the minimum threshold.

    Args:
        acoustic_paths: List of acoustic paths to filter
        min_gain_db: Minimum gain threshold in dB

    Returns:
        List of acoustic paths with gain >= min_gain_db
    """
    return [path for path in acoustic_paths if path.gain >= min_gain_db]


def filter_by_maximum_itd(
    acoustic_paths: list[AcousticPath],
    max_itd_ms: float,
) -> list[AcousticPath]:
    """
    Filter paths to only include those that arrive before the maximum time threshold.

    Args:
        acoustic_paths: List of acoustic paths to filter
        max_time_ms: Maximum time threshold in milliseconds

    Returns:
        List of acoustic paths that arrive before max_time_ms
    """
    filtered_paths = []

    for path in acoustic_paths:
        # Calculate direct path time

        if itd(path) <= max_itd_ms:
            filtered_paths.append(path)

    return filtered_paths


def filter_by_walls(
    acoustic_paths: list[AcousticPath], wall_names: list[str]
) -> list[AcousticPath]:
    """
    Filter paths to only include those that interact with any of the specified walls.
    Uses OR logic - path is included if it touches ANY of the specified walls.
    """
    filtered_paths = []

    for path in acoustic_paths:
        # Check if any reflection in the path involves any of the specified walls
        if any(
            any(reflection.surface.name == wall_name for wall_name in wall_names)
            for reflection in path.reflections
        ):
            filtered_paths.append(path)

    return filtered_paths


def exclude_by_walls(
    acoustic_paths: list[AcousticPath], wall_names: list[str]
) -> list[AcousticPath]:
    """
    Filter out paths that interact with any of the specified walls.
    Path is excluded if it touches ANY of the specified walls.
    """
    filtered_paths = []

    for path in acoustic_paths:
        # Keep path only if it doesn't touch any of the excluded walls
        if not any(
            any(reflection.surface.name == wall_name for wall_name in wall_names)
            for reflection in path.reflections
        ):
            filtered_paths.append(path)

    return filtered_paths


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


def plot_final_reflection_positions(
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
        scene.add_geometry(
            trimesh.PointCloud(
                vertices=[
                    path.reflections[-1].position.to_array(),
                ],
                colors=[
                    [255, 0, 0, 255],
                ],
            )
        )
    scene.show(flags={"wireframe": True})


def plot_reflection_positions(
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
        for i, ref in enumerate(path.reflections):
            if i == len(path.reflections) - 1:
                scene.add_geometry(
                    trimesh.PointCloud(
                        vertices=[
                            ref.position.to_array(),
                        ],
                        colors=[
                            [255, 0, 0, 255],
                        ],
                    )
                )
            else:
                scene.add_geometry(
                    trimesh.PointCloud(
                        vertices=[
                            ref.position.to_array(),
                        ],
                        colors=[
                            [0, 0, 255, 255],
                        ],
                    )
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

        scene.add_geometry(create_normal_path(current_path.shot.sourceNormal, 1))

        scene.add_geometry(
            [
                create_path_geometry(p)
                for p in create_normal_paths_from_reflections(
                    current_path.reflections, 0.4
                )
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
            (
                (
                    current_path.nearest_approach.position.x
                    - current_path.shot.ray.origin.x
                )
                ** 2
            )
            + (
                (
                    current_path.nearest_approach.position.y
                    - current_path.shot.ray.origin.y
                )
                ** 2
            )
            + (
                (
                    current_path.nearest_approach.position.z
                    - current_path.shot.ray.origin.z
                )
                ** 2
            )
        )

        print(f"direct_dist:{direct_dist}")
        print(f"path dist:{current_path.distance}")

        itd = (current_path.distance - direct_dist) / 343 * 1000
        print(f"ITD:{itd}ms")
        print(
            f"gain contribution from reflections:{current_path.gain_from_reflections}dB"
        )
        print(f"gain contribution from distance:{current_path.gain_from_distance}dB")
        print(f"total gain:{current_path.gain}dB")
        print(f"shot gain:{current_path.shot.gain}dB")
        print(f"yaw:{current_path.shot.yaw}deg")
        print(f"pitch:{current_path.shot.pitch}deg")
        print(f"{len(current_path.reflections)} reflections")
        print(f"last reflection from {current_path.reflections[-1].surface.name}")
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


def score_reflection_match(
    path: AcousticPath,
    target_itd: float,
    target_gain: Optional[float] = None,
    itd_window: float = 2.0,
) -> Optional[float]:
    """
    Score how well a reflection matches the target ITD and gain.
    Returns None if the reflection is outside the acceptable window.
    Lower score is better.
    """
    # Calculate ITD for this path
    direct_dist = math.sqrt(
        ((path.nearest_approach.position.x - path.shot.ray.origin.x) ** 2)
        + ((path.nearest_approach.position.y - path.shot.ray.origin.y) ** 2)
        + ((path.nearest_approach.position.z - path.shot.ray.origin.z) ** 2)
    )
    path_itd = (path.distance - direct_dist) / 343 * 1000

    # Check if within ITD window
    itd_diff = abs(path_itd - target_itd)
    if itd_diff > itd_window:
        return None

    # If no gain specified, just use ITD difference
    if target_gain is None:
        return itd_diff**2

    # Check if within gain window (±9dB)
    gain_diff = abs(path.gain - target_gain)
    if gain_diff > 9.0:
        return None

    # Combined score using weighted sum
    return (itd_diff**2) + (gain_diff**2 / 81)


def find_matching_reflections(
    acoustic_paths: List[AcousticPath],
    target_itd: float,
    target_gain: Optional[float] = None,
    max_results: int = 1,
) -> List[AcousticPath]:
    """
    Find reflections that best match the target ITD and optional gain.
    Returns up to max_results paths, sorted by best match first.
    """
    # Score all paths and filter out None scores (outside window)
    scored_paths = [
        (path, score_reflection_match(path, target_itd, target_gain))
        for path in acoustic_paths
    ]
    valid_paths = [(path, score) for path, score in scored_paths if score is not None]

    # Sort by score (lower is better)
    valid_paths.sort(key=lambda x: x[1])

    # Return the best matching paths
    return [path for path, _ in valid_paths[:max_results]]


def visualize_matching_reflections(
    room_mesh: trimesh.Trimesh,
    acoustic_paths: list[AcousticPath],
    points: list[Point] = None,
    paths: list[Path] = None,
    zones: list[Zone] = None,
) -> None:
    """Visualize the matching reflections (similar to step mode but without navigation)."""
    if not acoustic_paths:
        print("No matching reflections found!")
        return

    for i, current_path in enumerate(acoustic_paths):
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

        # Add the matching acoustic path
        scene.add_geometry(create_acoustic_path_geometry(current_path))
        scene.add_geometry(
            trimesh.PointCloud(
                [
                    current_path.nearest_approach.position.to_array(),
                    current_path.shot.ray.origin.to_array(),
                ]
            ),
        )

        # Add normal vectors for reflections
        scene.add_geometry(
            [
                create_path_geometry(p)
                for p in create_normal_paths_from_reflections(
                    current_path.reflections, 0.4
                )
            ]
        )

        # Calculate and display path information
        direct_dist = math.sqrt(
            (
                (
                    current_path.nearest_approach.position.x
                    - current_path.shot.ray.origin.x
                )
                ** 2
            )
            + (
                (
                    current_path.nearest_approach.position.y
                    - current_path.shot.ray.origin.y
                )
                ** 2
            )
            + (
                (
                    current_path.nearest_approach.position.z
                    - current_path.shot.ray.origin.z
                )
                ** 2
            )
        )

        itd = (current_path.distance - direct_dist) / 343 * 1000

        print(f"\nMatching reflection {i + 1} of {len(acoustic_paths)}")
        print(f"direct_dist: {direct_dist:.2f}")
        print(f"path dist: {current_path.distance:.2f}")
        print(f"ITD: {itd:.2f}ms")
        print(f"gain: {current_path.gain:.2f}dB")
        print(f"shot gain: {current_path.shot.gain:.2f}dB")
        print(f"{len(current_path.reflections)} reflections")
        print(f"last reflection from {current_path.reflections[-1].surface.name}")
        print("\nPress any key to continue, 'q' to quit")

        try:
            # Create queue for key press communication
            key_queue = Queue()

            # Create and start visualization process
            viz_process = Process(target=show_scene_and_wait, args=(scene, key_queue))
            viz_process.start()

            # Wait for input
            key = input().lower()
            if key == "q":
                if viz_process.is_alive():
                    viz_process.terminate()
                viz_process.join()
                return

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
    parser.add_argument("path", help="Path to the experiment")
    parser.add_argument(
        "--filter-source",
        type=str,
        help="Filter reflections to only include those above the specified gain in",
    )
    parser.add_argument(
        "--filter-gain",
        type=float,
        help="Filter reflections to only include those above the specified gain in",
        default=-100,
    )
    parser.add_argument(
        "--filter-itd",
        type=float,
        help="Filter reflections to only include those below the specified itd in ms",
        default=150,
    )
    parser.add_argument(
        "--filter-walls",
        nargs="+",
        help="Filter reflections to only include those that interact with the specified walls (OR logic)",
    )
    parser.add_argument(
        "--exclude-walls",
        nargs="+",
        help="Exclude reflections that interact with any of the specified walls",
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
    parser.add_argument(
        "--points",
        action="store_true",
        help="Show the location of the final reflection in each arrival",
        default=False,
    )
    parser.add_argument(
        "--search-itd",
        type=float,
        help="Search for reflection with specific ITD (in ms)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        help="Target gain for ITD search (in dB)",
    )
    args = parser.parse_args()

    scene = trimesh.Scene()

    room_mesh = trimesh.load(os.path.join(args.path, "room.stl"))
    room_mesh.fix_normals()
    n_faces = len(room_mesh.faces)
    face_colors = np.ones((n_faces, 4), dtype=np.uint8) * [255, 255, 255, 100]
    room_mesh.visual.face_colors = face_colors

    scene.add_geometry(room_mesh)

    points = []
    paths = []
    acoustic_paths = []
    zones = []

    if os.path.exists(os.path.join(args.path, "summary.json")):
        data = load_json_data(os.path.join(args.path, "summary.json"))
        results = SummaryResults.from_dict(data)

    if os.path.exists(os.path.join(args.path, "annotations.json")):
        data = load_json_data(os.path.join(args.path, "annotations.json"))

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

    if args.filter_source:
        acoustic_paths = filter_by_source_name(acoustic_paths, args.filter_source)

    if args.filter_walls:
        acoustic_paths = filter_by_walls(acoustic_paths, args.filter_walls)

    if args.exclude_walls:
        acoustic_paths = exclude_by_walls(acoustic_paths, args.exclude_walls)

    if args.filter_gain:
        acoustic_paths = filter_by_minimum_gain(acoustic_paths, args.filter_gain)

    if args.filter_itd:
        acoustic_paths = filter_by_maximum_itd(acoustic_paths, args.filter_itd)

    if args.search_itd is not None:
        # Find and visualize matching reflections
        matching_paths = find_matching_reflections(
            acoustic_paths,
            args.search_itd,
            args.gain,
            max_results=1,  # Currently hardcoded to 1, but easily changeable
        )
        if args.points:
            plot_reflection_positions(room_mesh, matching_paths, points, paths, zones)
            return
        else:
            visualize_matching_reflections(
                room_mesh, matching_paths, points, paths, zones
            )
        return

    if args.cull > 0.0:
        acoustic_paths = culling.cull_acoustic_paths(acoustic_paths, args.cull)

    if args.points:
        plot_reflection_positions(room_mesh, acoustic_paths, points, paths, zones)
        return
    if args.step:
        visualize_reflections_step(room_mesh, acoustic_paths, points, paths, zones)
        return
    visualize_reflections(room_mesh, acoustic_paths, points, paths, zones)


if __name__ == "__main__":
    # For macOS support
    multiprocessing.set_start_method("spawn")
    main()
