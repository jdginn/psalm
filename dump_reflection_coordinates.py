import argparse
import json
import os
import sys
from models import AcousticPath


def load_json_data(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def compute_itd(path: AcousticPath) -> float:
    shot_origin = path.shot.ray.origin
    nearest_pos = path.nearest_approach.position
    direct_distance = (
        (nearest_pos.x - shot_origin.x) ** 2
        + (nearest_pos.y - shot_origin.y) ** 2
        + (nearest_pos.z - shot_origin.z) ** 2
    ) ** 0.5
    return (path.distance - direct_distance) / 343 * 1000


def filter_paths(acoustic_paths, args):
    filtered = []
    for path in acoustic_paths:
        gain = path.gain
        itd_val = compute_itd(path)
        for reflection in path.reflections:
            # Gain filters
            if args.gain_gt is not None and not (gain > args.gain_gt):
                continue
            if args.gain_lt is not None and not (gain < args.gain_lt):
                continue
            # ITD filters
            if args.itd_gt is not None and not (itd_val > args.itd_gt):
                continue
            if args.itd_lt is not None and not (itd_val < args.itd_lt):
                continue
            # Coordinate filters
            if args.x_gt is not None and not (reflection.position.x > args.x_gt):
                continue
            if args.x_lt is not None and not (reflection.position.x < args.x_lt):
                continue
            if args.y_gt is not None and not (reflection.position.y > args.y_gt):
                continue
            if args.y_lt is not None and not (reflection.position.y < args.y_lt):
                continue
            if args.z_gt is not None and not (reflection.position.z > args.z_gt):
                continue
            if args.z_lt is not None and not (reflection.position.z < args.z_lt):
                continue
            filtered.append((reflection.position, gain, itd_val))
    return filtered


def bounding_rectangle(points, plane="xy"):
    coords = {
        "x": [getattr(p, "x") for p in points],
        "y": [getattr(p, "y") for p in points],
        "z": [getattr(p, "z") for p in points],
    }
    if not points:
        print("No points to bound.")
        return
    if plane == "xy":
        min_x, max_x = min(coords["x"]), max(coords["x"])
        min_y, max_y = min(coords["y"]), max(coords["y"])
        print(
            f"All final reflection points lie within rectangle: x=[{min_x:.3f}, {max_x:.3f}], y=[{min_y:.3f}, {max_y:.3f}]"
        )
    elif plane == "yz":
        min_y, max_y = min(coords["y"]), max(coords["y"])
        min_z, max_z = min(coords["z"]), max(coords["z"])
        print(
            f"All final reflection points lie within rectangle: y=[{min_y:.3f}, {max_y:.3f}], z=[{min_z:.3f}, {max_z:.3f}]"
        )
    elif plane == "xz":
        min_x, max_x = min(coords["x"]), max(coords["x"])
        min_z, max_z = min(coords["z"]), max(coords["z"])
        print(
            f"All final reflection points lie within rectangle: x=[{min_x:.3f}, {max_x:.3f}], z=[{min_z:.3f}, {max_z:.3f}]"
        )
    else:
        raise ValueError("Plane must be one of 'xy', 'yz', 'xz'")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze final reflection positions from acoustic paths in annotations.json."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common filters
    def add_common_args(sp):
        sp.add_argument("annotations_path", help="Path to annotations.json")
        sp.add_argument(
            "--gain-gt", type=float, help="Only show paths with gain > threshold (dB)"
        )
        sp.add_argument(
            "--gain-lt", type=float, help="Only show paths with gain < threshold (dB)"
        )
        sp.add_argument(
            "--itd-gt", type=float, help="Only show paths with ITD > threshold (ms)"
        )
        sp.add_argument(
            "--itd-lt", type=float, help="Only show paths with ITD < threshold (ms)"
        )
        sp.add_argument(
            "--x-gt",
            type=float,
            help="Only show paths where final reflection x > threshold",
        )
        sp.add_argument(
            "--x-lt",
            type=float,
            help="Only show paths where final reflection x < threshold",
        )
        sp.add_argument(
            "--y-gt",
            type=float,
            help="Only show paths where final reflection y > threshold",
        )
        sp.add_argument(
            "--y-lt",
            type=float,
            help="Only show paths where final reflection y < threshold",
        )
        sp.add_argument(
            "--z-gt",
            type=float,
            help="Only show paths where final reflection z > threshold",
        )
        sp.add_argument(
            "--z-lt",
            type=float,
            help="Only show paths where final reflection z < threshold",
        )

    # Dump command
    dump_parser = subparsers.add_parser(
        "dump", help="Dump the final reflection positions"
    )
    add_common_args(dump_parser)

    # Bound command
    bound_parser = subparsers.add_parser(
        "bound", help="Show bounding rectangle for final reflections"
    )
    add_common_args(bound_parser)
    bound_parser.add_argument(
        "--plane",
        choices=["xy", "yz", "xz"],
        default="xy",
        help="Choose plane for region summary",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.annotations_path):
        print(f"File not found: {args.annotations_path}", file=sys.stderr)
        sys.exit(1)
    data = load_json_data(args.annotations_path)
    acoustic_paths = [AcousticPath.from_dict(p) for p in data.get("acousticPaths", [])]
    filtered = filter_paths(acoustic_paths, args)

    if args.command == "dump":
        if not filtered:
            print("No matching paths found.")
            return
        print("x\ty\tz\tgain\titd")
        for pos, gain, itd_val in filtered:
            print(f"{pos.x:.3f}\t{pos.y:.3f}\t{pos.z:.3f}\t{gain:.2f}\t{itd_val:.2f}")

    elif args.command == "bound":
        bounding_rectangle(
            [pos for pos, _, _ in filtered], getattr(args, "plane", "xy")
        )


if __name__ == "__main__":
    main()
