import subprocess
import time
import json
import argparse
from datetime import datetime


def load_json_data(file_path: str) -> dict:
    """Load JSON data from file."""
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", help="Path to the mesh file")
    parser.add_argument("annotations", help="Path to the annotations file")
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between visualizations (default: 2.0)",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Starting path index (default: 0)"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="Ending path index (default: last path)"
    )
    args = parser.parse_args()

    # Load annotations to get total number of paths
    data = load_json_data(args.annotations)
    total_paths = len(data.get("acousticPaths", []))

    if total_paths == 0:
        print("No acoustic paths found in annotations file")
        return

    # Validate and adjust start/end indices
    start_idx = max(0, min(args.start, total_paths - 1))
    end_idx = total_paths if args.end is None else min(args.end + 1, total_paths)

    print(f"Starting visualization sequence:")
    print(f"Total paths: {total_paths}")
    print(f"Range: {start_idx} to {end_idx - 1}")
    print(f"Delay between paths: {args.delay} seconds")
    print("\nPress Ctrl+C to stop the sequence\n")

    try:
        for i in range(start_idx, end_idx):
            print(
                f"Showing path {i} at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )

            # Build and run the visualization command
            cmd = [
                "python",
                "main.py",
                args.mesh,
                f"--annotations={args.annotations}",
                f"--path={i}",
            ]

            # Run the visualization
            subprocess.run(cmd)

            # Only delay if this isn't the last path
            if i < end_idx - 1:
                time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\nVisualization sequence interrupted by user")

    print("\nVisualization sequence completed")


if __name__ == "__main__":
    main()
