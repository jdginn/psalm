import argparse
import json
import os
import matplotlib.pyplot as plt


def load_json_data(file_path: str) -> dict:
    """Load and parse JSON data from file."""
    with open(file_path, "r") as f:
        return json.load(f)


def compute_itd(path_dict):
    """Compute ITD (ms) for an acoustic path dict (parsed from annotations.json)."""
    origin = path_dict["shot"]["ray"]["origin"]
    nearest = path_dict["nearestApproach"]["position"]
    direct_distance = (
        (nearest["x"] - origin["x"]) ** 2
        + (nearest["y"] - origin["y"]) ** 2
        + (nearest["z"] - origin["z"]) ** 2
    ) ** 0.5
    # Speed of sound = 343 m/s
    itd_ms = (path_dict["distance"] - direct_distance) / 343 * 1000
    return itd_ms


def plot_impulse_response(annotations_path, gain_floor_db=-50, save_path=None):
    """Parse annotations.json and plot impulse response bar chart, bars from floor to gain."""
    data = load_json_data(annotations_path)
    acoustic_paths = [p for p in data.get("acousticPaths", [])]

    times_ms = []
    gains_db = []
    for path in acoustic_paths:
        itd = compute_itd(path)
        gain = path.get("gain", None)
        if gain is not None and gain >= gain_floor_db:
            times_ms.append(itd)
            gains_db.append(gain)

    # Bar heights should be (gain - gain_floor_db), bottoms at gain_floor_db
    bar_heights = [g - gain_floor_db for g in gains_db]

    plt.figure(figsize=(12, 6))
    plt.bar(
        times_ms, bar_heights, width=0.1, color="blue", alpha=0.5, bottom=gain_floor_db
    )
    plt.xlabel("Arrival Time (ms)")
    plt.ylabel("Gain (dB)")
    plt.title("Impulse Response (Acoustic Paths)")
    plt.ylim(gain_floor_db, 0)
    plt.grid(True, which="both", axis="both", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations_json", help="Path to annotations.json file")
    parser.add_argument(
        "--gain-floor",
        type=float,
        default=-25,
        help="Minimum gain (dB) to plot (Y axis lower limit)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save chart to file instead of displaying (specify path)",
    )
    args = parser.parse_args()
    plot_impulse_response(
        args.annotations_json, gain_floor_db=args.gain_floor, save_path=args.output
    )


if __name__ == "__main__":
    main()
