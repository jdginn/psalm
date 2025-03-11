from typing import List, Dict
import numpy as np
from collections import defaultdict

from models import AcousticPath


def cull_acoustic_paths(
    paths: List[AcousticPath], distance_threshold: float = 0.2
) -> List[AcousticPath]:
    """
    Cull acoustic paths by grouping those with similar final reflection points and keeping only the highest-gain path from each group.

    Args:
        paths: List of AcousticPath objects to process
        distance_threshold: Maximum distance between final reflection points to be considered part of the same group (in meters)

    Returns:
        List of culled AcousticPath objects, containing only the highest-gain path from each group
    """
    if not paths:
        return []

    # Create buckets to group paths
    buckets: Dict[int, List[AcousticPath]] = defaultdict(list)
    processed_indices = set()

    # For each path
    for i, path1 in enumerate(paths):
        if i in processed_indices:
            continue

        # Get the final reflection point
        point1 = np.array(path1.points[-1].to_array())

        # Create a new bucket for this path
        current_bucket = [path1]
        processed_indices.add(i)

        # Compare with all other unprocessed paths
        for j, path2 in enumerate(paths[i + 1 :], start=i + 1):
            if j in processed_indices:
                continue

            # Get the final reflection point of the comparison path
            point2 = np.array(path2.points[-1].to_array())

            # Calculate distance between final reflection points
            distance = np.linalg.norm(point1 - point2)

            # If within threshold, add to current bucket
            if distance <= distance_threshold:
                current_bucket.append(path2)
                processed_indices.add(j)

        # Store bucket using first path's index as key
        buckets[i] = current_bucket

    # Select highest-gain path from each bucket
    culled_paths = []
    for bucket in buckets.values():
        # Sort bucket by gain (descending) and take the first path
        highest_gain_path = max(bucket, key=lambda p: p.gain)
        culled_paths.append(highest_gain_path)

    return culled_paths
