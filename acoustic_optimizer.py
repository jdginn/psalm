"""
Acoustic Optimization System
Created: 2025-03-14 by jdginn
Uses pre-trained validity model to optimize acoustic parameters
"""

import subprocess
from pathlib import Path
import numpy as np
from datetime import datetime, timezone
import json
import joblib
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import logging
import sys


class AcousticSimulator:
    def __init__(self, program_path: Path, timeout: int = 30):
        """
        Wrapper for acoustic simulation program

        Args:
            program_path: Path to simulator executable
            timeout: Maximum seconds to wait for simulation
        """
        self.program_path = Path(program_path)
        self.timeout = timeout
        self.results_history: List[Dict] = []

        # Setup logging
        self.logger = logging.getLogger("AcousticSimulator")
        self.logger.setLevel(logging.INFO)

    def run_simulation(self, params: Dict[str, float]) -> Tuple[float, bool]:
        """
        Run single acoustic simulation

        Args:
            params: Dictionary of parameter values

        Returns:
            (score, success) tuple
        """
        start_time = datetime.now(timezone.utc)

        print("params: ", params)

        # Create YAML config (similar to validity checker)
        config_str = f"""
input:
  mesh:
    path: "/Users/justinginn/repos/go-recording-studio/testdata/without_walls.3mf"

materials:
  from_file: "/Users/justinginn/repos/go-recording-studio/testdata/materials.yaml"

surface_assignments:
  inline:
    default: "brick"
    Floor: "wood"
    Front A: "gypsum"
    Front B: "gypsum"
    Back Diffuser: "diffuser"
    Ceiling Absorber: "rockwool_24cm"
    Secondary Ceiling Absorber L: "rockwool_24cm"
    Secondary Ceiling Absorber R: "rockwool_24cm"
    Street Absorber: "rockwool_24cm"
    Front Hall Absorber: "rockwool_24cm"
    Back Hall Absorber: "rockwool_24cm"
    Cutout Top: "rockwool_24cm"
    Door: "rockwool_12cm"
    L Speaker Gap: "rockwool_24cm"
    R Speaker Gap: "rockwool_24cm"
    Window A: "glass"
    Window B: "glass"
    left speaker wall: "gypsum"
    right speaker wall: "gypsum"

speaker:
  model: "MUM8"
  dimensions:
    x: 0.38
    y: 0.256
    z: 0.52
  offset:
    y: 0.096
    z: 0.412
  directivity:
    horizontal:
      0: 0
      30: -1
      40: -3
      50: -3
      60: -4
      70: -6
      80: -9
      90: -12
      120: -13
      150: -20
      180: -30
    vertical:
      0: 0
      30: 0
      60: -4
      70: -7
      80: -9
      100: -9
      120: -9
      150: -15

simulation:
  rfz_radius: 0.5
  shot_count: 10000
  shot_angle_range: 180
  order: 10
  gain_threshold_db: -15
  time_threshold_ms: 100

flags:
  skip_speaker_in_room_check: false
  skip_add_speaker_wall: false

listening_triangle:
  reference_position: [0, 2.37, 0.0]
  reference_normal: [1, 0, 0]
  distance_from_front: {params['distance_from_front']}
  distance_from_center: {params['distance_from_center']}
  source_height: {params['source_height']}
  listen_height: {params['listen_height']}
"""

        # Create temporary config file
        temp_config = Path(
            f"temp_config_{datetime.now(timezone.utc).strftime('%H%M%S_%f')}.yaml"
        )
        temp_config.write_text(config_str)

        try:
            # Run simulation
            result = subprocess.run(
                [str(self.program_path), str(temp_config)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            # Print program output
            if result.stdout:
                print("Program stdout:", result.stdout)
            if result.stderr:
                print("Program stderr:", result.stderr)

            success = result.returncode == 0
            if success:
                try:
                    score = float(result.stdout.strip())
                except ValueError:
                    self.logger.error(
                        f"Could not parse output as float: {result.stdout}"
                    )
                    score = 0.0
                    success = False
            else:
                score = 0.0

            # Record result
            self.results_history.append(
                {
                    "params": params,
                    "score": score,
                    "success": success,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "duration": (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds(),
                }
            )

            return score, success

        except subprocess.TimeoutExpired:
            self.logger.error(f"Simulation timed out after {self.timeout}s")
            return 0.0, False

        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            return 0.0, False

        finally:
            # Cleanup
            if temp_config.exists():
                temp_config.unlink()


class AcousticOptimizer:
    def __init__(
        self,
        simulator: AcousticSimulator,
        validity_model_path: Path,
        output_dir: Path,
        n_parallel: int = 4,
        validity_threshold: float = 0.8,
    ):
        """
        Acoustic parameter optimizer

        Args:
            simulator: AcousticSimulator instance
            validity_model_path: Path to saved validity model
            output_dir: Directory for saving results
            n_parallel: Number of parallel simulations
            validity_threshold: Minimum validity prediction confidence
        """
        self.simulator = simulator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_parallel = n_parallel
        self.validity_threshold = validity_threshold

        # Load validity model
        model_data = joblib.load(validity_model_path)
        self.validity_model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.bounds = model_data["bounds"]

        # Setup results tracking
        self.results: List[Dict] = []
        self.best_score = float("-inf")
        self.best_params = None

        # Setup logging
        self.logger = logging.getLogger("AcousticOptimizer")
        self.logger.setLevel(logging.INFO)

    def predict_validity(self, params: Dict[str, float]) -> float:
        """Get validity prediction confidence"""
        point_array = np.array([params[param] for param in self.bounds.keys()])
        point_scaled = self.scaler.transform([point_array])
        return self.validity_model.predict_proba(point_scaled)[0][1]

    def generate_exploration_points(self, n_points: int) -> List[Dict[str, float]]:
        """Generate diverse points predicted to be valid"""
        points = []
        attempts = 0
        max_attempts = n_points * 10

        while len(points) < n_points and attempts < max_attempts:
            # Generate random point
            params = {
                param: np.random.uniform(low, high)
                for param, (low, high) in self.bounds.items()
            }

            # Check validity
            if self.predict_validity(params) >= self.validity_threshold:
                points.append(params)

            attempts += 1

        return points

    # def generate_exploration_points(self, n_points: int) -> List[Dict[str, float]]:
    #     """Generate test points that we believe should work"""
    #     # Hard-coded test points that should be valid
    #     known_points = [
    #         {
    #             "distance_from_front": 0.6,  # Far from front
    #             "distance_from_center": 1.3,  # Close to center
    #             "source_height": 1.7,  # Low source
    #             "listen_height": 1.4,  # Reasonable listening height
    #         },
    #         {
    #             "distance_from_front": 0.7,
    #             "distance_from_center": 1.1,
    #             "source_height": 1.6,
    #             "listen_height": 1.2,
    #         },
    #         {
    #             "distance_from_front": 0.6,
    #             "distance_from_center": 1.4,
    #             "source_height": 1.5,
    #             "listen_height": 1.0,
    #         },
    #         # Add more known points if needed...
    #     ]
    #
    #     self.logger.info("Using hard-coded test points instead of random exploration")
    #     for point in known_points:
    #         validity_score = self.predict_validity(point)
    #         self.logger.info(f"Point {point} has validity score: {validity_score}")
    #
    #     # Return either the requested number of points or all known points
    #     return known_points[: min(n_points, len(known_points))]

    def optimize(self, n_iterations: int = 100, plot_progress: bool = True):
        """
        Run optimization process

        Args:
            n_iterations: Number of points to evaluate
            plot_progress: Whether to show progress plot
        """
        start_time = datetime.now(timezone.utc)
        self.logger.info(f"Starting optimization with {n_iterations} iterations")

        # Generate initial points
        points_to_evaluate = self.generate_exploration_points(n_iterations)
        self.logger.info(
            f"Generated {len(points_to_evaluate)} valid points to evaluate"
        )

        # Run simulations in parallel
        with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
            futures = []
            for params in points_to_evaluate:
                futures.append(executor.submit(self.simulator.run_simulation, params))

            # Process results as they complete
            for i, future in enumerate(futures):
                score, success = future.result()

                result = {
                    "params": points_to_evaluate[i],
                    "score": score,
                    "success": success,
                    "iteration": i,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                self.results.append(result)

                if score > self.best_score:
                    self.best_score = score
                    self.best_params = points_to_evaluate[i]
                    self.logger.info(f"New best score: {score}")

                if plot_progress and (i + 1) % 10 == 0:
                    self.plot_progress()

        # Final results
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.logger.info(f"Optimization completed in {duration:.1f}s")
        self.logger.info(f"Best score: {self.best_score}")
        self.logger.info(f"Best parameters: {self.best_params}")

        # Save results
        self.save_results()

    def plot_progress(self):
        """Plot optimization progress"""
        plt.figure(figsize=(10, 6))

        # Plot scores over iterations
        scores = [r["score"] for r in self.results]
        iterations = range(len(scores))

        plt.plot(iterations, scores, "b.", label="Scores")
        plt.plot(iterations, np.maximum.accumulate(scores), "r-", label="Best so far")

        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.title("Optimization Progress")
        plt.legend()

        # Save plot
        plt.savefig(self.output_dir / "optimization_progress.png")
        plt.close()

    def save_results(self):
        """Save optimization results and metadata"""
        results_file = (
            self.output_dir
            / f"results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )

        output_data = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user": "jdginn",
                "n_iterations": len(self.results),
                "best_score": float(self.best_score),
                "best_params": self.best_params,
            },
            "results": self.results,
        }

        with open(results_file, "w") as f:
            json.dump(output_data, f, indent=2)

        self.logger.info(f"Saved results to {results_file}")


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("acoustic_optimization.log"),
        ],
    )


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)

    if len(sys.argv) != 3:
        print(
            "Usage: python acoustic_optimizer.py <simulator_path> <validity_model_path>"
        )
        sys.exit(1)

    simulator_path = Path(sys.argv[1])
    validity_model_path = Path(sys.argv[2])

    # Verify paths exist
    if not simulator_path.exists():
        logger.error(f"Simulator not found: {simulator_path}")
        sys.exit(1)
    if not validity_model_path.exists():
        logger.error(f"Validity model not found: {validity_model_path}")
        sys.exit(1)

    # Setup optimization
    simulator = AcousticSimulator(simulator_path)
    optimizer = AcousticOptimizer(
        simulator=simulator,
        validity_model_path=validity_model_path,
        output_dir=Path("optimization_results"),
        n_parallel=4,
    )

    # Run optimization
    optimizer.optimize(n_iterations=10_000, plot_progress=True)
