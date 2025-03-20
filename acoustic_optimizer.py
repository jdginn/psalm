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

from simulation_runner import SimulationRunner
from name_generator import generate_experiment_id


class AcousticOptimizer:
    def __init__(
        self,
        simulator: SimulationRunner,
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
                futures.append(executor.submit(self.simulator.run_simulation, generate_experiment_id(), params))

            # Process results as they complete
            for i, future in enumerate(futures):
                simulation_result, success = future.result()
                print(f"simulation_result: {simulation_result}")
                score = 0.0
                if simulation_result["status"] == "success":
                    score = simulation_result["results"]["ITD"]


                store_result = {
                    "params": points_to_evaluate[i],
                    "score": score,
                    "success": success,
                    "iteration": i,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                self.results.append(store_result)

                if score > self.best_score:
                    self.best_score = score
                    self.best_params = points_to_evaluate[i]
                    self.logger.info(f"New best score: {simulation_result}")

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

    if len(sys.argv) != 5:
        print(
            "Usage: python acoustic_optimizer.py <simulator_path> <path_to_save_results> <config_path> <validity_model_path>"
        )
        sys.exit(1)

    program_path = Path(sys.argv[1])
    base_directory = Path(sys.argv[2])
    config_path = Path(sys.argv[3])
    validity_model_path = Path(sys.argv[4])

    # Verify paths exist
    if not program_path.exists():
        logger.error(f"Simulator not found: {program_path}")
        sys.exit(1)
    if not validity_model_path.exists():
        logger.error(f"Validity model not found: {validity_model_path}")
        sys.exit(1)

    # Setup optimization
    simulator = SimulationRunner(program_path=program_path, base_directory=base_directory, config_path=config_path)
    optimizer = AcousticOptimizer(
        simulator=simulator,
        validity_model_path=validity_model_path,
        output_dir=Path("optimization_results"),
        n_parallel=4,
    )

    # Run optimization
    optimizer.optimize(n_iterations=100, plot_progress=True)
