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

# Add scikit-optimize imports
from skopt import Optimizer
from skopt.space import Real
from skopt.utils import use_named_args


from simulation_runner import SimulationRunner
from name_generator import generate_experiment_id


class AcousticOptimizer:
    def __init__(
        self,
        simulator: SimulationRunner,
        output_dir: Path,
        n_parallel: int = 4,
        bounds: Dict[str, Tuple[float, float]] = {},
    ):
        """
        Acoustic parameter optimizer

        Args:
            simulator: AcousticSimulator instance
            validity_model_path: Path to saved validity model
            output_dir: Directory for saving results
            n_parallel: Number of parallel simulations
            validity_threshold: Minimum validity prediction confidence
            additional_params: Parameters that were not in the validity training but should be included in this training
        """
        self.simulator = simulator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_parallel = n_parallel
        self.bounds = bounds

        # Setup Bayesian optimization
        self.space = [
            Real(low, high, name=param)
            for param, (low, high) in self.bounds.items()
        ]
        self.param_names = list(self.bounds.keys())
        
        # Initialize Bayesian optimizer with custom base estimator
        self.optimizer = Optimizer(
            dimensions=self.space,
            base_estimator="GP",  # Gaussian Process
            acq_func="EI",       # Expected Improvement
            acq_optimizer="auto",
            n_initial_points=10,  # Number of random points before using surrogate model
            random_state=42
        )

        # Setup results tracking
        self.results: List[Dict] = []
        self.best_score = float("-inf")
        self.best_params = None
        self.best_experiment_name = ""

        # Setup logging
        self.logger = logging.getLogger("AcousticOptimizer")
        self.logger.setLevel(logging.INFO)


    def generate_exploration_points(self, n_points: int) -> List[Dict[str, float]]:
        """Generate points using Bayesian optimization strategy"""
        points = []
        attempts = 0
        max_attempts = n_points * 100
        rejected_count = 0

        while len(points) < n_points and attempts < max_attempts:
            # Ask the optimizer for a batch of points
            candidate_points = self.optimizer.ask(n_points=min(n_points - len(points), self.n_parallel))
            
            for point in candidate_points:
                # Convert to dictionary
                params = dict(zip(self.param_names, point))
                
                points.append(params)


                attempts += 1
                if len(points) >= n_points:
                    break

            if attempts >= max_attempts:
                self.logger.warning(
                    f"Hit max attempts ({max_attempts}). Found {len(points)} valid points. "
                    f"Rejected {rejected_count} points. "
                )

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

        for iteration in range(0, n_iterations, self.n_parallel):
            # Generate batch of points
            batch_size = min(self.n_parallel, n_iterations - iteration)
            points_to_evaluate = self.generate_exploration_points(batch_size)
            
            # Run simulations in parallel
            with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
                futures = []
                for params in points_to_evaluate:
                    futures.append(
                        executor.submit(
                            self.simulator.run_simulation,
                            generate_experiment_id(),
                            params
                        )
                    )

                # Process results as they complete
                for i, future in enumerate(futures):
                    point = points_to_evaluate[i]
                    name, simulation_result, success = future.result()
                    if simulation_result["status"] == "success":
                        score = simulation_result["results"]["ITD"]
                        print(f"score: {score}")
                    else:
                        print(f"Experiment failed")
                        score = -1
                    

                    # Tell the optimizer about the result
                    self.optimizer.tell([point[param] for param in self.param_names], -score)  # Negative because we're maximizing

                    store_result = {
                        "params": points_to_evaluate[i],
                        "score": score,
                        "success": success,
                        "iteration": iteration + i,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    self.results.append(store_result)

                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = points_to_evaluate[i]
                        self.best_experiment_name = name
                        self.logger.info(f"New best score: {score}")

                    if plot_progress and (iteration + i + 1) % 10 == 0:
                        self.plot_progress()

        # Final results
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.logger.info(f"Optimization completed in {duration:.1f}s")
        self.logger.info(f"Best score: {self.best_score}")
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best experiment: {self.best_experiment_name}")

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
            "Usage: python acoustic_optimizer.py <simulator_path> <path_to_save_results> <config_path>"
        )
        sys.exit(1)

    program_path = Path(sys.argv[1])
    base_directory = Path(sys.argv[2])
    config_path = Path(sys.argv[3])

    # Verify paths exist
    if not program_path.exists():
        logger.error(f"Simulator not found: {program_path}")
        sys.exit(1)

    # Setup optimization
    simulator = SimulationRunner(program_path=program_path, base_directory=base_directory, config_path=config_path)
    optimizer = AcousticOptimizer(
        simulator=simulator,
        output_dir=Path("optimization_results"),
        n_parallel=16,
        bounds = {
            "distance_from_front": (0.3, 0.8),
            "distance_from_center": (0.8, 3.0),
            "source_height": (0.8, 2.2),
            "ceiling_center_height": (2.2, 2.7),
            "ceiling_center_width": (1.0, 3.5),
            "ceiling_center_xmin": (0.3, 1.0),
            "ceiling_center_xmax": (1.1, 3.0),
            "wall_absorbers_Street_A": (0.5, 1.0),
            "wall_absorbers_Hall_B": (0.5, 1.0),
            "wall_absorbers_Street_A": (0.5, 1.0),
            "wall_absorbers_Door_Side_A": (0.5, 1.0),
            "wall_absorbers_Hall_E": (0.5, 1.0),
            "wall_absorbers_Street_D": (0.5, 1.0),
            "wall_absorbers_Street_B": (0.5, 1.0),
            "wall_absorbers_Door_Side_B": (0.5, 1.0),
            "wall_absorbers_Entry_Back": (0.5, 1.0),
            "wall_absorbers_Street_C": (0.5, 1.0),
            "wall_absorbers_Street_E": (0.5, 1.0),
            "wall_absorbers_Hall_A": (0.5, 1.0),
            "wall_absorbers_Entry_Front": (0.5, 1.0),
            "wall_absorbers_Door": (0.5, 1.0),
            "wall_absorbers_Back_A": (0.5, 1.0),
            "wall_absorbers_Back_B": (0.5, 1.0),
        }
    )

    # Run optimization
    optimizer.optimize(n_iterations=1000, plot_progress=True)
