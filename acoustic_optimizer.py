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

        # Add convergence tracking
        self.score_history = []
        self.window_size = 20  # Look at last 20 iterations
        self.improvement_threshold = 0.1  # 1% improvement
        self.stall_count = 0
        self.min_stall_iterations = 40  # Require this many stalled iterations before claiming convergence

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
            n_initial_points=30,  # Number of random points before using surrogate model
            random_state=42
        )

        # Setup results tracking
        self.results: List[Dict] = []
        self.best_score = float("-inf")
        self._last_iter_best_score = float("-inf")
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

    def check_convergence(self) -> bool:
        """
        Check if optimization appears to be converging.
        Returns True if we're likely converging.
        """
        print(f"score_history length: {len(self.score_history)}")
        if len(self.score_history) < self.window_size:
            return False

        # Look at recent scores
        recent_scores = self.score_history[-self.window_size:]
        
        # Calculate relative improvement over window
        max_recent = max(recent_scores)
        min_recent = min(recent_scores)
        print(f"prev best: {self._last_iter_best_score} vs {self.best_score}")
        relative_improvement = (max_recent - self._last_iter_best_score) / self._last_iter_best_score
        print(f"relative improvement: {relative_improvement}")
        self._last_iter_best_score = self.best_score

        # Check if we're still making meaningful improvements
        if relative_improvement < self.improvement_threshold:
            self.stall_count += 1 * self.n_parallel
            print(f"stalls: {self.stall_count}")
        else:
            self.stall_count = 0

        # Log convergence metrics
        self.logger.debug(
            f"Convergence check: improvement={relative_improvement:.4f}, "
            f"stall_count={self.stall_count}"
        )

        return self.stall_count >= self.min_stall_iterations

    def optimize(self, n_iterations: int = 100, plot_progress: bool = True):
        """
        Run optimization process

        Args:
            n_iterations: Number of points to evaluate
            plot_progress: Whether to show progress plot
        """

        start_time = datetime.now(timezone.utc)
        self.logger.info(f"Starting optimization with {n_iterations} iterations")

        current_resolution = "low"
        resolution_map = {"low": 100_000, "medium": 500_000, "high": 200_000}

        for iteration in range(0, n_iterations, self.n_parallel):
            # Generate batch of points
            batch_size = min(self.n_parallel, n_iterations - iteration)
            points_to_evaluate = self.generate_exploration_points(batch_size)
            
            # Run simulations in parallel
            with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
                futures = []
                for params in points_to_evaluate:
                    
                    params["shot_count"] = resolution_map[current_resolution]

                    futures.append(
                        executor.submit(
                            self.simulator.run_simulation,
                            generate_experiment_id(),
                            params
                        )
                    )

                # Process results as they complete
                batch_scores = []
                for i, future in enumerate(futures):
                    point = points_to_evaluate[i]
                    absorber_area = point["ceiling_center_width"] * (point["ceiling_center_xmax"] - point["ceiling_center_xmin"])
                    name, simulation_result, success = future.result()
                    itd = 0.0
                    avg_energy_over_window = 0.1
                    if simulation_result["status"] == "success":
                        score = simulation_result["results"]["ITD"]

                        itd = simulation_result["results"]["ITD"]
                        avg_energy_over_window = simulation_result["results"].get("avg_energy_over_window")

                        score = min(itd, 30) - absorber_area * 1.5 

                        batch_scores.append(score)
                    else:
                        score = -1
                        batch_scores.append(score)
                    

                    # Tell the optimizer about the result
                    self.optimizer.tell([point[param] for param in self.param_names], -score)  # Negative because we're maximizing

                    store_result = {
                        "params": points_to_evaluate[i],
                        "score": score,
                        "foms": {
                                "ITD": itd,
                                "avg_energy_over_window": avg_energy_over_window,
                                "absorber_area": absorber_area,
                                },
                        "simulation_results": simulation_result["results"],
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
                        self.logger.info(f"\t: ITD: {store_result['foms']['ITD']}")
                        self.logger.info(f"\t: avg_energy_25ms: {store_result['foms']['avg_energy_over_window']}")
                        self.logger.info(f"\t: absorber_area: {store_result['foms']['absorber_area']}")


                    
            # NEW: Update score history after batch completes
            self.score_history.extend(batch_scores)

            if self.check_convergence():
                # if plot_progress and (iteration + i + 1) % 10 == 0:
                if plot_progress: 
                    self.plot_progress(True)
                # if current_resolution == "low":
                #     self.logger.info("Converging at low resolution - switching to medium resolution")
                #     current_resolution = "medium"
                #     # Reset convergence tracking for new resolution
                #     self.score_history = []
                #     self.stall_count = 0
                #     self.best_score = 0
                # elif current_resolution == "medium":
                #     self.logger.info("Converging at medium resolution - switching to high resolution")
                #     current_resolution = "high"
                #     # Reset convergence tracking for new resolution
                #     self.score_history = []
                #     self.stall_count = 0
                #     self.best_score = 0
                # elif current_resolution == "high":
                #     self.logger.info(
                #         f"Optimization appears to have fully converged after {iteration} iterations. "
                #         f"Best score: {self.best_score:.4f}"
                #     )
                    break  # End optimization early
            if plot_progress: 
                self.plot_progress(False)


        # Final results
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.logger.info(f"Optimization completed in {duration:.1f}s")
        self.logger.info(f"Best score: {self.best_score}")
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best experiment: {self.best_experiment_name}")

        # Save results
        self.save_results()

    def plot_progress(self, reset_best_so_far: bool = False):
        """
        Plot optimization progress
        
        Args:
            reset_best_so_far: If True, reset the "best so far" line at this point
        """

        if reset_best_so_far:
            self._plot_from = len(self.results) - 1

        plt.figure(figsize=(10, 6))

        # Plot scores over iterations
        scores = [r["score"] for r in self.results]
        iterations = range(len(scores))

        # Always plot all scores
        plt.plot(iterations, scores, "b.", label="Scores")

        if hasattr(self, '_plot_from'):
            plt.plot(iterations[self._plot_from:], np.maximum.accumulate(scores[self._plot_from:]), 
                    "r-", label="Best so far")
        else:
            plt.plot(iterations, np.maximum.accumulate(scores), 
                    "r-", label="Best so far")

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

    if len(sys.argv) != 4:
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
            "ceiling_center_width": (0.25, 3.5),
            "ceiling_center_xmin": (0.3, 1.1),
            "ceiling_center_xmax": (1.3, 3.0),
            "wall_absorbers_Street_A": (0.5, 1.3),
            "wall_absorbers_Hall_B": (0.5, 1.3),
            "wall_absorbers_Street_A": (0.5, 1.3),
            "wall_absorbers_Door_Side_A": (0.5, 1.3),
            "wall_absorbers_Hall_E": (0.5, 1.3),
            "wall_absorbers_Street_D": (0.5, 1.3),
            "wall_absorbers_Street_B": (0.5, 1.3),
            "wall_absorbers_Door_Side_B": (0.5, 1.3),
            "wall_absorbers_Entry_Back": (0.5, 1.3),
            "wall_absorbers_Street_C": (0.5, 1.3),
            "wall_absorbers_Street_E": (0.5, 1.3),
            "wall_absorbers_Hall_A": (0.5, 1.3),
            "wall_absorbers_Entry_Front": (0.5, 1.3),
            "wall_absorbers_Door": (0.5, 1.3),
            "wall_absorbers_Back_A": (0.5, 1.3),
            "wall_absorbers_Back_B": (0.5, 1.3),
        }
    )

    # Run optimization
    optimizer.optimize(n_iterations=1000, plot_progress=True)
