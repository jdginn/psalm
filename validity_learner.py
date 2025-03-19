"""
Validity Learning System for Acoustic Simulation Parameters
Created: 2025-03-14 by jdginn
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, UTC
import sys
import joblib

import yaml
import subprocess
from pathlib import Path
import tempfile

from simulation_runner import SimulationRunner
from name_generator import generate_experiment_id


class ValidityLearner:
    def __init__(self, bounds, simulator):
        """
        Initialize the validity learner with parameter bounds and validity checker.

        Args:
            bounds: Dictionary of parameter bounds like:
                   {'distance_from_front': (min, max), ...}
            simluator: SimulationRunner instance
        """
        self.bounds = bounds
        self.simulator = simulator

        # Known trends about parameter impacts on validity
        self.parameter_trends = {
            "distance_from_front": 1,  # positive correlation with validity
            "distance_from_center": -1,  # negative correlation with validity
            "source_height": -1,  # negative correlation with validity
            "listen_height": 0,  # no known trend
        }

        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, min_samples_leaf=10
        )

        self.valid_points = []
        self.invalid_points = []
        self.samples_X = []
        self.validity_y = []

        # print(
        #     f"ValidityLearner initialized at {datetime.datetime.now(datetime.timezone.utc)}"
        # )

    def check_validity(self, params):
        """
        Wrapper around the provided validity checker

        Args:
            params: Either dictionary or array of parameters
        Returns:
            Validity score from the checker
        """

        name = generate_experiment_id()

        results, ok = self.simulator.run_simulation(name, params)
        if not ok:
            return -1.0
        if results["status"] != "success":
            return -1.0
        return 1.0

    def _generate_biased_samples(self, n_samples):
        """
        Generate samples with bias towards likely valid regions
        based on known parameter trends
        """
        samples = []
        for _ in range(n_samples):
            sample = {}
            for param, (min_val, max_val) in self.bounds.items():
                trend = self.parameter_trends[param]
                if trend == 1:
                    # Bias towards larger values
                    alpha = np.random.beta(2, 1)
                elif trend == -1:
                    # Bias towards smaller values
                    alpha = np.random.beta(1, 2)
                else:
                    # No bias
                    alpha = np.random.random()

                value = min_val + alpha * (max_val - min_val)
                sample[param] = value
            samples.append(sample)
        return samples

    def _smart_boundary_sampling(self, n_samples):
        """
        Sample more densely near the decision boundary,
        taking into account known trends
        """
        if len(self.valid_points) < 10 or len(self.invalid_points) < 10:
            return self._generate_biased_samples(n_samples)

        valid_array = np.array(self.valid_points)
        invalid_array = np.array(self.invalid_points)

        boundary_samples = []

        # Sample between valid and invalid points, with bias
        for valid_point in valid_array[: min(len(valid_array), n_samples // 2)]:
            # Find nearest invalid neighbors
            distances = np.linalg.norm(invalid_array - valid_point, axis=1)
            nearest_idx = np.argpartition(distances, 3)[:3]

            for invalid_point in invalid_array[nearest_idx]:
                # Generate multiple points along the line between valid and invalid
                alphas = np.random.beta(2, 2, size=3)
                for alpha in alphas:
                    interpolated = valid_point * alpha + invalid_point * (1 - alpha)
                    # Apply trend-based adjustment
                    for i, param in enumerate(self.bounds.keys()):
                        trend = self.parameter_trends[param]
                        if trend != 0:
                            interpolated[i] += trend * np.random.normal(0, 0.05)
                    boundary_samples.append(interpolated)

        return boundary_samples

    def _record_sample(self, point, validity):
        """Record a sample and its validity"""
        if isinstance(point, dict):
            point_array = np.array([point[param] for param in self.bounds.keys()])
        else:
            point_array = point

        self.samples_X.append(point_array)
        self.validity_y.append(validity > 0)  # Convert to boolean

        if validity > 0:
            self.valid_points.append(point_array)
        else:
            self.invalid_points.append(point_array)

    def _update_model(self):
        """Update the classifier with current data"""
        X = np.array(self.samples_X)
        y = np.array(self.validity_y)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def learn_validity_boundary(
        self, initial_samples=1000, n_rounds=10, samples_per_round=500
    ):
        """
        Actively learn the validity boundary with bias towards known trends

        Args:
            initial_samples: Number of initial samples to take
            n_rounds: Number of refinement rounds
            samples_per_round: Number of samples per refinement round
        """
        print(f"Starting validity learning at {datetime.utcnow()}")

        # Initial sampling biased by known trends
        initial_points = self._generate_biased_samples(initial_samples)

        for point in initial_points:
            validity = self.check_validity(point)
            self._record_sample(point, validity)

        print(
            f"Initial sampling complete. Valid: {len(self.valid_points)}, Invalid: {len(self.invalid_points)}"
        )

        # Iterative refinement
        for round in range(n_rounds):
            self._update_model()

            # Generate boundary-focused samples
            boundary_samples = self._smart_boundary_sampling(samples_per_round)

            for point in boundary_samples:
                validity = self.check_validity(point)
                self._record_sample(point, validity)

            print(
                f"Round {round+1} complete. Valid: {len(self.valid_points)}, Invalid: {len(self.invalid_points)}"
            )

        self._update_model()

    def predict_validity(self, params):
        """
        Predict validity of a new point

        Args:
            params: Dictionary or array of parameters
        Returns:
            Boolean prediction of validity
        """
        if isinstance(params, dict):
            params_array = np.array([params[param] for param in self.bounds.keys()])
        else:
            params_array = params

        params_scaled = self.scaler.transform([params_array])
        return self.model.predict(params_scaled)[0]

    def suggest_optimization_starting_points(self, n_points=10):
        """
        Suggest diverse starting points for acoustic optimization

        Args:
            n_points: Number of starting points to suggest
        Returns:
            Array of parameter sets to try
        """
        valid_array = np.array(self.valid_points)
        if len(valid_array) < n_points:
            return valid_array

        # Get predicted probabilities for all valid points
        probs = self.model.predict_proba(self.scaler.transform(valid_array))[:, 1]

        # Combine validity probability with diversity
        selected_indices = []
        remaining_indices = set(range(len(valid_array)))

        # Select first point with highest validity probability
        first_idx = np.argmax(probs)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Select remaining points balancing validity and diversity
        while len(selected_indices) < n_points:
            best_score = -np.inf
            best_idx = None

            for idx in remaining_indices:
                point = valid_array[idx]
                # Validity score
                validity_score = probs[idx]
                # Diversity score (minimum distance to selected points)
                min_dist = min(
                    np.linalg.norm(point - valid_array[s]) for s in selected_indices
                )
                # Combined score
                score = validity_score * min_dist

                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return valid_array[selected_indices]

    def save_model(self, output_dir: str = "validity_models") -> Path:
        """
        Save the trained model and associated data

        Args:
            output_dir: Directory to save the model

        Returns:
            Path to the saved model file
        """
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create model data dictionary
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "bounds": self.bounds,
            "parameter_trends": self.parameter_trends,
            "valid_points": self.valid_points,
            "invalid_points": self.invalid_points,
            "metadata": {
                "created_at": datetime.now(UTC).isoformat(),
                "created_by": "jdginn",
                "n_valid_points": len(self.valid_points),
                "n_invalid_points": len(self.invalid_points),
            },
        }

        # Create filename with timestamp
        filename = (
            f"validity_model_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.joblib"
        )
        model_path = output_dir / filename

        # Save the model
        joblib.dump(model_data, model_path)
        print(f"Saved model to {model_path}")
        return model_path

    @classmethod
    def load_model(cls, model_path: Path, checker):
        """
        Load a saved model

        Args:
            model_path: Path to the saved model file
            checker: ValidityChecker instance to use with the loaded model

        Returns:
            ValidityLearner instance with loaded model
        """
        model_data = joblib.load(model_path)

        # Create new instance
        learner = cls(model_data["bounds"], checker)

        # Load saved attributes
        learner.model = model_data["model"]
        learner.scaler = model_data["scaler"]
        learner.parameter_trends = model_data["parameter_trends"]
        learner.valid_points = model_data["valid_points"]
        learner.invalid_points = model_data["invalid_points"]

        print(f"Loaded model from {model_path}")
        print(f"Model metadata: {model_data['metadata']}")

        return learner


if __name__ == "__main__":
    # Get the program path from command line argument
    if len(sys.argv) != 4:
        print("Usage: python validity_learner.py <path_to_check_program> <path_to_save_results> <path_to_base_config_file>")
        sys.exit(1)

    program_path = Path(sys.argv[1])
    base_directory = Path(sys.argv[2])
    config_path = Path(sys.argv[3])
    if not program_path.exists():
        print(f"Error: Program not found at {program_path}")
        sys.exit(1)

    # Example bounds
    bounds = {
        "distance_from_front": (0, 0.8),
        "distance_from_center": (0, 3.0),
        "source_height": (0, 2.2),
        "listen_height": (1.0, 1.6),
    }

    # Create checker and learner
    simulator = SimulationRunner(program_path=program_path, base_directory=base_directory, config_path=config_path)
    learner = ValidityLearner(bounds, simulator)

    # Test a single point first
    test_params = {
        "distance_from_front": 0.516,
        "distance_from_center": 1.352,
        "source_height": 1.7,
        "listen_height": 1.4,
    }

    print("Testing single point...")
    result = learner.check_validity(test_params)
    print(f"Test result: {result}")

    if input("Continue with learning? (y/n): ").lower().strip() == "y":
        # Learn validity boundary
        learner.learn_validity_boundary(
            initial_samples=100,  # Starting with fewer samples for testing
            n_rounds=5,
            samples_per_round=50,
        )

        # Get starting points for optimization
        starting_points = learner.suggest_optimization_starting_points(n_points=5)
        print("\nSuggested starting points:")
        for point in starting_points:
            print(point)
            # Save the model
        if input("\nSave model? (y/n): ").lower().strip() == "y":
            model_path = learner.save_model()
            print(f"\nModel saved to: {model_path}")

            # Example of loading the model
            print("\nTesting model loading...")
            loaded_learner = ValidityLearner.load_model(model_path, checker)

            # Verify loaded model works
            test_point = starting_points[0]
            original_prediction = learner.predict_validity(test_point)
            loaded_prediction = loaded_learner.predict_validity(test_point)
            print(f"Original model prediction: {original_prediction}")
            print(f"Loaded model prediction: {loaded_prediction}")
