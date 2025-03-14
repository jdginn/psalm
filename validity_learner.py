"""
Validity Learning System for Acoustic Simulation Parameters
Created: 2025-03-14 by jdginn
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import datetime
import sys

import yaml
import subprocess
from pathlib import Path
import tempfile


class ValidityChecker:
    def __init__(self, program_path):
        """
        Initialize the validity checker

        Args:
            program_path: Path to the external program to run
        """
        self.program_path = Path(program_path)

        # Template as a string - exactly as specified
        self.template = """listening_triangle:
  distance_from_front: {distance_from_front}
  distance_from_center: {distance_from_center}
  source_height: {source_height}
  listen_height: {listen_height}
"""
        # Create a temporary directory for our YAML files
        self.temp_dir = Path(tempfile.mkdtemp())
        print(f"Using temporary directory: {self.temp_dir}")

    def check(self, params):
        """
        Check validity of parameters by running external program

        Args:
            params: Dictionary containing:
                   - distance_from_front
                   - distance_from_center
                   - source_height
                   - listen_height
        Returns:
            1.0 if valid (program exits 0)
            -1.0 if invalid (program exits non-zero)
        """
        # Format the template with the provided parameters
        config_str = self.template.format(**params)

        # Create a temporary YAML file
        temp_yaml = (
            self.temp_dir
            / f"config_{params['distance_from_front']}_{params['distance_from_center']}_{params['source_height']}_{params['listen_height']}.yaml"
        )
        with open(temp_yaml, "w") as f:
            f.write(config_str)

        try:
            # Run the external program
            result = subprocess.run(
                [str(self.program_path), str(temp_yaml)],
                capture_output=True,
                text=True,
                check=False,  # Don't raise on non-zero exit
            )

            # Clean up the temporary file
            temp_yaml.unlink()

            # Return based on exit code
            return 1.0 if result.returncode == 0 else -1.0

        except Exception as e:
            print(f"Error running validity check: {e}")
            # Clean up the temporary file
            if temp_yaml.exists():
                temp_yaml.unlink()
            return -1.0

    def __del__(self):
        """Cleanup temporary directory when the checker is destroyed"""
        try:
            import shutil

            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")


# Example usage
if __name__ == "__main__":
    # Example parameters
    params = {
        "distance_from_front": 0.516,
        "distance_from_center": 1.352,
        "source_height": 1.7,
        "listen_height": 1.4,
    }

    checker = ValidityChecker(program_path="path/to/program")

    result = checker.check(params)
    print(f"Validity check result: {result}")


class ValidityLearner:
    def __init__(self, bounds, validity_checker_func):
        """
        Initialize the validity learner with parameter bounds and validity checker.

        Args:
            bounds: Dictionary of parameter bounds like:
                   {'distance_from_front': (min, max), ...}
            validity_checker_func: Function that takes a dictionary of parameters
                                 and returns validity score
        """
        self.bounds = bounds
        self.check_validity_impl = validity_checker_func

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

        print(f"ValidityLearner initialized at {datetime.datetime.utcnow()}")

    def check_validity(self, params):
        """
        Wrapper around the provided validity checker

        Args:
            params: Either dictionary or array of parameters
        Returns:
            Validity score from the checker
        """
        if isinstance(params, np.ndarray):
            params_dict = {name: params[i] for i, name in enumerate(self.bounds.keys())}
        else:
            params_dict = params

        return self.check_validity_impl(params_dict)

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
        print(f"Starting validity learning at {datetime.datetime.utcnow()}")

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


# Example usage
if __name__ == "__main__":
    # Example validity checker function
    def example_validity_checker(params):
        """
        Example validity checker - replace with your actual checker
        Returns positive number for valid configurations, negative for invalid
        """
        return 1.0  # Replace with actual validity checking logic

    checker = ValidityChecker(program_path=sys.argv[1])

    # Example bounds
    example_bounds = {
        "distance_from_front": (0, 5.0),
        "distance_from_center": (0, 3.0),
        "source_height": (0, 2.5),
        "listen_height": (0, 2.0),
    }

    # Create learner
    learner = ValidityLearner(example_bounds, example_validity_checker)

    # Learn validity boundary
    learner.learn_validity_boundary(
        initial_samples=2000, n_rounds=15, samples_per_round=1000
    )

    # Get starting points for optimization
    starting_points = learner.suggest_optimization_starting_points(n_points=15)

    # Example of how to use with optimization
    def constrained_acoustic_objective(params):
        """Example objective function for optimization"""
        if not learner.predict_validity(params):
            return float("inf")
        return 0.0  # Replace with actual acoustic simulation
