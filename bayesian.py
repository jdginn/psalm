import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from typing import List, Dict, Any
import logging


class AcousticOptimizer:
    """
    Handles the Bayesian optimization logic using GPyTorch/BoTorch.
    Maintains the surrogate model and generates new parameter suggestions.

    Interacts with:
    - ExperimentManager: Receives results and provides new suggestions
    """

    def __init__(self, bounds: torch.Tensor, initial_data: Dict[str, torch.Tensor]):
        """
        Initializes the optimizer with parameter bounds and initial data.

        Args:
            bounds: Tensor specifying the bounds for each parameter
            initial_data: Dictionary containing initial training data with keys 'train_X' and 'train_Y'
        """
        self.bounds = bounds
        self.train_X = initial_data["train_X"]
        self.train_Y = initial_data["train_Y"]

        # Initialize the Gaussian Process model
        self.model = SingleTaskGP(self.train_X, self.train_Y)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )

        # Fit the model
        fit_gpytorch_model(self.mll)

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def update_model(self, new_results: Dict[str, torch.Tensor]):
        """
        Updates the surrogate model with new experimental results.

        Args:
            new_results: Dictionary containing new training data with keys 'train_X' and 'train_Y'
        """
        self.train_X = torch.cat([self.train_X, new_results["train_X"]], dim=0)
        self.train_Y = torch.cat([self.train_Y, new_results["train_Y"]], dim=0)

        # Reinitialize and refit the model with updated data
        self.model = SingleTaskGP(self.train_X, self.train_Y)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        fit_gpytorch_model(self.mll)

        self.logger.info("Surrogate model updated with new results")

    def suggest_batch(self, batch_size: int = 8) -> List[Dict[str, float]]:
        """
        Generates a batch of new parameter suggestions using Expected Improvement acquisition function.

        Args:
            batch_size: Number of parameter sets to suggest

        Returns:
            List of dictionaries containing suggested parameter sets
        """
        # Define the acquisition function
        acq_func = ExpectedImprovement(self.model, best_f=self.train_Y.max())

        # Optimize the acquisition function to get new candidate points
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=batch_size,
            num_restarts=10,
            raw_samples=512,
        )

        # Convert tensor candidates to list of dictionaries
        suggestions = []
        for candidate in candidates:
            suggestion = {
                f"param_{i}": candidate[i].item() for i in range(candidate.shape[0])
            }
            suggestions.append(suggestion)

        self.logger.info(f"Generated {batch_size} new parameter suggestions")
        return suggestions

    def get_optimization_state(self) -> Dict[str, torch.Tensor]:
        """
        Returns the current state of the optimization process.

        Returns:
            Dictionary containing the current training data
        """
        return {"train_X": self.train_X, "train_Y": self.train_Y}

    def update_acquisition_strategy(self, stage_config: Dict[str, Any]):
        """
        Updates the optimization parameters based on the current stage configuration.

        Args:
            stage_config: Dictionary containing stage-specific parameters
        """
        # Update any stage-specific parameters here
        # For example, you might adjust the exploration coefficient (κ) or FOM weights
        self.exploration_coefficient = stage_config.get("kappa", 2.0)
        self.fom_weights = stage_config.get(
            "fom_weights", {"ITD": 0.33, "AVG_GAIN_5ms": 0.33, "ITD_2": 0.33}
        )

        self.logger.info("Acquisition strategy updated with new stage parameters")
