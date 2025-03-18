"""
Model Management and Training Integration
Created: 2025-03-14 by jdginn
"""

import joblib
from pathlib import Path
import json
from datetime import datetime, UTC
import numpy as np
from typing import Dict, List, Tuple, Optional


class ModelManager:
    def __init__(self, base_dir: Path = Path("acoustic_models")):
        """
        Initialize model manager

        Args:
            base_dir: Base directory for storing models and training data
        """
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.training_data_dir = self.base_dir / "training_data"
        self.results_dir = self.base_dir / "results"

        # Create directories
        for dir_path in [self.models_dir, self.training_data_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Metadata for tracking model evolution
        self.metadata_file = self.base_dir / "model_evolution.json"
        self.load_metadata()

    def load_metadata(self):
        """Load or initialize model evolution metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "models": [],
                "last_updated": datetime.now(UTC).isoformat(),
                "current_model": None,
                "user": "jdginn",
            }
            self.save_metadata()

    def save_metadata(self):
        """Save model evolution metadata"""
        self.metadata["last_updated"] = datetime.now(UTC).isoformat()
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def save_model(
        self,
        model_data: Dict,
        model_type: str = "validity",
        description: str = "",
        acoustic_scores: Optional[Dict] = None,
    ) -> Path:
        """
        Save a model with metadata

        Args:
            model_data: Dictionary containing model and associated data
            model_type: Type of model ("validity" or "acoustic")
            description: Description of the model/training
            acoustic_scores: Optional acoustic evaluation results
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_model_{timestamp}.joblib"
        model_path = self.models_dir / model_name

        # Add metadata
        model_data["metadata"] = {
            "created": datetime.now(UTC).isoformat(),
            "created_by": "jdginn",
            "model_type": model_type,
            "description": description,
            "acoustic_scores": acoustic_scores,
        }

        # Save model
        joblib.dump(model_data, model_path)

        # Update metadata
        self.metadata["models"].append(
            {
                "path": str(model_path),
                "type": model_type,
                "timestamp": datetime.now(UTC).isoformat(),
                "description": description,
            }
        )
        self.metadata["current_model"] = str(model_path)
        self.save_metadata()

        print(f"Saved model to {model_path}")
        return model_path

    def load_latest_model(self, model_type: str = "validity") -> Dict:
        """Load the most recent model of specified type"""
        if not self.metadata["models"]:
            raise ValueError("No models available")

        # Find latest model of specified type
        matching_models = [
            m for m in self.metadata["models"] if m["type"] == model_type
        ]

        if not matching_models:
            raise ValueError(f"No models of type {model_type} available")

        latest_model = max(matching_models, key=lambda m: m["timestamp"])
        return joblib.load(latest_model["path"])

    def save_training_data(
        self, data: Dict[str, List], data_type: str = "acoustic"
    ) -> Path:
        """Save training data with metadata"""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{data_type}_training_{timestamp}.json"
        file_path = self.training_data_dir / filename

        output_data = {
            "metadata": {
                "created": datetime.now(UTC).isoformat(),
                "created_by": "jdginn",
                "data_type": data_type,
            },
            "data": data,
        }

        with open(file_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved training data to {file_path}")
        return file_path


class AcousticTrainer:
    def __init__(
        self, model_manager: ModelManager, validity_confidence_threshold: float = 0.8
    ):
        """
        Initialize acoustic trainer

        Args:
            model_manager: ModelManager instance for model persistence
            validity_confidence_threshold: Minimum confidence for validity predictions
        """
        self.model_manager = model_manager
        self.confidence_threshold = validity_confidence_threshold

        # Load latest validity model
        self.validity_model_data = model_manager.load_latest_model("validity")

    def train_with_acoustic_data(
        self,
        acoustic_results: List[Tuple[Dict[str, float], float]],
        min_score_threshold: Optional[float] = None,
    ) -> Path:
        """
        Train model incorporating acoustic simulation results

        Args:
            acoustic_results: List of (parameters, score) tuples
            min_score_threshold: Optional minimum score to consider "good"
        """
        # Save acoustic training data
        training_data = {
            "parameters": [params for params, _ in acoustic_results],
            "scores": [float(score) for _, score in acoustic_results],
        }
        self.model_manager.save_training_data(training_data, "acoustic")

        # If no threshold provided, use median of scores
        if min_score_threshold is None:
            min_score_threshold = np.median([score for _, score in acoustic_results])

        # Combine validity and acoustic data
        X = []
        y = []

        for params, score in acoustic_results:
            # Check validity confidence
            point_array = np.array(
                [params[param] for param in self.validity_model_data["bounds"].keys()]
            )
            point_scaled = self.validity_model_data["scaler"].transform([point_array])
            validity_proba = self.validity_model_data["model"].predict_proba(
                point_scaled
            )[0]

            # Only include points that are confidently valid
            if validity_proba[1] >= self.confidence_threshold:
                X.append(point_array)
                # Mark as valid only if both valid and acoustically good
                y.append(score >= min_score_threshold)

        if not X:
            raise ValueError("No valid points found in acoustic data")

        # Train new model
        X = np.array(X)
        y = np.array(y)

        new_scaler = StandardScaler()
        X_scaled = new_scaler.fit_transform(X)

        new_model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, min_samples_leaf=10
        )
        new_model.fit(X_scaled, y)

        # Save new model
        model_data = {
            "model": new_model,
            "scaler": new_scaler,
            "bounds": self.validity_model_data["bounds"],
            "parameter_trends": self.validity_model_data["parameter_trends"],
            "acoustic_threshold": min_score_threshold,
        }

        return self.model_manager.save_model(
            model_data,
            model_type="acoustic",
            description=f"Trained on {len(X)} acoustic samples",
            acoustic_scores={
                "threshold": float(min_score_threshold),
                "n_samples": len(X),
                "n_good": int(sum(y)),
            },
        )


# Example usage:
if __name__ == "__main__":
    # Initialize model manager
    model_manager = ModelManager()

    # Load existing validity model
    try:
        validity_model = model_manager.load_latest_model("validity")
        print("Loaded existing validity model")
    except ValueError:
        print("No existing validity model found")

    # Example of saving acoustic training results
    acoustic_results = [
        (
            {
                "distance_from_front": 4.0,
                "distance_from_center": 0.5,
                "source_height": 0.8,
                "listen_height": 1.0,
            },
            0.85,
        ),  # Good score
        (
            {
                "distance_from_front": 3.0,
                "distance_from_center": 1.5,
                "source_height": 1.2,
                "listen_height": 1.5,
            },
            0.45,
        ),  # Poor score
    ]

    # Train with acoustic data
    trainer = AcousticTrainer(model_manager)
    new_model_path = trainer.train_with_acoustic_data(acoustic_results)

    print(f"Trained and saved new model to {new_model_path}")
