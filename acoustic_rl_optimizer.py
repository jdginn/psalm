"""
Reinforcement Learning-based Acoustic Optimizer
Created: 2025-03-14 by jdginn
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from torch.distributions import Normal
from pathlib import Path
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
from collections import deque
import joblib

from acoustic_optimizer import AcousticSimulator


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("acoustic_rl_optimization.log"),
        ],
    )


class AcousticEnvironment:
    """Environment wrapper for acoustic simulation"""

    def __init__(self, simulator, model_data: Dict, bounds: Dict):
        self.simulator = simulator
        # Correctly access the loaded model data
        self.validity_model = model_data.get("model")  # The actual model
        if not self.validity_model:
            raise ValueError("No model found in model_data")

        self.scaler = model_data.get("scaler")  # The scaler
        if not self.scaler:
            raise ValueError("No scaler found in model_data")

        self.bounds = bounds
        self.parameter_names = list(bounds.keys())

        # Normalize bounds for RL
        self.normalized_bounds = {param: (-1.0, 1.0) for param in bounds}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Execute one environment step"""
        # Convert normalized action to parameter values
        params = self.denormalize_parameters(action)

        # Check validity using proper predict_proba call
        param_values = [params[param] for param in self.parameter_names]
        scaled_params = self.scaler.transform([param_values])
        try:
            validity_score = self.validity_model.predict_proba(scaled_params)[0][1]
        except Exception as e:
            logging.error(f"Error predicting validity: {e}")
            return action, -1.0, True

        # If invalid, return negative reward
        if validity_score < 0.8:
            return action, -1.0, True

        # Run simulation
        score, success = self.simulator.run_simulation(params)

        # Return normalized state, reward, done
        return action, float(score), True

    def denormalize_parameters(self, normalized_params: np.ndarray) -> Dict[str, float]:
        """Convert normalized parameters (-1 to 1) to actual values"""
        params = {}
        for i, param in enumerate(self.parameter_names):
            low, high = self.bounds[param]
            normalized_value = np.clip(normalized_params[i], -1.0, 1.0)
            params[param] = low + (normalized_value + 1.0) * 0.5 * (high - low)
        return params

    def normalize_parameters(self, params: Dict[str, float]) -> np.ndarray:
        """Convert actual parameters to normalized values (-1 to 1)"""
        normalized = []
        for param in self.parameter_names:
            low, high = self.bounds[param]
            value = params[param]
            normalized.append(2.0 * (value - low) / (high - low) - 1.0)
        return np.array(normalized)


class Actor(nn.Module):
    """Policy network"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Mean and log std networks
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std


class Critic(nn.Module):
    """Value network"""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class AcousticRLOptimizer:
    def __init__(
        self,
        simulator,
        validity_model_path: Path,
        output_dir: Path,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load and verify validity model data
        try:
            self.model_data = joblib.load(validity_model_path)
            if not isinstance(self.model_data, dict):
                raise ValueError(
                    f"Expected dictionary from joblib load, got {type(self.model_data)}"
                )

            required_keys = ["model", "scaler", "bounds"]
            missing_keys = [key for key in required_keys if key not in self.model_data]
            if missing_keys:
                raise ValueError(f"Missing required keys in model data: {missing_keys}")

            self.bounds = self.model_data["bounds"]

        except Exception as e:
            self.logger.error(f"Error loading model data: {e}")
            raise

        # Setup environment
        self.env = AcousticEnvironment(simulator, self.model_data, self.bounds)

        # Setup networks
        state_dim = len(self.bounds)
        action_dim = state_dim  # Same as state dimension for our case

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.target_critic = Critic(state_dim, hidden_dim).to(device)

        # Copy critic parameters to target
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Setup optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Training parameters
        self.device = device
        self.gamma = gamma
        self.tau = tau

        # Experience buffer
        self.buffer = deque(maxlen=10000)

        # Results tracking
        self.results = []
        self.best_score = float("-inf")
        self.best_params = None

        # Setup logging
        self.logger = logging.getLogger("AcousticRLOptimizer")
        self.logger.setLevel(logging.INFO)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean, log_std = self.actor(state_tensor)
            std = log_std.exp()

            # Sample action from Gaussian distribution
            normal = Normal(mean, std)
            action = normal.sample()
            action = torch.tanh(action)  # Bound to [-1, 1]

        return action.cpu().numpy()[0]

    def optimize(self, n_iterations: int = 1000, batch_size: int = 64):
        """Run RL optimization"""
        start_time = datetime.now(timezone.utc)
        self.logger.info(f"Starting RL optimization with {n_iterations} iterations")

        for iteration in range(n_iterations):
            # Sample initial state (current best or random)
            if len(self.results) > 0 and np.random.random() < 0.8:
                state = self.env.normalize_parameters(
                    max(self.results, key=lambda x: x["score"])["params"]
                )
            else:
                state = np.random.uniform(-1, 1, len(self.bounds))

            # Select and execute action
            action = self.select_action(state)
            next_state, reward, done = self.env.step(action)

            # Store experience
            self.buffer.append((state, action, reward, next_state, done))

            # Record results
            params = self.env.denormalize_parameters(action)
            self.results.append(
                {
                    "params": params,
                    "score": reward,
                    "iteration": iteration,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            if reward > self.best_score:
                self.best_score = reward
                self.best_params = params
                self.logger.info(f"New best score: {reward}")

            # Update networks if we have enough samples
            if len(self.buffer) >= batch_size:
                self.update_networks(batch_size)

            # Plot progress
            if (iteration + 1) % 10 == 0:
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
        """Save optimization results with proper type conversion"""
        output_data = {
            "best_score": float(self.best_score),  # Convert to regular float
            "best_params": {
                k: float(v)
                for k, v in self.best_params.items()  # Convert all values to regular floats
            },
            "results": [
                {
                    "params": {k: float(v) for k, v in result["params"].items()},
                    "score": float(result["score"]),
                    "iteration": result["iteration"],
                    "timestamp": result["timestamp"],
                }
                for result in self.results
            ],
        }

        output_path = self.output_dir / "optimization_results.json"
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        self.logger.info(f"Saved results to {output_path}")

    def update_networks(self, batch_size: int):
        """Update actor and critic networks"""
        # Sample batch
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for idx in batch:
            state, action, reward, next_state, done = self.buffer[idx]
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # Update critic
        with torch.no_grad():
            next_value = self.target_critic(next_state_batch).squeeze()
            target_value = reward_batch + (1 - done_batch) * self.gamma * next_value

        current_value = self.critic(state_batch).squeeze()
        critic_loss = nn.MSELoss()(current_value, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        mean, log_std = self.actor(state_batch)
        std = log_std.exp()
        normal = Normal(mean, std)
        action = normal.rsample()
        action = torch.tanh(action)

        actor_loss = -self.critic(state_batch).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


# Example usage
if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)

    if len(sys.argv) != 3:
        print(
            "Usage: python acoustic_rl_optimizer.py <simulator_path> <validity_model_path>"
        )
        sys.exit(1)

    simulator_path = Path(sys.argv[1])
    validity_model_path = Path(sys.argv[2])

    # Setup simulator
    simulator = AcousticSimulator(simulator_path)

    # Setup RL optimizer
    optimizer = AcousticRLOptimizer(
        simulator=simulator,
        validity_model_path=validity_model_path,
        output_dir=Path("rl_optimization_results"),
    )

    # Run optimization
    optimizer.optimize(n_iterations=10_000)
