from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional
from uuid import uuid4
import asyncio
import logging


@dataclass
class ExperimentParameters:
    """
    Represents a single set of parameters for an experiment
    Includes both Group 1 and Group 2 parameters from the spec
    """

    group1_params: Dict[str, float]  # Variable parameters like thickness, height
    group2_params: Dict[str, float]  # Fixed parameters for the simulation
    experiment_id: str = None

    def __post_init__(self):
        if not self.experiment_id:
            # Generate unique experiment ID with timestamp and random component
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            self.experiment_id = f"{timestamp}-{uuid4().hex[:6]}"


@dataclass
class ExperimentResult:
    """
    Holds the results of a single experiment
    Includes all FOMs and metadata
    """

    experiment_id: str
    itd: float
    avg_gain_5ms: float
    itd_2: float
    computation_time: float
    success: bool
    error_message: Optional[str] = None


class ExperimentStatus(Enum):
    """
    Tracks the status of each experiment through its lifecycle
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"


class ExperimentManager:
    """
    Central coordinator for the optimization process.

    Responsibilities:
    1. Maintain experiment queue and status
    2. Coordinate between optimizer and simulation runner
    3. Handle stage transitions
    4. Manage parallel execution
    5. Track experiment history
    """

    def __init__(
        self,
        optimizer,  # AcousticOptimizer instance
        simulator,  # SimulationRunner instance
        stage_manager,  # OptimizationStageManager instance
        max_parallel: int = 8,
        max_retries: int = 2,
    ):
        self.optimizer = optimizer
        self.simulator = simulator
        self.stage_manager = stage_manager
        self.max_parallel = max_parallel
        self.max_retries = max_retries

        # Track experiment status
        self.experiment_status: Dict[str, ExperimentStatus] = {}
        self.retry_count: Dict[str, int] = {}

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Active experiments semaphore
        self.active_experiments = asyncio.Semaphore(max_parallel)

    async def run_optimization(self, max_experiments: int = 120):
        """
        Main optimization loop. Runs until max_experiments is reached or
        optimization converges.
        """
        experiments_completed = 0

        while experiments_completed < max_experiments:
            # Check if we should transition to next stage
            if self.stage_manager.evaluate_transition_criteria():
                await self._handle_stage_transition()

            # Get next batch of experiments
            batch = await self._prepare_next_batch()

            # Run batch in parallel
            results = await self._run_batch(batch)

            # Process results
            experiments_completed += await self._process_batch_results(results)

            # Check for convergence
            if self._check_convergence():
                self.logger.info("Optimization converged")
                break

    async def _prepare_next_batch(self) -> List[ExperimentParameters]:
        """
        Prepares the next batch of experiments based on current stage and state
        """
        stage_config = self.stage_manager.get_stage_parameters()
        suggested_params = self.optimizer.suggest_batch(
            batch_size=self.max_parallel, stage_config=stage_config
        )

        batch = []
        for params in suggested_params:
            exp_params = ExperimentParameters(**params)
            self.experiment_status[exp_params.experiment_id] = ExperimentStatus.QUEUED
            batch.append(exp_params)

        return batch

    async def _run_batch(
        self, batch: List[ExperimentParameters]
    ) -> List[ExperimentResult]:
        """
        Runs a batch of experiments in parallel
        """

        async def run_single_experiment(params: ExperimentParameters):
            async with self.active_experiments:
                self.experiment_status[params.experiment_id] = ExperimentStatus.RUNNING
                try:
                    result = await self.simulator.run_simulation(params)
                    self.experiment_status[params.experiment_id] = (
                        ExperimentStatus.COMPLETED
                    )
                    return result
                except Exception as e:
                    self.logger.error(
                        f"Experiment {params.experiment_id} failed: {str(e)}"
                    )
                    self.experiment_status[params.experiment_id] = (
                        ExperimentStatus.FAILED
                    )
                    return ExperimentResult(
                        experiment_id=params.experiment_id,
                        success=False,
                        error_message=str(e),
                    )

        # Run all experiments in parallel
        tasks = [run_single_experiment(params) for params in batch]
        results = await asyncio.gather(*tasks)
        return results

    async def _process_batch_results(self, results: List[ExperimentResult]) -> int:
        """
        Processes batch results and returns number of successful experiments
        """
        successful = 0

        for result in results:
            if result.success:
                # Store results
                await self.results_manager.store_experiment(result)

                # Update optimizer
                self.optimizer.update_model(result)

                successful += 1
            else:
                # Handle failed experiment
                await self._handle_failed_experiment(result.experiment_id)

        return successful

    async def _handle_failed_experiment(self, experiment_id: str):
        """
        Handles failed experiments, implementing retry logic
        """
        self.retry_count[experiment_id] = self.retry_count.get(experiment_id, 0) + 1

        if self.retry_count[experiment_id] <= self.max_retries:
            self.logger.warning(
                f"Retrying experiment {experiment_id} "
                f"(attempt {self.retry_count[experiment_id]})"
            )
            self.experiment_status[experiment_id] = ExperimentStatus.QUEUED
        else:
            self.logger.error(
                f"Experiment {experiment_id} failed after {self.max_retries} attempts"
            )
            # Could implement fallback strategy here

    async def _handle_stage_transition(self):
        """
        Handles transition between optimization stages
        """
        old_stage = self.stage_manager.current_stage
        new_stage = self.stage_manager.transition_to_next_stage()

        self.logger.info(f"Transitioning from stage {old_stage} to {new_stage}")

        # Update optimization parameters for new stage
        stage_config = self.stage_manager.get_stage_parameters()
        self.optimizer.update_acquisition_strategy(stage_config)

        # Generate stage transition report
        await self.results_manager.store_stage_transition(old_stage, new_stage)

    def _check_convergence(self) -> bool:
        """
        Checks if optimization has converged based on current results
        """
        # Implement convergence criteria here
        # Could consider:
        # - Improvement rate
        # - Parameter stability
        # - Achievement of targets
        return False
