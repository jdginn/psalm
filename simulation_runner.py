from dataclasses import dataclass
from typing import Dict, Optional
import logging
from pathlib import Path
import json
import yaml
import subprocess
from typing import Tuple
import shutil
import os

import config
import models


def load_json_data(file_path: str) -> dict:
    """Load and parse JSON data from file."""
    with open(file_path, "r") as f:
        return json.load(f)


@dataclass
class SimulationConfig:
    """
    Configuration for a single simulation run.
    This would be specific to your acoustic simulation needs.
    """

    working_directory: Path
    output_directory: Path
    timeout_seconds: int = 3600  # 1 hour default timeout


class SimulationRunner:
    """
    Manages execution of acoustic simulations.

    Responsibilities:
    1. Convert optimization parameters to simulation inputs
    2. Execute simulations
    3. Parse simulation outputs into ExperimentResult format
    4. Handle simulation timeouts and failures
    """

    def __init__(
        self,
        program_path: Path,
        base_directory: Path,
        config_path: Path,
        timeout: int = 300,
    ):
        self.program_path = Path(program_path)
        self.base_directory = Path(base_directory)
        self.config_path = Path(config_path)
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def run_simulation(self, name: str, params: Dict[str, float]) -> Tuple[str, dict, bool]:
        """
        Runs a single simulation with the given parameters.

        Args:
            parameters: ExperimentParameters containing both group1 and group2 parameters

        Returns:
            ExperimentResult containing FOMs and metadata

        Raises:
            SimulationError: If simulation fails or times out
        """
        # Create experiment directory
        exp_dir = self._create_experiment_directory(name)

        try:
            # Prepare input files
            input_files = self._prepare_input_files(params, exp_dir)

            # Run the actual simulation
            # This is where you'll implement the interface to your simulator
            simulation_success = self._execute_simulation(input_files, exp_dir)

            if not simulation_success:
                return (name, {}, False)
                raise SimulationError("Simulation failed to complete successfully")

            # Parse results
            results = self._parse_simulation_results(exp_dir)

            return (name, results, True)

        except Exception as e:
            self.logger.error(f"Simulation failed for experiment {name}: {str(e)}")
            raise SimulationError(f"Simulation failed: {str(e)}")

        finally:
            # Cleanup if needed
            self._cleanup_experiment(exp_dir)

    def _create_experiment_directory(self, experiment_id: str) -> Path:
        """
        Creates and returns a directory for the experiment
        """
        exp_dir = self.base_directory / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _prepare_input_files(self, parameters: dict, exp_dir: Path) -> Dict[str, Path]:
        """
        Converts optimization parameters into simulation input files

        You'll need to implement this based on your simulation's input format
        """


        def copy_file(source_path, config_path, destination_path):

            # Check if the source_path is absolute
            if os.path.isabs(str(source_path)):
                if not os.path.exists(str(source_path)):
                    raise FileNotFoundError(f"File not found at absolute path: {source_path}")
            else:
                # Get the directory of spec_path and update source_path
                config_directory = os.path.dirname(config_path)
                source_path = os.path.join(config_directory, source_path)
                if not os.path.exists(source_path):
                    raise FileNotFoundError(f"File not found at relative path: {config_directory}/{source_path}")


            # Ensure the destination directory exists
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)

            # Copy the file
            shutil.copyfile(source_path, os.path.join(destination_path, os.path.basename(source_path)))


        spec = config.load_from_file(str(self.config_path))

        # Call the function to copy the file
        copy_file(spec.input.mesh.path, self.config_path, exp_dir)
        if spec.materials.from_file != "":
            copy_file(spec.materials.from_file, self.config_path, exp_dir)

        # Define the mapping of parameter names to their attribute paths
        PARAM_MAPPING = {
            "shot_count": "simulation.shot_count",
            "distance_from_front": "listening_triangle.distance_from_front",
            "distance_from_center": "listening_triangle.distance_from_center",
            "source_height": "listening_triangle.source_height",
            "ceiling_center_height": "ceiling_panels.center.height",
            "ceiling_center_width": "ceiling_panels.center.width",
            "ceiling_center_xmax": "ceiling_panels.center.xmax",
            "ceiling_center_xmin": "ceiling_panels.center.xmin",
            "wall_absorbers_Hall_B": "wall_absorbers.heights|Hall B",
            "wall_absorbers_Street_A": "wall_absorbers.heights|Street A",
            "wall_absorbers_Door_Side_A": "wall_absorbers.heights|Door Side A",
            "wall_absorbers_Hall_E": "wall_absorbers.heights|Hall E",
            "wall_absorbers_Street_D": "wall_absorbers.heights|Street D",
            "wall_absorbers_Street_B": "wall_absorbers.heights|Street B",
            "wall_absorbers_Door_Side_B": "wall_absorbers.heights|Door Side B",
            "wall_absorbers_Entry_Back": "wall_absorbers.heights|Entry Back",
            "wall_absorbers_Street_C": "wall_absorbers.heights|Street C",
            "wall_absorbers_Street_E": "wall_absorbers.heights|Street E",
            "wall_absorbers_Hall_A": "wall_absorbers.heights|Hall A",
            "wall_absorbers_Entry_Front": "wall_absorbers.heights|Entry Front",
            "wall_absorbers_Door": "wall_absorbers.heights|Door",
            "wall_absorbers_Back_A": "wall_absorbers.heights|Back A",
            "wall_absorbers_Back_B": "wall_absorbers.heights|Back B",
        }

        def apply_parameters(spec, parameters):
            # Only process parameters that are actually provided
            for param_name in parameters.keys() & PARAM_MAPPING.keys():
                value = parameters[param_name]
                path = PARAM_MAPPING[param_name]
                
                # First check if we have a dictionary access
                if '|' in path:
                    attr_path, dict_key = path.split('|', 1)
                    # Handle the attribute path
                    obj = spec
                    path_parts = attr_path.split('.')
                    for part in path_parts:
                        obj = getattr(obj, part)
                    # Finally set the dictionary key
                    obj[dict_key] = value
                else:
                    # Original attribute-only logic
                    obj = spec
                    *path_parts, final_attr = path.split('.')
                    for part in path_parts:
                        obj = getattr(obj, part)
                    setattr(obj, final_attr, value)


        # def apply_parameters(spec, parameters):
        #     # Only process parameters that are actually provided
        #     for param_name in parameters.keys() & PARAM_MAPPING.keys():
        #         value = parameters[param_name]
        #         # Split path and set nested attribute
        #         obj = spec
        #         *path_parts, final_attr = PARAM_MAPPING[param_name].split('.')
        #         for part in path_parts:
        #             obj = getattr(obj, part)
        #         setattr(obj, final_attr, value)

        # spec.listening_triangle.distance_from_front = parameters["distance_from_front"]
        # spec.listening_triangle.distance_from_center = parameters[
        #     "distance_from_center"
        # ]
        # spec.listening_triangle.source_height = parameters["source_height"]

        apply_parameters(spec, parameters)

        spec_file = exp_dir / "config.yaml"

        config.save_to_file(spec, str(spec_file))

        return {"config": spec_file}

    def _execute_simulation(self, input_files: Dict[str, Path], exp_dir: Path) -> bool:
        """
        Actually runs the simulation

        This is where you'll implement the interface to your simulator.
        Returns True if simulation completed successfully.
        """

        # Run simulation
        result = subprocess.run(
            [str(self.program_path), "simulate", input_files["config"], exp_dir],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            # Print program output
            if result.stdout:
                print("Program stdout:", result.stdout)
            if result.stderr:
                print("Program stderr:", result.stderr)
        return True

    def _parse_simulation_results(self, exp_dir: Path) -> Dict:
        """
        Parses simulation output files into FOMs

        You'll need to implement this based on your simulation's output format
        """

        data = load_json_data(str(exp_dir / "summary.json"))

        results = models.deserialize_analysis_results(data)
        return data

    def _cleanup_experiment(self, exp_dir: Path):
        """
        Performs any necessary cleanup after simulation

        Implement based on your needs - you might want to:
        - Remove temporary files
        - Compress results
        - Move files to long-term storage
        """
        pass


class SimulationError(Exception):
    """Custom exception for simulation failures"""

    pass
