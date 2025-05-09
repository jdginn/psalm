# psalm

Visualization and optimization tools for recording studio acoustic simulations created with [go-recording-studio](https://github.com/jdginn/go-recording-studio).

## What it does

This toolkit provides two main functionalities for acoustic studio design:

1. 3D visualization of acoustic simulation results, helping studio designers:

   - View reflection paths and their interaction with room surfaces
   - Identify problematic reflection points
   - Visualize the Reflection-Free Zone (RFZ)
   - Analyze reflection patterns with reduced visual noise

2. Bayesian optimization of studio parameters, enabling:
   - Automated search for optimal speaker and listening positions
   - Multi-process parameter optimization
   - Customizable optimization objectives
   - Constraint-based parameter exploration

## How it works

### Visualization (main.py)

The visualization tool processes simulation results from go-recording-studio and provides interactive 3D views using trimesh:

1. Load simulation results from a go-recording-studio output directory
2. Render the room geometry, reflection paths, and RFZ
3. Support interactive viewing with zoom, pan, and rotate capabilities
4. Provide specialized viewing modes:
   - Step mode (`--step`): View individual reflection paths sequentially
   - Culling mode (`--cull`): Group similar reflections for clearer visualization

### Optimization (acoustic_optimizer.py)

The optimizer uses Bayesian optimization to find optimal studio configurations:

1. Define parameter bounds and constraints
2. Run parallel simulations across multiple processes
3. Primary optimization goals:
   - Achieve minimum 30ms Initial Time Delay (ITD)
   - Minimize required absorptive surface area
4. Support for custom optimization objectives

## Getting Started

### Installation

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Prerequisites

- Python 3.x
- A completed simulation from [go-recording-studio](https://github.com/jdginn/go-recording-studio)

### Visualization Usage

Basic usage:

```bash
python main.py /path/to/experiment [options]
```

Available options:

```
--step         Step through reflections one at a time
--cull FLOAT   Cull very similar paths from the render
               (Values between 0.05 and 0.3 meters recommended)
--points       Show the location of the final reflection in each arrival
```

### Visualization Features

For each reflection path, the visualization displays:

- ITD (Initial Time Delay) in milliseconds
- Path gain in dB
- Shot gain in dB
- Number of reflections
- Surface name of last reflection

In step mode, navigation controls are:

- 'n': next reflection
- 'p': previous reflection
- 'q': quit

### File Format Documentation

The visualization tool reads two key files from the simulation output:

- `annotations.json`: Contains reflection paths and geometry data (see annotations_schema.md)
- `summary.json`: Contains acoustic parameters and results (see summary_schema.md)

### Optimization Usage

1. Create a configuration file based on the [example config](https://github.com/jdginn/go-recording-studio/blob/main/testdata/default_config.yaml)

2. Run the optimizer:

```bash
python acoustic_optimizer.py config.yaml
```

## License

[License information needed]

## Acknowledgments

- Uses [go-recording-studio](https://github.com/jdginn/go-recording-studio) for acoustic simulation
- Built with [trimesh](https://trimsh.org/) for 3D visualization
- Optimization powered by [scikit-optimize](https://scikit-optimize.github.io/)
