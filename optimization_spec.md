# Acoustic Simulation Optimization Specification

## Overview
This document specifies the design for a Bayesian Optimization approach to optimize acoustic simulation parameters. The goal is to maximize a primary figure of merit (ITD) while balancing secondary objectives and minimizing necessary parameters.

## Table of Contents
1. [Data Structure and Parameter Representation](#1-data-structure-and-parameter-representation)
2. [Initial Sampling Strategy](#2-initial-sampling-strategy)
3. [Surrogate Model Design](#3-surrogate-model-design)
4. [Acquisition Function Design](#4-acquisition-function-design)
5. [Parallel Evaluation Strategy](#5-parallel-evaluation-strategy)
6. [Multi-Stage Optimization Process](#6-multi-stage-optimization-process)
7. [Result Tracking and Validation](#7-result-tracking-and-validation)

## 1. Data Structure and Parameter Representation

### 1.1 Parameters

#### Group 1 Parameters (Monotonic)
All thickness parameters: 4 < thickness < 24
All height parameters: 0.7 < height < 1.6

Ceiling Panels:
- Center:
  - thickness
  - height
  - width
  - xmin
  - xmax
- Sides:
  - thickness
  - height
  - width
  - spacing
  - xmin
  - xmax

Wall Absorbers:
- thickness
- heights for locations:
  - Hall B
  - Street A
  - Door Side A
  - Hall E
  - Street D
  - Street B
  - Door Side B
  - Entry Back
  - Street C
  - Street E
  - Hall A
  - Entry Front
  - Door
  - Back A
  - Back B

#### Group 2 Parameters (Non-monotonic)
Listening Triangle:
- distance_from_front: 0.2 < value < 0.8
- distance_from_center: 0.6 < value < 2.0
- source_height: 0.7 < value < 2.0
- listen_height: fixed at 1.4 (not optimized)

### 1.2 Figures of Merit (FOMs)
1. Primary FOM:
   - ITD (maximize)
   - Target: 30
   - Current best known: 11

2. Secondary FOMs:
   - AVG_GAIN_5ms (minimize)
     - Target: -20
   - ITD_2 (maximize)
     - Target: 30

Relationships:
- During exploration phase, improvements in one FOM likely correlate with improvements in others
- Near optimum, trade-offs between FOMs become more significant

### 1.3 Experiment Storage
- Experiments stored in uniquely named directories with timestamps
  - Format example: "bold-haze-20250315-201654"
- Each experiment directory contains:
  - Full input parameters
  - Complete simulation results
- Summary format for tracking:
  ```
  <experiment_name>
      ITD: <value>
      AVG_GAIN_5ms: <value>
      ITD_2: <value>
  ```
- Parameter combinations and results to be read directly from experiment directories
- No additional metadata tracking required


## 2. Initial Sampling Strategy

### 2.1 Sample Size and Composition
- Start with N initial samples (suggest N = 24 for 3 batches of 8 parallel simulations)
- Distribution:
  - 1 known good solution
  - 1 variation of known good solution with reduced Group 1 parameters
  - 22 exploration samples

### 2.2 Sampling Methodology
1. Known Solution:
   - Include existing best-known solution (ITD = 11)
   - Serves as baseline for optimization

2. Reduced Parameter Variation:
   - Take best-known solution
   - Reduce all Group 1 parameters by X% (suggest X = 20)
   - Maintains same Group 2 parameters
   - Tests potential for parameter reduction

3. Exploration Samples:
   - Use Latin Hypercube Sampling (LHS) for parameter space coverage
   - Stratify sampling to ensure:
     * Even coverage of Group 2 parameters (critical for understanding non-monotonic relationships)
     * Varied combinations of Group 1 parameters
     * Mix of high and low values for all parameters

### 2.3 Implementation Notes
- Run samples in parallel batches of 8
- Verify validity of Group 2 parameters using existing model before simulation
- Store results in experiment directories following established format
- Track completion status of initial sampling phase

### 2.4 Success Criteria
- Complete set of N valid experiments
- Coverage metrics:
  * No significant gaps in Group 2 parameter space
  * Range of Group 1 parameter values tested
  * At least one result within 20% of known good solution

## 3. Surrogate Model Design

### 3.1 Model Architecture
- Use Gaussian Process (GP) regression as the base model
- Single unified model for all parameters to capture interactions
- Inputs:
  * All Group 1 parameters (normalized to [0,1] range)
  * All Group 2 parameters (normalized to [0,1] range)
- Outputs:
  * Primary FOM (ITD)
  * Secondary FOMs (AVG_GAIN_5ms, ITD_2)

### 3.2 Kernel Selection
- Primary kernel: Matérn 5/2 kernel
  * Provides balance between smoothness and flexibility
  * Suitable for physical parameters
  * Better suited than RBF for non-monotonic relationships
- Kernel hyperparameters:
  * Separate length scales for each parameter
  * Automatically tuned during model fitting
  * Initial length scales set based on parameter ranges

### 3.3 Model Training
- Frequency: Update after each batch of experiments
- Training process:
  1. Normalize all input parameters to [0,1]
  2. Fit separate GP for each FOM
  3. Optimize hyperparameters using maximum likelihood estimation
  4. Validate model predictions using leave-one-out cross-validation

### 3.4 Uncertainty Handling
- Track predictive uncertainty for each FOM
- Use uncertainty estimates to:
  * Guide exploration in acquisition function
  * Identify regions needing additional sampling
  * Detect potential model misspecification

### 3.5 Model Validation
- Cross-validation metrics:
  * R² score
  * Mean squared error
  * Root mean squared error
- Validation checks:
  * Uncertainty estimates should be larger in unexplored regions
  * Predictions should be accurate near known good solutions
  * Model should capture known parameter relationships

### 3.6 Implementation Notes
- Cache model predictions for efficiency
- Retrain model completely after every N experiments
- Monitor for signs of model degradation
- Store model state for possible rollback

### 3.7 Questions for Discussion
1. Should we use different kernel functions for Group 1 vs Group 2 parameters?
2. Do we need separate models for each FOM, or should we use a multi-output GP?
3. Should we add any specific validation criteria for the model?
4. How frequently should we perform full model retraining?

## 4. Acquisition Function Design

### 4.1 Primary Acquisition Function
- Base: Upper Confidence Bound (UCB)
  * Balances exploration and exploitation
  * Formula: μ(x) + κ * σ(x)
  * κ: exploration parameter (starts high, decreases over time)
- Initial κ value: 5.0 (emphasizing exploration)
- Schedule for κ reduction:
  * Reduce by 0.5 every N experiments
  * Floor value: 1.0
  * Reset if uncertainty becomes too high in promising regions

### 4.2 Multi-Objective Handling
- Weighted sum approach for combining FOMs
- Primary weights:
  * ITD: 0.6
  * AVG_GAIN_5ms: 0.2
  * ITD_2: 0.2
- Normalize each FOM to [0,1] range using:
  * ITD: normalized by target (30)
  * AVG_GAIN_5ms: normalized by target (-20)
  * ITD_2: normalized by target (30)

### 4.3 Constraint Handling
- Soft constraints on Group 1 parameters
  * Penalty term increases as parameters grow
  * Penalty weight: starts low (0.1) and increases with optimization progress
- Hard constraints:
  * Enforce parameter bounds through parameter transformation
  * Reject invalid Group 2 parameter combinations before evaluation

### 4.4 Batch Selection
- Select 8 points per batch using:
  1. Select highest UCB point
  2. Update GP with fantasized outcome
  3. Repeat until batch is full
- Maintain minimum distance between selected points
- Include at least one point with reduced Group 1 parameters in each batch

### 4.5 Adaptive Strategies
- Increase exploration (κ) if:
  * No improvement in best FOM for N experiments
  * Uncertainty estimates become too small
- Adjust FOM weights if:
  * One FOM consistently underperforms
  * Trade-offs between FOMs become apparent
- Update penalty weights for Group 1 parameters based on:
  * Current best solution
  * Rate of improvement
  * Distance from targets

### 4.6 Implementation Notes
- Cache acquisition function evaluations
- Parallel computation of acquisition function
- Store acquisition function state for analysis
- Track exploration vs exploitation balance

### 4.7 Questions for Discussion
1. Should we consider alternative acquisition functions (e.g., Expected Improvement)?
2. How quickly should we reduce the exploration parameter κ?
3. Should we adjust the FOM weights dynamically?
4. Do we need additional strategies for handling the Group 1 parameter reduction objective?

## 5. Parallel Evaluation Strategy

### 5.1 Batch Processing Structure
- Fixed batch size: 8 simulations
- Batch composition:
  * 6-7 points from acquisition function
  * 1-2 points for parameter reduction exploration
- Run batches sequentially, simulations within batch in parallel

### 5.2 Experiment Management
- Unique experiment naming:
  * Format: "{adjective}-{noun}-{YYYYMMDD}-{HHMMSS}"
  * Example: "bold-haze-20250315-201654"
- Directory structure:
  * One directory per experiment
  * Standard format for input parameters
  * Consistent output structure
- Status tracking:
  * Track running experiments
  * Monitor completion status
  * Handle failed simulations

### 5.3 Resource Management
- CPU/Memory allocation:
  * 8 simultaneous simulations
  * Monitor resource usage
  * Prevent system overload
- Storage management:
  * Regular cleanup of temporary files
  * Archive completed experiments
  * Maintain result summaries

### 5.4 Fault Tolerance
- Handling failed simulations:
  * Retry failed experiments once
  * Log failures for analysis
  * Replace consistently failing points
- Recovery strategy:
  * Save state after each batch
  * Ability to resume from last completed batch
  * Regular backup of optimization state

### 5.5 Results Processing
- Real-time processing:
  * Calculate FOMs as simulations complete
  * Update model with new results
  * Track optimization progress
- Batch completion:
  * Validate all results
  * Update surrogate model
  * Generate batch summary

### 5.6 Implementation Notes
- Use async/await pattern for batch management
- Implement proper cleanup on interruption
- Maintain experiment metadata
- Track computation time and resource usage

### 5.7 Questions for Discussion
1. Should we implement dynamic batch sizing?
2. Do we need additional fault tolerance mechanisms?
3. Should we add more sophisticated resource monitoring?
4. How should we handle partial batch completion?

## 6. Multi-Stage Optimization Process

### 6.1 Optimization Stages

#### Stage 1: Initial Exploration (First 24 experiments)
- Focus on broad parameter space coverage
- High exploration coefficient (κ = 5.0)
- Equal weight to all FOMs
- Minimal penalty on Group 1 parameter values
- Success criteria:
  * Complete coverage of parameter space
  * At least one result within 20% of known good solution

#### Stage 2: Focused Search (Experiments 25-72)
- Identify promising regions
- Gradually reduce exploration coefficient (κ = 3.0)
- Adjust weights based on FOM correlations
- Begin increasing penalties on Group 1 parameters
- Success criteria:
  * Improvement in primary FOM (ITD)
  * Identification of multiple promising regions

#### Stage 3: Local Optimization (Experiments 73-120)
- Focus on best regions from Stage 2
- Lower exploration coefficient (κ = 2.0)
- Increased weight on parameter reduction
- Strong penalties on unnecessary Group 1 parameters
- Success criteria:
  * Surpass known good solution
  * Identify parameter reduction opportunities

#### Stage 4: Refinement (Experiments 121+)
- Fine-tune best solutions
- Minimal exploration (κ = 1.0)
- Maximum emphasis on parameter reduction
- Validate solution robustness
- Success criteria:
  * Stable, reproducible results
  * Minimal Group 1 parameter values
  * Meet or exceed all FOM targets

### 6.2 Stage Transitions
- Automatic transitions based on:
  * Number of experiments completed
  * Achievement of stage success criteria
  * Rate of improvement
- Manual override options for:
  * Extended exploration if needed
  * Early transition to next stage
  * Return to previous stage

### 6.3 Progress Tracking
- Key metrics per stage:
  * Best FOM values achieved
  * Parameter reduction progress
  * Exploration coverage
  * Model uncertainty
- Decision points:
  * Stage transition readiness
  * Need for additional exploration
  * Optimization completion

### 6.4 Implementation Notes
- Store stage information with results
- Track stage-specific parameters
- Maintain stage transition history
- Regular progress reports

### 6.5 Questions for Discussion
1. Should stages have flexible durations based on progress?
2. Do we need additional stages or different transition criteria?
3. How should we handle cases where progress stalls in a stage?
4. Should we add more sophisticated stage transition logic?

## 7. Result Tracking and Validation

### 7.1 Result Storage
- Primary storage:
  * Experiment directories with full simulation data
  * Optimization state database (SQLite)
  * Periodic state snapshots
- Data structure:
  ```python
  {
    'experiment_id': 'bold-haze-20250315-201654',
    'timestamp': '2025-03-15 20:16:54',
    'stage': 2,
    'parameters': {
        'group1': {...},
        'group2': {...}
    },
    'results': {
        'ITD': float,
        'AVG_GAIN_5ms': float,
        'ITD_2': float
    },
    'model_uncertainty': {...},
    'optimization_state': {...}
  }
  ```

### 7.2 Progress Monitoring
- Real-time metrics:
  * Best FOM values achieved
  * Parameter reduction progress
  * Model uncertainty evolution
  * Stage progression
- Visualization:
  * Parameter space coverage plots
  * FOM improvement trends
  * Uncertainty maps
  * Parameter correlation heatmaps

### 7.3 Validation Methods
- Solution validation:
  * Repeat best configurations
  * Verify parameter sensitivity
  * Cross-validate model predictions
- Statistical analysis:
  * Confidence intervals on FOMs
  * Parameter importance ranking
  * Correlation analysis
  * Outlier detection

### 7.4 Reporting
- Per-batch reports:
  * Experiment summaries
  * FOM improvements
  * Parameter trends
  * Model updates
- Stage completion reports:
  * Success criteria evaluation
  * Stage objectives review
  * Recommendations for next stage
- Final optimization report:
  * Best configurations found
  * Parameter reduction achievements
  * Performance comparisons
  * Future recommendations

### 7.5 Archival
- Data retention:
  * All experiment results
  * Model states at stage transitions
  * Configuration history
  * Validation results
- Export formats:
  * JSON for configuration data
  * CSV for result summaries
  * PDF for reports
  * PNG for visualizations

### 7.6 Implementation Notes
- Automated backup system
- Version control for configurations
- Reproducibility checks
- Data integrity validation

### 7.7 Questions for Discussion
1. Should we implement real-time monitoring dashboards?
2. Do we need additional validation methods?
3. What additional metrics should we track?
4. How long should we retain full experiment data?

