# Dynamic Bayesian Network & Unscented Kalman Filter for Predictive Maintenance

This directory contains implementations of probabilistic models for Remaining Useful Life (RUL) prediction on the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) turbofan engine degradation dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
  - [Dynamic Bayesian Network (DBN)](#dynamic-bayesian-network-dbn)
  - [Unscented Kalman Filter (UKF)](#unscented-kalman-filter-ukf)
- [Performance Results](#performance-results)
- [Key Findings](#key-findings)
- [Notebooks Description](#notebooks-description)

## Overview

This project explores two distinct probabilistic modeling approaches for predictive maintenance:

1. **Dynamic Bayesian Networks (DBN)**: A graphical model that captures temporal and instantaneous dependencies between operational settings and sensor readings to predict future sensor states.

2. **Unscented Kalman Filter (UKF)**: A state-space model that tracks engine health degradation over time using an exponential decay function, enabling RUL prediction through extrapolation.

Both approaches aim to predict when a turbofan engine will fail, enabling proactive maintenance scheduling.

## Dataset

**CMAPSS FD001**: Turbofan engine degradation simulation data
- **Training Set**: 100 engines run to failure
- **Test Set**: 100 engines with partial run-to-failure data
- **Features**: 21 sensors + 3 operational settings
- **Target**: Remaining Useful Life (RUL) in cycles

### Preprocessing
- Removed constant features: `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`, `op_setting_3`
- Normalized features using Min-Max scaling
- Created rolling statistics (5-cycle mean and slope) for trend analysis
- Discretized continuous features into 5 bins using quantile binning (for DBN)

### Health State Categories
- **Healthy**: RUL > 100 cycles (10,131 instances)
- **Degradation**: 50 ≤ RUL ≤ 100 cycles (5,100 instances)
- **Critical**: RUL < 50 cycles (5,000 instances)

## Models Implemented

### Dynamic Bayesian Network (DBN)

#### Model Architecture
A DBN represents the evolution of a system over discrete time steps using two types of dependencies:

**Structure**:
- **Temporal Edges** (17 edges): Link each feature from time `t` to `t+1`, capturing autocorrelation
  - Example: `sensor_11(t) → sensor_11(t+1)`
- **Instantaneous Edges** (60 edges): Model dependencies within the same time slice
  - Example: `op_setting_1(t) → sensor_2(t)`

**Total**: 77 edges connecting 34 nodes (17 features × 2 time slices)

#### Learning & Inference
- **Parameter Learning**: Fitted 34 Conditional Probability Distributions (CPDs) on 20,531 transition samples using Maximum Likelihood Estimation
- **Inference Method**: Converted to static Bayesian Network and used `VariableElimination` for probabilistic queries
- **Library**: `pgmpy` (Python package for Bayesian Networks)

#### DBN Workflow
1. Discretize sensor readings into 5 bins
2. Create 2-time-slice training pairs: `(t, t+1)`
3. Learn CPDs from data
4. Query `P(sensor(t+1) | all_features(t))` for prediction

### Unscented Kalman Filter (UKF)

#### Model Overview
The UKF is a recursive Bayesian filter that estimates the hidden health state of an engine from noisy sensor observations.

**State Space Model**:
```
State Equation (Process Model):
  x(t+1) = x(t) - decay_rate * dt
  where decay_rate = baseline * exp(curvature * (1 - x(t)))

Measurement Equation (Observation Model):
  z(t) = [x(t), x(t)]  (observing health state through 2 PCA components)
```

**Key Components**:
- **State Variable**: `x(t)` - normalized health state (1.0 = healthy, 0.0 = failed)
- **Observations**: `z(t)` - 2 principal components derived from 14 sensors
- **Process Noise**: `Q` - uncertainty in health degradation
- **Measurement Noise**: `R` - sensor observation uncertainty

#### Mark I: Basic Implementation
- **Observations**: 2 raw sensors (sensor_2, sensor_3)
- **Decay Model**: Linear degradation (`-0.005` per cycle)
- **Results**: Poor performance due to oversimplification

#### Mark II: Enhanced with PCA
- **Feature Engineering**: Applied PCA to 14 sensors → 2 principal components
  - Sensors used: 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21
  - PC alignment: Flipped to ensure positive correlation with RUL
- **Decay Model**: Exponential degradation capturing accelerating failure
  ```python
  exponent = clip(curvature * (1 - x), -20, 10)
  decay_rate = baseline * exp(exponent)
  ```
- **Hyperparameter Tuning**: Grid search on validation set (engines 81-90)
  - Baseline: [0.001, 0.002, 0.003, 0.004]
  - Curvature: [2.0, 3.0, 3.5, 4.0, 4.5]
  - **Optimal**: baseline = 0.003, curvature = 2.0 (validation score: 9.12)

#### UKF Workflow
1. Apply PCA to reduce 14 sensors to 2 principal components
2. Initialize filter: `x(0) = 1.0` (healthy state)
3. For each cycle, predict and update:
   - **Predict**: Propagate state using exponential decay model
   - **Update**: Correct prediction using sensor observations
4. At test cutoff, extrapolate to failure threshold (x = 0.0)
5. Count cycles until failure = Predicted RUL

#### Noise Matrix Learning
- **Rauch-Tung-Striebel (RTS) Smoother**: Backward pass to refine state estimates
- **Expectation-Maximization (EM)**: Iteratively update `Q` and `R` matrices
  - 15 epochs with convergence tolerance of 1e-5
  - Exponentially weighted moving average (α = 0.2)

## Performance Results

### Dynamic Bayesian Network (DBN1)

| Metric | Value |
|--------|-------|
| **Task** | Sensor value prediction (sensor_11) |
| **Last 5 Cycles Accuracy** | 60% (3/5 correct) |
| **Full Trajectory Accuracy** | ~47-58% (varies by unit) |
| **Selected Units** | 22, 68, 77 |
| **Mean Accuracy (100 units)** | ~52.6% |
| **Standard Deviation** | ~8.4% |

**Observations**:
- High variability across engines (best: 70%, worst: 34%)
- Discrete binning loses fine-grained information
- Temporal correlations captured but limited by discretization

**Correlation Analysis** (Top Sensors with RUL):
| Sensor | Correlation |
|--------|------------|
| sensor_11 | -0.696 |
| sensor_4 | -0.679 |
| sensor_12 | +0.672 |
| sensor_7 | +0.657 |
| sensor_15 | -0.643 |

### Unscented Kalman Filter (DBN3 - Mark II)

| Metric | Test Set | Training Set* |
|--------|----------|---------------|
| **RMSE** | 35.85 | 63.60 |
| **CMAPSS Score** | 3,311.77 | 90,500,546.75 |

*Training set evaluated with random truncation points

#### CMAPSS Scoring Function
The CMAPSS score penalizes late predictions more heavily than early predictions:
```
s = Σ penalty(error)
where:
  penalty(d) = exp(-d/13) - 1  if d < 0 (early prediction)
  penalty(d) = exp(d/10) - 1   if d > 0 (late prediction)
```

**Top 5 Challenging Engines** (Highest Penalties):
| Engine | True RUL | Predicted RUL | Error | Penalty |
|--------|----------|---------------|-------|---------|
| 68 | 8 | 115 | +107 | 44,354.86 |
| 34 | 7 | 113 | +106 | 40,133.84 |
| 76 | 10 | 115 | +105 | 36,314.50 |
| 81 | 8 | 109 | +101 | 24,342.01 |
| 31 | 8 | 108 | +100 | 22,025.47 |

**Analysis**: The model struggles with engines that fail unexpectedly early, resulting in dangerously late predictions.

### Model Comparison

| Aspect | DBN | UKF (Mark II) |
|--------|-----|---------------|
| **Approach** | Discrete probabilistic graphical model | Continuous state-space filtering |
| **Output** | Sensor state probabilities | RUL estimate (cycles) |
| **Strengths** | Captures complex dependencies | Handles noisy data, tracks trends |
| **Weaknesses** | Information loss from discretization | Requires good process model |
| **Interpretability** | High (CPDs show relationships) | Medium (hidden state + physics) |
| **Performance** | ~53% bin accuracy | 35.85 RMSE |

## Key Findings

### Data Insights
1. **Constant Features**: 7 features showed no variance in FD001 dataset
2. **Strong Predictors**: Sensors 11, 4, 12, 7, 15 show |correlation| > 0.64 with RUL
3. **Temporal Autocorrelation**: sensor_14 (r ≈ 0.97) and sensor_9 (r ≈ 0.96) are highly stable across time steps
4. **PCA Effectiveness**: 2 principal components capture major degradation patterns from 14 sensors

### Model Insights

**DBN**:
- Effective for understanding sensor relationships
- Limited by discrete representation
- Better for diagnostic tasks than prognostic tasks
- Computational complexity grows with network size

**UKF**:
- Superior for RUL prediction when process model is well-specified
- Exponential decay model captures accelerating degradation
- PCA dimensionality reduction improves generalization
- Sensitive to hyperparameters (baseline, curvature)
- Fails catastrophically on early-failure engines

### Failure Modes
Both models struggle with:
- **Sudden failures**: Engines with RUL < 20 cycles at test cutoff
- **Outlier degradation patterns**: Units that deviate from training distribution
- **Cold start problem**: Limited initial observations lead to poor state estimation

## Notebooks Description

### DBN1.ipynb
Primary DBN implementation with complete pipeline:
- Data loading and preprocessing
- Feature discretization (5 bins)
- DBN structure definition (temporal + instantaneous edges)
- Parameter learning (CPDs)
- Inference and evaluation on test units
- Visualization of network structure and prediction trajectories

### DBN2.ipynb
Exploratory data analysis and feature engineering:
- Rolling statistics (5-cycle windows)
- Health state categorization
- Lagged correlation analysis (t vs t+1)
- Identification of constant features
- Temporal dependency heatmaps

### DBN3.ipynb
Advanced UKF implementation with two marks:
- **Mark I**: Baseline with 2 raw sensors
- **Mark II**: Enhanced with PCA and exponential decay
- Hyperparameter grid search
- EM-based noise learning (Q, R matrices)
- RTS smoothing for improved state estimates
- Comprehensive evaluation with error analysis

### DBN4.ipynb
Variant of DBN3 with modified extrapolation changepoint:
- Changed extrapolation limit from 500 to 120 cycles
- Training set evaluation with random truncation
- Demonstrates overfitting issues on training data

## Visualization Highlights

### DBN Structure
A bipartite graph showing 17 features across 2 time slices (t=0 and t=1) with:
- Blue nodes: time t=0
- Green nodes: time t=1
- Temporal edges: vertical connections (feature persistence)
- Instantaneous edges: horizontal connections (within-time dependencies)

### UKF Degradation Trajectory
Example for a single engine showing:
- Blue line: UKF-tracked health state from observations
- Red dashed line: Extrapolated degradation using process model
- Black dashed line: Failure threshold (0.2)
- Gray vertical line: End of test data

### Performance Scatter Plots
- **True vs Predicted RUL**: Diagonal line = perfect prediction
- **Error Distribution**: Histogram showing prediction bias
- Most predictions cluster around true values with outliers causing large CMAPSS penalties

## Usage Notes

### Dependencies
```
pandas
numpy
sklearn
pgmpy (for DBN)
filterpy (for UKF)
matplotlib
seaborn
```

### Data Path
Update `path` variable to point to CMAPSS data directory containing:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

### Reproducibility
- DBN results are deterministic (no random seed required)
- UKF results depend on initial noise matrices but converge reliably
- Random truncation evaluation (DBN4) requires `np.random.seed(42)`

## Future Work

1. **Hybrid Approach**: Combine DBN for sensor prediction with UKF for RUL estimation
2. **Deep Learning Integration**: Replace PCA with autoencoder for feature extraction
3. **Multi-Engine DBN**: Model dependencies between multiple engines
4. **Adaptive Thresholding**: Dynamic failure threshold based on operational context
5. **Changepoint Detection**: Identify regime shifts in degradation patterns
6. **Ensemble Methods**: Average predictions from multiple models to reduce variance
