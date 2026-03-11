# Hierarchical Bayesian Model for Remaining Useful Life (RUL) Prediction

## Overview

This project implements a **Hierarchical Bayesian Linear Regression model** to estimate the Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPSS dataset.

The objective is to predict the number of cycles remaining before engine failure while explicitly quantifying predictive uncertainty through Bayesian inference.

The model integrates:

- Operational settings
- Sensor measurements
- First-order sensor differences
- Cycle information
- Posterior uncertainty via MCMC sampling

---

## Dataset

We use the NASA CMAPSS turbofan engine degradation dataset:

- **Train set**: Engines run until failure.
- **Test set**: Engines truncated before failure.
- **RUL file**: True remaining cycles for test engines.

Each engine contains time-series measurements per cycle:

- 3 operational settings
- 21 sensor variables

RUL is computed as:

$$
RUL = \text{failure cycle} - \text{current cycle}
$$

---

## Target Transformation

RUL exhibits heteroskedasticity and long-tailed behavior at high values.

To stabilize variance and improve posterior geometry, we apply:

$$
y = \log(1 + RUL)
$$

The transformed target is standardized:

$$
y_{std} = \frac{y - \mu_y}{\sigma_y}
$$

After prediction, we invert the transformation:

$$
\hat{RUL} = \exp(y_{std} \cdot \sigma_y + \mu_y) - 1
$$

This improves sampling efficiency and numerical stability.

---

## Feature Engineering

The model uses:

- `cycle`
- `setting_1`, `setting_2`, `setting_3`
- `s1` to `s21`
- First differences of each sensor (`sX_diff`)

Sensor differences help capture degradation dynamics.

All features are standardized using training statistics only.

---

## Model Specification

We implement a Hierarchical Bayesian Linear Regression model.

### Likelihood

We assume a Student-t observation model:

$$
y_i \sim \text{StudentT}(\nu, \mu_i, \sigma)
$$

with linear mean:

$$
\mu_i = \alpha + X_i \beta
$$

Where:

- $\alpha$ = global intercept  
- $\beta$ = regression coefficients  
- $\sigma$ = scale parameter  
- $\nu$ = degrees of freedom (controls tail heaviness)  

The Student-t likelihood provides robustness to outliers and heavy-tailed residual behavior, which is common in RUL prediction problems.

---

### Prior Distributions

We specify weakly-informative priors:

Intercept:

$$
\alpha \sim \mathcal{N}(0, 1)
$$

Regression coefficients:

$$
\beta_j \sim \mathcal{N}(0, \tau^2)
$$

Global shrinkage parameter:

$$
\tau \sim \text{HalfNormal}(1)
$$

Observation noise:

$$
\sigma \sim \text{HalfNormal}(1)
$$

Degrees of freedom parameter:

$$
\nu \sim \text{Exponential}(1/10)
$$

---

### Hierarchical Structure

The prior on coefficients induces partial pooling:

$$
\beta_j \sim \mathcal{N}(0, \tau^2)
$$

This hierarchical shrinkage:

- Prevents overfitting  
- Regularizes coefficients  
- Improves stability in high-dimensional settings  

---

## Posterior Inference

Inference is performed using **MCMC (NUTS sampler)**.

We obtain:

- Posterior parameter distributions
- Posterior predictive distributions
- 90% predictive intervals
- Uncertainty quantification

Convergence is assessed using:

- Trace plots
- Posterior summaries
- Energy diagnostics (BFMI)

---

## Evaluation Strategy

Performance is evaluated under three regimes:

### 1. Row-Level Validation

Internal validation split using:

- RMSE
- MAE
- Bias
- NASA Asymmetric Score

---

### 2. Engine-Level Evaluation (Last Cycle)

One prediction per engine at the final observed cycle.

This reflects real operational decision-making.

---

### 3. Degradation-Regime Evaluation (RUL ≤ 120)

Following turbofan literature, evaluation is also performed for:

$$
RUL \leq 120
$$

Rationale:

- Early-life cycles contain weak degradation signal.
- Large RUL values are difficult to distinguish from sensor data.
- Performance in degradation regime is operationally more relevant.

Important:

The model is trained on full data.  
Filtering is applied only when computing metrics.

---

## NASA Asymmetric Score

NASA's scoring function penalizes late predictions more heavily:

$$
Score =
\begin{cases}
e^{-error/10} - 1 & \text{if } error < 0 \\
e^{error/13} - 1 & \text{if } error \ge 0
\end{cases}
$$

Lower values indicate better performance.

---

## Outputs

The notebook exports:

- `train_rul_predictions.csv`
- `test_rul_predictions.csv`

Each file includes:

- Unit ID
- Cycle
- True RUL
- Posterior mean prediction
- Absolute error
- 5th percentile prediction
- 95th percentile prediction

---

## Strengths

- Fully probabilistic modeling framework
- Uncertainty quantification
- Hierarchical shrinkage regularization
- Degradation-focused evaluation
- Engine-level predictive assessment

---

## Limitations

- Linear mean structure
- Does not explicitly model nonlinear degradation dynamics
- Does not incorporate sequential memory (e.g., LSTM/BRNN)

---

## Future Work

- Nonlinear mean functions
- Gaussian Process regression
- Bayesian Neural Networks
- Change-point modeling
- Sequential deep learning models (BRNN, LSTM)

---

## Author

MS in Applied Data Science  
University of Chicago  

Hierarchical Bayesian Modeling Project
