# Business Problem: Predictive Maintenance for Aircraft Engines

Aircraft engine maintenance is traditionally performed using **scheduled inspections** or **reactive replacement after failure**. Both strategies introduce inefficiencies:

- **Reactive maintenance** can lead to unexpected failures, operational disruptions, and safety risks.
- **Preventive maintenance** often replaces components too early, increasing operational costs.

Predicting **Remaining Useful Life (RUL)** allows operators to estimate how many operating cycles remain before an engine fails.

Accurate RUL prediction enables **predictive maintenance**, which helps:

- Schedule maintenance before catastrophic failure  
- Reduce unnecessary component replacements  
- Improve operational safety  
- Optimize maintenance planning  

This project uses the **NASA C-MAPSS turbofan degradation dataset** to build predictive models capable of estimating engine RUL from sensor measurements.

The dataset contains time-series measurements of multiple engines, including:

- **3 operational settings**
- **21 sensor variables**
- **Cycle-by-cycle degradation data**

The objective is to predict the number of cycles remaining before engine failure.

---

# Hierarchical Bayesian RUL Model

To address the RUL prediction problem, the first modeling approach was a **Hierarchical Bayesian Linear Regression model** that estimates RUL while also quantifying prediction uncertainty.

Unlike deterministic approaches, a Bayesian model produces **probabilistic predictions**, allowing us to evaluate not only the predicted RUL but also the **uncertainty associated with that prediction**.

This is particularly valuable in predictive maintenance, where **risk-aware decisions** are required.

## Model Overview

The model predicts Remaining Useful Life using the operational settings and sensor measurements prvided in the dataset.

### Feature Engineering

The model incorporates:

- Engine cycle information
- Operational settings (`setting_1–3`)
- Sensor measurements (`s1–s21`)
- First-order sensor differences (`sensor_diff`)

Sensor differences capture **degradation dynamics between consecutive cycles**, which helps the model detect early patterns of engine deterioration.

### Target Transformation

Remaining Useful Life often exhibits **heteroskedasticity and long-tailed behavior** at high values.

To stabilize variance and improve posterior sampling, the target variable is transformed as:

$$
y = \log(1 + RUL)
$$

## Bayesian Model Specification

The model assumes a Student-t likelihood:

$$
y_i \sim \text{StudentT}(\nu, \mu_i, \sigma)
$$

with linear predictor:

$$
\mu_i = \alpha + X_i \beta
$$

Where:

- $\alpha$ = global intercept  
- $\beta$ = regression coefficients  
- $\sigma$ = observation noise  
- $\nu$ = degrees of freedom controlling tail heaviness  

The Student-t distribution provides robustness to outliers and heavy-tailed residuals commonly observed in degradation data.

## Hierarchical Priors

To prevent overfitting and regularize the model, regression coefficients follow a hierarchical prior:

$$
\beta_j \sim \mathcal{N}(0, \tau^2)
$$

where the global shrinkage parameter is defined as:

$$
\tau \sim \text{HalfNormal}(1)
$$

Observation noise is modeled as:

$$
\sigma \sim \text{HalfNormal}(1)
$$

The Student-t degrees of freedom parameter follows:

$$
\nu \sim \text{Exponential}(0.1)
$$

This hierarchical shrinkage structure stabilizes coefficient estimates in high-dimensional sensor data.

## Posterior Inference

Model parameters are estimated using **Bayesian inference with the No-U-Turn Sampler (NUTS)**.

Posterior inference provides:

- Posterior parameter distributions  
- Posterior predictive distributions  
- Predictive uncertainty intervals  

These outputs allow the model to quantify uncertainty in RUL predictions.

## Evaluation Strategy

Model performance is evaluated under multiple perspectives.

### - Row-Level Validation

Prediction accuracy across validation observations using:

- RMSE  
- MAE  
- Bias  
- NASA Asymmetric Score  

### - Engine-Level Evaluation

In real predictive maintenance settings, decisions are made **once per engine**.

Therefore the model is also evaluated using **only the final observed cycle per engine**.

This reflects how RUL predictions would be used operationally.

### - Degradation-Focused Evaluation

Following turbofan prognostics literature, evaluation also focuses on the degradation regime:

$$
RUL \leq 120
$$

Early-life cycles often contain weak degradation signals, making them difficult to distinguish from healthy operating conditions.

Filtering metrics to the degradation regime provides a more realistic measure of predictive performance.

### - NASA Asymmetric Score

The NASA scoring function penalizes **late predictions more heavily than early predictions**:

$$
Score =
\begin{cases}
e^{-error/10} - 1 & \text{if } error < 0 \\
e^{error/13} - 1 & \text{if } error \geq 0
\end{cases}
$$

where:

$$
error = \hat{RUL} - RUL
$$

Lower scores indicate better predictive performance.

## Summary

The **Hierarchical Bayesian modeling framework** that provides:

- probabilistic predictions  
- uncertainty quantification  
- hierarchical regularization  
- robust likelihood modeling  

These characteristics make Bayesian models particularly attractive for **risk-aware industrial applications**, where maintenance decisions must balance prediction accuracy and operational safety.

However, this approach also has important limitations:

- The model assumes a **linear mean structure**, which may not fully capture nonlinear degradation patterns.
- Temporal dependencies are only indirectly captured through engineered features (e.g., sensor differences).
- The model does not explicitly learn sequential degradation dynamics as deep learning architectures can.

Because turbofan degradation is inherently **temporal and nonlinear**, more expressive models may achieve improved predictive performance. For this reason, the project also explores additional modeling approaches.

---

