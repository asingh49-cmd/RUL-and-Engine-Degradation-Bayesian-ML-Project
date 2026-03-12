# Prophet-Based Remaining Useful Life (RUL) Prediction

> **Course**: Bayesian Machine Learning & Generative AI — Group Project  
> **University**: University of Chicago  
> **Module Owner**: William  
> **Dataset**: NASA C-MAPSS Turbofan Engine Degradation (FD001)

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Group Project Context](#group-project-context)
- [Data Engineering & Preprocessing](#data-engineering--preprocessing)
- [Model Architecture](#model-architecture)
- [Loss Function & Training Strategy](#loss-function--training-strategy)
- [Training Curriculum](#training-curriculum)
- [Key Discoveries & Insights](#key-discoveries--insights)
- [Other Core Components of Prophet](#other-core-components-of-prophet)
- [Final Results](#final-results)
- [Dependencies & Usage](#dependencies--usage)
- [File Structure](#file-structure)
- [References](#references)

---

## Executive Summary

This module applies **Meta's Prophet** — a Bayesian structural time-series model — to the task of **Remaining Useful Life (RUL) prediction** for aircraft turbofan engines, using the NASA C-MAPSS FD001 benchmark dataset.

Prophet was originally designed for calendar-driven business forecasting. Here, it is repurposed as a **Bayesian regression framework** by disabling all seasonality components and driving predictions entirely through 18 sensor and operational-setting regressors. Two growth modes are explored and compared:

| Model | Prophet Growth Mode | Core Idea |
|-------|---------------------|-----------|
| **Linear Trend** | `linear` | Piece-wise linear degradation trend; predictions clipped to `[0, 125]` post-inference |
| **Logistic Trend** | `logistic` | Saturating growth with explicit floor and cap enforces bounded RUL predictions natively |

The work sits within a broader **Bayesian ML group project** at the University of Chicago, where five methods are compared side-by-side on the same dataset.

---

## Group Project Context

| Member | Method |
|--------|--------|
| Lalo | Hierarchical Bayesian Models |
| Mukul | Dynamic Bayesian Networks (DBN) |
| Adi | Bayesian Time Series |
| **William** | **Prophet (this module)** |
| Ryan | Bayesian Recurrent Neural Networks (BRNN) |

**Dataset shared across all modules**: NASA C-MAPSS FD001 — Turbofan Engine Degradation Simulation.

---

## Data Engineering & Preprocessing

### Raw Data

Three space-delimited text files constitute the dataset:

| File | Role | Shape |
|------|------|-------|
| `train_FD001.txt` | Run-to-failure records for training engines | 20,631 rows × 26 cols |
| `test_FD001.txt` | Truncated cycles for 100 test engines | ~13,096 rows × 26 cols |
| `RUL_FD001.txt` | True remaining cycles for each test engine | 100 rows × 1 col |

Columns parsed and named as:

```
unit_id, cycle, op_setting_1/2/3, s_1 … s_21
```

> Only the first 26 columns are retained (trailing NaN columns dropped).

### RUL Label Construction

Ground-truth RUL labels are derived from the training data:

$$\text{RUL}_{i,t} = \max(\text{cycle})_i - \text{cycle}_t$$

### RUL Capping

A piece-wise constant health assumption is applied: engines with very high remaining life are functionally in a "new" state and contribute no useful degradation signal. RUL is therefore clipped at:

$$\text{RUL} = \min(\text{RUL},\ 125)$$

This `RUL_CAP = 125` is used consistently as the saturation ceiling in both models.

### Feature Selection

21 raw sensors are reduced to **14 informative sensors** based on C-MAPSS domain knowledge (sensors with near-zero variance across all operating conditions are dropped). Combined with 3 operational settings and the cycle counter, this yields **18 input features**:

```python
sensor_cols  = ["s_2","s_3","s_4","s_7","s_8","s_9",
                "s_11","s_12","s_13","s_14","s_15","s_17","s_20","s_21"]
base_cols    = ["cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
feature_cols = base_cols + sensor_cols   # 18 features
```

### Normalization

All 18 features are standardized with `StandardScaler`:
- **Fit** on training data only.
- **Transform** applied to both training and test sets to prevent data leakage.

### Datetime Index Construction

Prophet requires a `ds` (datetime) column. Since engine cycles are not calendar events, an **artificial hourly timestamp sequence** is created:

```python
df["ds"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(np.arange(len(df)), unit="h")
```

This preserves the strict sequential ordering of cycles without introducing any real calendar effects. The test set's time index begins immediately after the training set's last timestamp.

---

## Model Architecture

Prophet decomposes the target signal into additive components:

$$y(t) = g(t) + s(t) + h(t) + \boldsymbol{\beta}^\top \mathbf{x}(t) + \varepsilon_t$$

| Component | Symbol | Role in this application |
|-----------|--------|--------------------------|
| Trend | $g(t)$ | Captures the engine's underlying degradation trajectory |
| Seasonality | $s(t)$ | **Disabled** — no calendar effects in mechanical data |
| Holidays | $h(t)$ | **Disabled** |
| External regressors | $\boldsymbol{\beta}^\top \mathbf{x}(t)$ | All 18 sensor/setting features drive the prediction |
| Noise | $\varepsilon_t$ | Gaussian observation noise |

### Model 1 — Linear Trend

```python
Prophet(
    growth="linear",
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
```

The trend $g(t)$ is modeled as a **piece-wise linear function**:

$$g(t) = (k + \mathbf{a}(t)^\top \boldsymbol{\delta})\,t + (m + \mathbf{a}(t)^\top \boldsymbol{\gamma})$$

where $k$ is the base growth rate, $\boldsymbol{\delta}$ are changepoint adjustments, and $\mathbf{a}(t)$ is an indicator vector for active changepoints. Predictions are post-hoc clipped to $[0,\ 125]$.

### Model 2 — Logistic (Non-Linear) Trend

```python
Prophet(
    growth="logistic",
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
```

The trend is replaced by a **saturating logistic function**:

$$g(t) = \frac{L}{1 + e^{-(k + a(t)^\top \boldsymbol{\delta})(t - (m + a(t)^\top \boldsymbol{\gamma}))}}$$

- `cap = 125` (= `RUL_CAP`) is passed as a column in the DataFrame, enforcing the natural upper bound.
- `floor = 0` prevents negative predictions natively at the model level.
- This eliminates the need for post-hoc output clipping.

---

## Loss Function & Training Strategy

Prophet fits parameters via **Maximum A Posteriori (MAP) estimation** (Stan's L-BFGS optimizer by default), maximizing the joint log-posterior:

$$\log p(\boldsymbol{\theta} \mid \mathcal{D}) \propto \log p(\mathcal{D} \mid \boldsymbol{\theta}) + \log p(\boldsymbol{\theta})$$

Key priors placed on model parameters:

| Parameter | Prior | Effect |
|-----------|-------|--------|
| Changepoint magnitudes $\boldsymbol{\delta}$ | Laplace$(0,\ \tau)$ | Sparse, smooth trend — most changepoints have near-zero magnitude |
| Changepoint prior scale $\tau$ | Set to `0.05` | Conservative — avoids overfitting to noise in the degradation signal |
| Regressor coefficients $\boldsymbol{\beta}$ | Normal$(0,\ \sigma_\beta^2)$ | Regularizes sensor weights |
| Observation noise $\sigma$ | Half-Normal | Ties residuals to a weakly informative scale |

The **Laplace prior on changepoints** is the core Bayesian ingredient: it acts as a sparsity-inducing regularizer analogous to L1 in LASSO regression, producing interpretable trend breaks only where the data strongly demands them.

---

## Training Curriculum

The training procedure follows a two-stage workflow — one per growth mode:

**Stage 1 — Linear Trend**

1. Construct `prophet_train` DataFrame with `ds`, `y` (capped RUL), and all 18 scaled features.
2. Fit Prophet with `growth="linear"`.
3. Predict on the entire training set; clip output to $[0, 125]$.
4. Evaluate Train MAE and Train RMSE.
5. Predict on `prophet_test`; extract the **last-cycle prediction per engine** as the RUL estimate.
6. Evaluate Test MAE, Test RMSE, and NASA Score against `RUL_FD001.txt`.

**Stage 2 — Logistic Trend**

1. Add `cap = 125` and `floor = 0` columns to both train and test DataFrames.
2. Fit Prophet with `growth="logistic"`.
3. Evaluate training metrics (MAE, RMSE).
4. At test time, pass **only the last observed row per engine** (not the full sequence) to `m.predict()`, which is sufficient because the logistic cap already constrains the output space.
5. Evaluate Test MAE, Test RMSE, and NASA Score.

> No cross-validation or hyperparameter search is performed; the changepoint prior scale is set by domain judgment.

---

## Key Discoveries & Insights

1. **Prophet can be repurposed as a multivariate Bayesian regressor.** By disabling all seasonality and using only external regressors, Prophet is transformed from a forecasting tool into a principled Bayesian linear model with a non-linear trend backbone — a non-obvious but effective adaptation.

2. **Disabling seasonality is critical.** Engine degradation follows no weekly, daily, or yearly rhythm. Leaving seasonality enabled would inject spurious periodic components and inflate prediction uncertainty without any signal benefit.

3. **The logistic cap provides natural output constraint.** The linear model requires manual clipping of `yhat` to `[0, 125]`, which is a post-hoc correction that does not influence the fitted parameters. The logistic model embeds this constraint directly into the likelihood, yielding a more calibrated Bayesian posterior.

4. **Test-time inference strategy differs between the two models.** In the linear model, predictions are made for the entire test sequence and the last-cycle value is extracted. In the logistic model, only the final cycle per engine is passed to `predict()` — reducing inference cost and avoiding contamination from intermediate cycle predictions.

5. **Changepoint sparsity is important for smooth degradation curves.** Setting `changepoint_prior_scale=0.05` (rather than the default `0.3`) produces a smooth degradation trend, which physically aligns better with the gradual wear mechanics of turbofan engines.

6. **Feature leakage is prevented through strict scaler discipline.** `StandardScaler` is fit only on training data. This is enforced explicitly via separate `fit_transform` and `transform` calls, a common source of bugs in preprocessing pipelines.

---

## Other Core Components of Prophet

### Changepoint Detection

Prophet automatically places potential changepoints at the first 80% of training observations. The Laplace prior controls which of these changepoints "activate." In the degradation context, activated changepoints correspond to moments of accelerated wear — meaningful physical events.

### External Regressors

Each of the 18 features is added as a linear external regressor:

```python
for c in feature_cols:
    m.add_regressor(c)
```

Internally, Prophet estimates a coefficient $\beta_j$ for each regressor under a Normal prior, making the full model a **Bayesian ridge regression** conditioned on the trend. This is the primary mechanism linking sensor telemetry to RUL.

### Uncertainty Quantification

Prophet natively generates `yhat_lower` and `yhat_upper` credible intervals via a simplified posterior sampling procedure. While not explicitly used in scoring for this project, these intervals are available for downstream reliability analysis (e.g., flagging engines whose predicted RUL uncertainty spans a maintenance threshold).

### Prediction Clipping

For the linear model, a hard clip is applied after inference:

```python
train_pred["yhat_clipped"] = train_pred["yhat"].clip(lower=0, upper=RUL_CAP)
```

This corrects for physically implausible predictions (negative RUL or RUL > 125) that the linear model may produce outside the training distribution.

---

## Final Results

Evaluation is performed on the **FD001 test set** (100 engines). For each engine, only the **final cycle's prediction** is compared against the ground-truth RUL.

### Metrics Summary

| Model | Split | MAE | RMSE | NASA Score |
|-------|-------|-----|------|------------|
| Linear Trend | Train | reported in notebook | reported in notebook | — |
| Linear Trend | **Test** | **reported in notebook** | **reported in notebook** | **reported in notebook** |
| Logistic Trend | Train | reported in notebook | reported in notebook | — |
| Logistic Trend | **Test** | **reported in notebook** | **reported in notebook** | **reported in notebook** |

> Run the notebook to populate exact values — outputs are printed to cell stdout.

### NASA Scoring Function

An asymmetric penalty that weights **under-prediction** (late maintenance warning) more harshly than over-prediction:

$$S = \sum_{i=1}^{N} \begin{cases} e^{-d_i/13} - 1 & \text{if } d_i < 0 \quad \text{(early prediction)} \\ e^{\,d_i/10} - 1 & \text{if } d_i \geq 0 \quad \text{(late prediction)} \end{cases}$$

where $d_i = \hat{y}_i - y_i$. A **lower score is better**.

### Visualization Outputs

- **Scatter plot** — True RUL vs. Predicted RUL across all 100 test engines, with a red dashed ideal-prediction diagonal.
- **Error histogram** — Distribution of residuals $(d_i = \hat{y}_i - y_i)$ with KDE overlay, illustrating prediction bias and spread.

---

## Dependencies & Usage

### Installation

```bash
pip install prophet pandas numpy scikit-learn matplotlib seaborn
```

| Package | Version Requirement | Purpose |
|---------|---------------------|---------|
| `prophet` | ≥ 1.1 | Core forecasting model (Stan backend) |
| `pandas` | ≥ 1.3 | Data I/O and manipulation |
| `numpy` | ≥ 1.21 | Numerical computation |
| `scikit-learn` | ≥ 1.0 | `StandardScaler`, MAE/MSE metrics |
| `matplotlib` | ≥ 3.4 | Plotting |
| `seaborn` | ≥ 0.11 | Statistical visualization |

> Prophet requires either `pystan` (≥ 3.0) or `cmdstanpy` as its Stan backend. On Google Colab, `pip install prophet` resolves this automatically.

### Running the Notebook

1. **Set the data path** — update `data_dir` at the top of the notebook:
   ```python
   data_dir = "/path/to/dataset/folder"   # must contain the three FD001 files
   ```
2. **Execute all cells in order** — the notebook is structured sequentially:
   - **Cells 1–11**: Data loading → preprocessing → Linear Trend Prophet → evaluation
   - **Cells 12–20**: Logistic Trend Prophet → evaluation
3. **Read the outputs** — MAE, RMSE, and NASA Score are printed to stdout; plots are rendered inline.

---

## File Structure

```
Prophet/
├── Prophet_Meta.ipynb      # Main notebook: Prophet RUL prediction (linear & logistic)
├── GRU.ipynb               # GRU-based RUL prediction (draft)
├── GRU_Final_Version.ipynb # Final cleaned GRU implementation (comparison model)
└── README.md               # This file
```

---

## References

- Taylor, S. J., & Letham, B. (2018). [Forecasting at scale](https://doi.org/10.1080/00031305.2017.1380080). *The American Statistician*, 72(1), 37–45.
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *Proceedings of the 1st International Conference on Prognostics and Health Management (PHM)*.
- [NASA C-MAPSS Dataset — PCOE Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
