# Uncertainty-Aware Prognostics: A Hierarchical Bayesian Framework for Probabilistic Health Monitoring of Turbofan Engines

![Status](https://img.shields.io/badge/Status-Active-success)
![Application](https://img.shields.io/badge/Domain-Aerospace%20%2F%20Predictive%20Maintenance-blue)
![Method](https://img.shields.io/badge/Method-Bayesian%20RNN-orange)

## 📖 Abstract

This project proposes an uncertainty-aware prognostics framework combining a Hierarchical Bayesian regression model and Bayesian Recurrent Neural Network (BRNN) architectures to estimate the probabilistic Remaining Useful Life (RUL) of turbofan engines. 

Standard deterministic models in aviation predictive maintenance often produce overconfident point estimates. By integrating Bayesian inference with the temporal sequence modeling capabilities of recurrent layers, our BRNN is uniquely suited for this task. It allows the model to learn from sequential degradation patterns while explicitly quantifying both **aleatoric** (sensor noise) and **epistemic** (model ignorance) uncertainty. 

This probabilistic approach enables a safety-first framework where maintenance decisions can be optimized based on the degree of predictive uncertainty rather than relying solely on a low RUL estimate.



## 🚀 Application Area: Aerospace & Predictive Maintenance (PdM)

This project focuses on the intersection of aviation safety and industrial IoT, specifically targeting the health monitoring of complex propulsion systems to reduce maintenance costs and prevent catastrophic failures.

### The "Prior" (Project Justification)
Standard deterministic models in aviation are often "overconfident" in their predictions. By implementing a Bayesian Recurrent Neural Network (BRNN), our team aims to update the posterior probabilities of engine failure in real-time. This allows for a safety-first approach where maintenance is triggered not just by a low RUL estimate, but by a high degree of predictive uncertainty.

## 🎯 Task Description

The core task involves **Probabilistic Remaining Useful Life (RUL) Estimation**, broken down into three main pillars:

* **Estimation (via Regression):** Predicting the specific number of flight cycles remaining before engine failure.
* **Uncertainty Analysis:** Quantifying aleatoric and epistemic uncertainty to provide a probability distribution rather than a single-point estimate.
* **Decision Optimization (Optional Extension):** Using Reinforcement Learning to determine optimal maintenance timing based on predicted risk.

## 🧠 Model Architecture

We will not be leveraging a pre-trained foundational model from an existing repository. Instead, the BRNN architecture will be built entirely from scratch.

* **Framework:** PyTorch or TensorFlow/Keras.
* **Bayesian Integration:** Developing the model from the ground up is necessary to strictly control the integration of Bayesian layers (e.g., utilizing **Monte Carlo Dropout** or **Bayes by Backprop**) within the recurrent temporal structures.
* **Purpose:** Tailoring the architecture specifically for our multivariate time-series regression task.



## 📊 Dataset

The analysis will be supported by the simulated **NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) dataset. This dataset consists of multivariate time-series data capturing run-to-failure degradation trajectories of turbofan engines.

* **Data Type:** Multivariate Time Series.
* **Features:** High-dimensional sensor readings (e.g., temperatures, pressures, fan speeds) and variable operating conditions (altitude and throttle settings) recorded over sequential time steps.
* **Complexity:** The project will utilize all four sub-datasets (**FD001 through FD004**) to evaluate the model across a spectrum of complexities, ranging from single-condition/single-fault scenarios (FD001) to highly complex multi-condition/multi-fault environments (FD004).

🔗 **Data Access:** The dataset is publicly available via the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

---

## 🛠️ Getting Started (To Be Updated)
*Instructions for environment setup, training the model, and running inference will be added here.*