# Burst Classification with Neural Operators and PINNs

This repository focuses on modelling, learning and classifying neuronal bursting dynamics using Deep Operator Networks (DeepONet) and Physics-Informed Neural Networks (PINNs). The project is based on simulated neuronal activity generated from a four-dimensional conductance-based model, with future integration of experimental electrophysiological recordings.

## What has been done in the project

- Simulate neuronal bursting activity from a four-dimensional ODE model.
- Generate datasets by varying the main model parameters.
- Preprocess simulated bursts (normalization, train/test splitting and formatting).
- Implement and train a Deep Operator Network (DeepONet).
- Develop the initial framework required for a Physics-Informed Neural Network (PINN).
- Incorporate the governing differential equations into a dedicated physics module for future PINN implementation.
- Evaluate model performance using Relative L2 Error and Mean Squared Error (MSE).
- Visualize simulated neuronal activity and inspect different bursting regimes.
- Organize the project into a modular and reproducible repository for future developments.

## Tools

- **Programming language:** Python 3.13

- **Main libraries used:**
  - `numpy` — numerical computations.
  - `scipy` — numerical integration utilities and MATLAB (`.mat`) file handling.
  - `torch` — implementation and training of DeepONet and the PINN framework.
  - `plotly` — interactive visualization of simulated neuronal activity.
  - `matplotlib` — visualization of datasets and training curves.
  - `scikit-learn` — dataset preprocessing and train/test splitting.
  - `pyyaml` — configuration file management.

For additional information about the libraries and packages used throughout the project, see:

- **[Resources and References](./task/Libraries_used.md)**
