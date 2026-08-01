# User Manual – Burst Classification with Neural Operators and PINNs

## Overview

This repository contains the implementation developed during an internship at **BCAM (Basque Center for Applied Mathematics)**.

The project focuses on modelling and learning neuronal bursting dynamics using neural operators and physics-informed neural networks. The current implementation is based on bursts generated from a four-dimensional conductance-based ordinary differential equation (ODE) model. The repository has been designed to facilitate future integration of experimental electrophysiological recordings (.abf files).

The repository includes scripts for dataset generation, preprocessing, visualization, model training, and evaluation.

---

## Repository Structure

| Folder / File | Description |
|---------------|-------------|
| **default_params.yml** | Configuration file containing training and model hyperparameters. |
| **deepONet_HH_pytorch.py** | Main script used for training DeepONet models. |
| **dataset/** | Generated datasets used for training and testing. |
| **src/** | Source code for neural network architectures, training, dataset handling and physics modules. |
| **README.md** | General description of the project. |

Inside `src/`:

| File | Description |
|------|-------------|
| **architectures.py** | Neural network architectures, activation functions, optimizers and loss functions. |
| **don.py** | DeepONet implementation. |
| **training.py** | Training and validation loops. |
| **utility_dataset.py** | Dataset loading, preprocessing and normalization utilities. |
| **physics.py** | Differential equations describing the neuronal model (under development for PINNs). |

---

## Requirements

The project was developed using:

- **Python 3.13**
- Jupyter Notebook (recommended for experiments)

### Main libraries

- `numpy`
- `scipy`
- `torch`
- `matplotlib`
- `plotly`
- `scikit-learn`
- `pyyaml`

Most libraries can be installed using:

```bash
pip install numpy scipy torch matplotlib plotly scikit-learn pyyaml
```

---

## Project Workflow

The current workflow is:

1. Generate neuronal bursts by solving a four-dimensional ODE model.
2. Build datasets by varying the model parameters.
3. Normalize and preprocess the generated data.
4. Split the dataset into training and testing subsets.
5. Train a DeepONet model to learn the mapping between model parameters and neuronal dynamics.
6. Evaluate the model using Relative L2 Error and Mean Squared Error (MSE).
7. Prepare the physics module for future PINN implementation.

Future developments include:

- Integration of experimental ABF recordings.
- Complete implementation of Physics-Informed Neural Networks.
- Comparison with Fourier Neural Operators (FNO) and Wavelet Neural Operators (WNO).

---

## Outputs

The repository generates:

- Simulated neuronal burst datasets (`.mat`)
- Training and testing datasets
- Trained DeepONet models
- Training logs
- Interactive visualizations of neuronal activity
- Performance metrics (Relative L2 Error and MSE)

---

## Notes

- The current implementation uses simulated neuronal activity generated from the ODE model.
- The `physics.py` module has been introduced as the basis for the future PINN implementation.
- Experimental ABF recordings are planned for future versions of the project.

---

## Author

**Manurki4317**

Internship at **BCAM (Basque Center for Applied Mathematics)**

GitHub: https://github.com/manurki4317-bot

---

## License

This repository is intended for academic and research purposes.

If you use or reference this work, please provide appropriate acknowledgment.

---

*This documentation was prepared with the assistance of AI tools and subsequently reviewed and adapted by the author.*
