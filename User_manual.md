# User Manual – Burst Classification with Neural Operators and PINNs

## Overview

This repository contains the work completed during an internship at **BCAM (Basque Center for Applied Mathematics)**.

The project focuses on modelling neuronal bursting dynamics using **Deep Operator Networks (DeepONet)** and **Physics-Informed Neural Networks (PINNs)**. The current implementation is based on bursts generated from a four-dimensional conductance-based ordinary differential equation (ODE) model, with future integration of experimental electrophysiological recordings stored in **ABF (Axon Binary File)** format.

The repository includes scripts for dataset generation, preprocessing, visualization, model training, and evaluation.

---

## Repository Structure

| Folder / File | Description |
|---------------|-------------|
| **[Context](./task/Context.md)** | General overview and objectives of the internship project. |
| **[Project Summary](./README.md)** | General description of the repository and implemented models. |
| **[Resources and References](./task/Libraries_used.md)** | Python libraries, packages and external references used throughout the project. |
| **[python](./codes)** | Python scripts and Jupyter notebooks used for dataset generation, preprocessing, visualization and model training. |
| **[dataset](./dataset)** | Simulated datasets used for training and testing. |
| **[src](./src)** | Source code containing DeepONet, PINN modules, architectures and utilities. |

---

## Requirements

The project was developed using:

- **Python 3.13**
- Jupyter Notebook

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

The repository follows the workflow below:

1. Generate neuronal bursts by solving a four-dimensional ODE model.
2. Build datasets by varying the model parameters.
3. Normalize and preprocess the simulated data.
4. Split the dataset into training and testing subsets.
5. Train and evaluate a DeepONet model.
6. Prepare the physics module for future PINN implementation.
7. Extend the framework to experimental ABF recordings and additional neural operators (FNO and WNO).

---

## Outputs

The repository generates:

- Simulated neuronal burst datasets (`.mat`)
- Training and testing datasets
- Trained DeepONet models
- Training logs
- Interactive visualizations
- Performance metrics (Relative L2 Error and MSE)

---

## Notes

- Interactive Plotly figures are supported for burst visualization.
- The current version is based on simulated neuronal activity.
- The `physics.py` module has been incorporated to facilitate the future implementation of PINNs.
- Experimental ABF recordings will be integrated in future versions of the project.

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
