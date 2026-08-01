# User Manual

## Overview

This repository contains the work carried out during my internship at **BCAM (Basque Center for Applied Mathematics)**.

It serves as a record of the different tasks, experiments, and developments completed throughout the internship. The main focus of the project is the application of **Deep Operator Networks (DeepONet)** and the ongoing implementation of **Physics-Informed Neural Networks (PINNs)** for modelling and classifying neuronal bursting dynamics.

The current work is based on bursts generated from a four-dimensional conductance-based ordinary differential equation (ODE) model. The future objective is to extend the framework to experimental electrophysiological recordings stored in **ABF (Axon Binary File)** format.

The repository contains documentation, Python scripts, Jupyter notebooks, simulated datasets, and the source code developed during the internship.

---

## Repository Structure

| Resource | Description |
|----------|-------------|
| **[Project Summary](./README.md)** | General overview of the internship repository and its organization. |
| **[Context](./task/Context.md)** | Initial background, objectives and motivation of the project. |
| **[Resources and References](./task/Libraries_used.md)** | Documentation, Python libraries and external resources used throughout the internship. |
| **[Task codes](./codes)** | Python scripts developed for the different tasks carried out during the project. |
| **[Data simulation](./codes/Data_simulation.ipynb)** | Notebook used to generate simulated neuronal bursts from the ODE model. |
| **[Execute in Google Colab](./codes/Execute_in_Colab.ipynb)** | Ready-to-run notebook for dataset visualization and DeepONet training in Google Colab. |
| **[Burst recordings](./task/bursting)** | Experimental ABF recordings reserved for future developments. |

---

## Requirements

The project was developed using:

- **Python 3.13**
- **Jupyter Notebook**
- **Google Colab**

### Main libraries

- `numpy`
- `scipy`
- `torch`
- `plotly`
- `matplotlib`
- `scikit-learn`
- `pyyaml`

Most of them can be installed using:

```bash
pip install numpy scipy torch plotly matplotlib scikit-learn pyyaml
```

---

## Project Workflow

The main workflow followed during the internship is:

1. Generate neuronal bursts by solving a four-dimensional ODE model.
2. Build simulated datasets by varying the model parameters.
3. Preprocess and normalize the generated data.
4. Split the datasets into training and testing subsets.
5. Train and evaluate a DeepONet model.
6. Develop the physics module required for future PINN implementation.
7. Prepare the framework for the future integration of experimental ABF recordings.

---

## Outputs

The repository contains and generates:

- Simulated neuronal burst datasets (`.mat`)
- Training and testing datasets
- DeepONet training scripts
- Training logs
- Interactive visualizations
- Performance metrics (Relative L2 Error and MSE)
- Documentation describing the work completed during the internship

---

## Notes

- Interactive Plotly figures are used to visualize the simulated bursts.
- The current implementation focuses on simulated neuronal activity.
- The PINN framework is under development.
- Experimental ABF recordings are included as future work and are not yet integrated into the training pipeline.

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
