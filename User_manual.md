# User Manual – Internship Work Repository

## Overview

This repository contains the work completed during an internship at **BCAM (Basque Center for Applied Mathematics)**.

The project focuses on preprocessing electrophysiological recordings stored in **ABF (Axon Binary File)** format, extracting neuronal bursting events, and preparing datasets for machine learning and neural operator models such as **Physics-Informed Neural Networks (PINNs)**, **DeepONet**, **Fourier Neural Operators (FNO)**, and **Wavelet Neural Operators (WNO)**.

The repository includes preprocessing scripts, generated datasets, visualizations, and the neural network architectures used throughout the project.

---

## Repository Structure

| Folder / File | Description |
|---------------|-------------|
| **[Context](./task/Context.md)** | General overview and objectives of the internship project. |
| **[Resources and References](./task/Libraries_used.md)** | Python libraries and external documentation used throughout the project. |
| **[python](./codes)** | Python scripts and Jupyter notebooks for preprocessing, visualization, and model training. |
| **[bursting](./bursting)** | Raw electrophysiological recordings in ABF format. |
| **[processed_bursts](./task/processed_bursts)** | Extracted bursts and processed datasets used for training and testing. |

---

## Requirements

The project was developed using:

- **Python 3.13**
- Jupyter Notebook

### Main libraries

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `pyabf`
- `torch`
- `scikit-learn`
- `joblib`
- `pyyaml`
- `tqdm`

Most libraries can be installed using:

```bash
pip install numpy scipy pandas matplotlib pyabf torch scikit-learn joblib pyyaml tqdm
```

---

## Project Workflow

The repository follows the workflow below:

1. Load electrophysiological recordings from `.abf` files.
2. Apply optional signal filtering.
3. Detect action potentials (spikes).
4. Identify bursting events using inter-spike interval (ISI) criteria.
5. Extract and normalize individual bursts.
6. Build training and testing datasets.
7. Train and evaluate PINNs, DeepONet, FNO, and WNO models.

---

## Outputs

The repository generates:

- Processed burst datasets
- CSV files containing extracted information
- Scientific plots and figures
- Training and testing datasets
- Saved preprocessing objects (when required)

---

## Notes

- Interactive plots require `%matplotlib widget`.
- Raw ABF recordings are not modified during preprocessing.
- All generated datasets are stored inside the [processed_bursts](./task/processed_bursts) directory.

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
