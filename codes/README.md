# Burst Classification with Neural Operators and Physics-Informed Neural Networks

This project investigates neural operator architectures for modelling and classifying neuronal bursting dynamics generated from a four-dimensional Hodgkin–Huxley-inspired model. The long-term objective is to incorporate experimental ABF recordings into the same framework.

Currently implemented:

- Deep Operator Network (DeepONet)
- Initial development towards Physics-Informed Neural Networks (PINNs)

The main objective is to learn the relationship between neuronal model parameters and the resulting bursting behaviour, with the aim of classifying different bursting regimes such as:

- Parabolic
- Square-Wave
- Triangular

**Project status:** Work in progress. DeepONet is fully functional, while the PINN implementation is currently under development.

# Project structure

```
.
├── default_params_don.yml          # Configuration file
├── deepONet_HH_pytorch.py          # Main training script
├── dataset/
│   ├── train_burst.mat
│   ├── test_burst.mat
│   └── dataset_raw.mat
└── src/
    ├── __init__.py
    ├── architectures.py           # Neural network architectures and loss functions
    ├── don.py                     # DeepONet implementation
    ├── training.py                # Training and validation routines
    ├── utility_dataset.py         # Dataset loading and preprocessing
    └── physics.py                 # Physics module (currently under development)
```

---

# Dataset

The framework is designed to work with two different sources of neuronal activity:

- Bursts generated from numerical simulations of a four-dimensional ODE model.
- Experimental electrophysiological recordings (.abf files) *(planned for future integration).*

Each burst is:

- extracted,
- resampled to a fixed temporal length,
- normalized,
- assigned a bursting-class label.

The processed datasets are stored as

```
train_burst.mat
test_burst.mat
```

and can be loaded directly for training.

---

# Configuration

Training parameters are specified in

```
default_params_don.yml
```

including:

- network architecture
- optimizer
- learning rate
- scheduler
- batch size
- number of epochs
- dataset paths
- network hyperparameters

---

# Main modules

## `architectures.py`

Provides the neural-network building blocks, including:

- Fully connected neural networks
- Adaptive linear layers
- Fourier feature mappings
- Activation functions
- Weight initialization methods
- Optimizers and learning-rate schedulers
- Custom loss functions

---

## `don.py`

Implements the DeepONet architecture using separate:

- Branch network
- Trunk network

whose outputs are combined through an inner product.

---

## `training.py`

Handles the complete training procedure:

- training loop
- validation loop
- loss computation
- optimizer updates
- scheduler management
- model evaluation

---

## `utility_dataset.py`

Contains utilities for

- loading MATLAB datasets
- preprocessing
- normalization
- scaling
- PyTorch dataset generation

---

## `physics.py`

Contains the differential equations describing the neuronal model, including:

- membrane dynamics
- gating-variable equations
- ionic conductances

This module will later be integrated into the training process to construct a full Physics-Informed Neural Network.

---

# Current status

At the current stage, the repository includes:

- DeepONet implementation
- Dataset generation from numerical simulations
- Data preprocessing and normalization
- Training and evaluation pipeline
- Initial development of the physics module required for PINNs

Future work includes:

- Full PINN implementation
- Integration of experimental ABF recordings
- Implementation of Fourier Neural Operators (FNO)
- Implementation of Wavelet Neural Operators (WNO)
- Comparison between all architectures

---

# Output

During training, the program reports:

- training loss
- relative L2 test error
- mean squared error (MSE)
- elapsed training time
