"""
This file defines the public interface of the package
"""

# Public API

# __all__ defines the list of objects that are considered public components of the package

# These names are exported when someone writes:
# from src import *

# Only the objects listed below will be imported automatically

# This also serves as documentation, indicating which functions, classes, and utilities are intended to be used by external users

__all__ = [

    # Main DeepONet model
    "DeepONet",

    # Training class
    "Training",

    # Neural network utilities

    # Activation function selector
    "activation",

    # Weight initialization methods
    "initializer",

    # Optimizer selector
    "get_optimizer",

    # Loss function selector
    "get_loss",

    # Fourier feature mappings
    "FourierFeatures",
    "AdaptFF",

    # Adaptive fully connected layer
    "AdaptiveLinear",

    # Loss functions
    "L2relLoss",
    "MSE",

    # Feed-forward neural network architectures
    "FNN",
    "FNN_BN",
    "FNN_LN",

    # Dataset loading functions
    "load_single_train",
    "load_single_test",
    "load_train",
    "load_test",

    # Physics module
    # Hodgkin-Huxley equations
    "rhs",

    # Activation functions
    "m_inf",
    "n_inf",
]


# Import neural network architectures and utilities

# These imports expose the neural network components defined in
# architectures.py directly through the package

from .architectures import (

    # Returns the activation function specified in the configuration
    activation,

    # Returns the desired weight initialization method
    initializer,

    # Creates the optimizer
    get_optimizer,

    # Creates the selected loss function
    get_loss,

    # Standard Fourier Feature encoding layer
    FourierFeatures,

    # Adaptive Fourier Feature layer
    AdaptFF,

    # Adaptive fully connected linear layer
    AdaptiveLinear,

    # Relative L2 loss function
    L2relLoss,

    # Mean Squared Error loss
    MSE,

    # Standard feed-forward neural network
    FNN,

    # Feed-forward neural network with Batch Normalization
    FNN_BN,

    # Feed-forward neural network with Layer Normalization
    FNN_LN,
)


# Import the DeepONet architecture
# Makes the DeepONet class directly accessible as:
#     src.DeepONet
from .don import DeepONet

# Import the training class

# The Training class contains the complete training loop, validation procedures, optimizer updates, checkpoint handling, and evaluation routines
from .training import Training

# Import dataset loading utilities

# These functions load and preprocess the datasets used for DeepONet training and testing
from .utility_dataset import (

    # Load one training sample
    load_single_train,

    # Load one testing sample
    load_single_test,

    # Load the complete training dataset
    load_train,

    # Load the complete testing dataset
    load_test,
)

# Import physics functions

# These functions define the Hodgkin-Huxley mathematical model
# used by the Physics-Informed Neural Network (PINN)

from .physics import (

    # Computes the right-hand side of the ODE system
    rhs,

    # Steady-state sodium activation function
    m_inf,

    # Steady-state potassium activation function
    n_inf,
)