# IMPORT LIBRARIES
# Used for plotting training curves
import matplotlib.pyplot as plt

# Allows displaying matplotlib figures inside notebooks (useful when running the code in Google Colab/Jupyter, as it was my case)
from IPython.display import display

# These functions read the .mat files and convert them into PyTorch tensors ready for training
from src.utility_dataset import (
    load_train,
    load_test,
)
# Functions used to create the optimizer and loss function based on the YAML configuration
from src.architectures import (
    get_optimizer,
    get_loss,
)

# DeepONet neural network implementation
from src.don import DeepONet

# Training class that contains: training loop, validation loop, loss computation and scheduler updates
from src.training import Training

# External libraries
import torch
import os
import yaml
import argparse

# DEFAULT DEVICE AND DATA TYPE
# Automatically select GPU if CUDA is available
# Using GPU greatly accelerates neural network training

# If no GPU is available, the model runs on CPU
mydevice = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

# Set the default device for all newly created tensors
torch.set_default_device(mydevice)

# Float32 is commonly used in deep learning because it provides a good balance between accuracy and memory usage
torch.set_default_dtype(torch.float32)

# COMMAND LINE ARGUMENTS
# The script receives the configuration file as an argument

# This allows running different experiments without modifying the Python code
parser = argparse.ArgumentParser(
    description="Learning the LHb bursting model with DeepONet"
)

parser.add_argument(
    "--config_file",
    type=str,
    default="default_params_don.yml",
    help="Path to the YAML configuration file"
)


args = parser.parse_args()

# LOAD YAML CONFIGURATION

# Read all training and model parameters from the YAML file
# Keeping parameters outside the code makes experiments easier to reproduce and compare
with open(args.config_file, "r") as config_file:

    config = yaml.safe_load(config_file)

# Extract experiment name from configuration filename
param_file_name = os.path.splitext(
    args.config_file
)[0]


print("Test name:", param_file_name)


# OUTPUT FILE NAMES
# Folder name where experiment information can be stored
name_log_dir = 'exp_' + param_file_name

# Name used when saving the trained model
name_model = 'model_' + param_file_name


# READ TRAINING HYPERPARAMETERS

# Extract general dataset and training parameters from the YAML configuration.
arc = config.get("arc")
dataset_train = config.get("dataset_train")
dataset_test = config.get("dataset_test")
batch_size = config.get("batch_size")
scaling = config.get("scaling")
N_FourierF = config.get("N_FourierF")
adapt_actfun = config.get("adapt_actfun")

# Optimization parameters
scheduler = config.get("scheduler")
Loss = config.get("Loss")
epochs = config.get("epochs")
lr = config.get("lr")
optim = config.get("optimizer")


# DEEPONET PARAMETERS
# Input dimension of the branch network
u_dim = config.get("u_dim")

# Input dimension of the trunk network
x_dim = config.get("x_dim")

# Dimension of the latent representation shared between branch and trunk networks
G_dim = config.get("G_dim")


# Hidden layers of the branch network
# The branch network processes physical parameters
inner_layer_b = config.get("inner_layer_b")

# Hidden layers of the trunk network
# The trunk network processes the temporal coordinate
inner_layer_t = config.get("inner_layer_t")

# Activation functions
activation_b = config.get("activation_b")
activation_t = config.get("activation_t")


# Architecture of branch and trunk networks
# FNN_LN corresponds to fully connected neural networks with layer normalization
arc_b = config.get("arc_b")
arc_t = config.get("arc_t")

# Weight initialization method
# Glorot initialization helps maintain stable gradients during training
initial_b = config.get("initial_b")
initial_t = config.get("initial_t")

# PLOTTING PARAMETERS
# Number of epochs between displaying training information
show_every = config.get("show_every")

# MAIN EXECUTION
# This condition ensures that the following code is only executed when this file is directly launched

# It prevents automatic execution if this script is imported as a module from another Python file
if __name__ == "__main__":

    # DEEPONET ARCHITECTURE DEFINITION

    # DeepONet is composed of two neural networks:
    # 1) Branch network:
    #    Receives the input parameters: [gK0, Iapp, tau_n, eps]
    # 2) Trunk network:
    #    Receives the location where the solution is evaluated: time t

    layers = {

        # Branch network architecture:
        # Input: u_dim parameters
        # Hidden layers: inner_layer_b
        # Output: G_dim latent features
        "branch": [
            u_dim
        ] + inner_layer_b + [
            G_dim
        ],

        # Trunk network architecture:
        # Input: temporal coordinate
        # If Fourier features are used: the input dimension increases
        # Here N_FourierF = 0, therefore only time is used
        "trunk": [
            x_dim * (N_FourierF == 0)
            +
            2 * N_FourierF
        ]
        +
        inner_layer_t
        +
        [
            G_dim
        ],
    }

    # Activation functions used in both subnetworks
    activ = {
        "branch": activation_b,

        "trunk": activation_t,
    }

    # Weight initialization strategy
    # Proper initialization helps avoid: exploding gradients, vanishing gradients and improves convergence
    init = {

        "branch": initial_b,

        "trunk": initial_t,
    }

    # LOAD DATASETS
    # Load training data
    # Returned tensors:
    # X_train: Input parameters
    # x_train: Time coordinates
    # V_train: Target trajectories
    # labels_train: Class labels (not used during DeepONet training)
    X_train, x_train, V_train, labels_train = load_train(
        dataset_train
    )

    # Load testing data using the same structure
    X_test, x_test, V_test, labels_test = load_test(
        dataset_test
    )

    # DATASET DEBUG INFORMATION
    # Print dataset dimensions to verify that the data has the expected shape before starting training
    # This is especially important when adapting the code to a new dataset
    print("\nTRAIN DATASET")
    print("X_train :", X_train.shape)
    print("x_train :", x_train.shape)
    print("V_train :", V_train.shape)
    print("labels  :", labels_train.shape)

    # Check normalization and data range
    # These values help identify: missing normalization, abnormal values and numerical problems
    print("V_train min :", V_train.min().item())
    print("V_train max :", V_train.max().item())
    print("V_train mean:", V_train.mean().item())
    print("V_train std :", V_train.std().item())

    # Display some samples to verify that the parameters and labels are loaded correctly
    print("\nFirst 5 input parameters:")
    print(X_train[:5])
    print("\nFirst labels:")
    print(labels_train[:20])

    # DATA LOADERS
    # DataLoader creates batches of samples during training
    # Instead of using the complete dataset at once, the model receives smaller groups of samples
    
    # Benefits: lower memory consumption, faster optimization and better generalization
    train_loader = torch.utils.data.DataLoader(

        torch.utils.data.TensorDataset(
            X_train,
            V_train
        ),

        batch_size=batch_size,

        # Shuffle training samples so that the model does not learn any artificial ordering from the dataset
        shuffle=True,

        generator=torch.Generator(device=mydevice),
    )



    # Test loader
    # No shuffling is required because no optimization happens during testing
    test_loader = torch.utils.data.DataLoader(

        torch.utils.data.TensorDataset(
            X_test,
            V_test
        ),

        batch_size=batch_size,
    )

    # CREATE DEEPONET MODEL
    # Currently only DeepONet architecture is implemented
    if arc != "DON":

        raise ValueError(
            "Only the DeepONet architecture (arc='DON') is supported."
        )

    # Initialize DeepONet using the previously defined: network layers, activation functions, initialization method and branch/trunk architecture
    model = DeepONet(

        layers,

        activ,

        init,

        arc_b,

        arc_t,

        adapt_actfun,
    )

    # MODEL OUTPUT TEST
    # Perform a forward pass without computing gradients
    print("\nTEST FORWARD")
    with torch.no_grad():
        out = model(
            (
                X_train[:2],
                x_train
            )
        )

    print("Output shape :", out.shape)
    print("Output mean  :", out.mean().item())
    print("Output std   :", out.std().item())

    # Count the number of trainable parameters
    # This gives information about model complexity
    par_tot = sum(
        p.numel()
        for p in model.parameters()
    )
    print(
        "Total trainable parameters:",
        par_tot
    )


    # OPTIMIZER
    # Create optimizer and learning-rate scheduler
    # The optimizer updates the neural network weights
    # The scheduler modifies the learning rate during training to improve convergence
    if optim is None:
        optimizer, schedulerName, scheduler = get_optimizer(
            model,
            lr,
            scheduler,
            epochs,
            X_train.shape[0],
            batch_size,
        )

    else:
        optimizer, schedulerName, scheduler = get_optimizer(
            model,
            lr,
            scheduler,
            epochs,
            X_train.shape[0],
            batch_size,
            optim,
        )


    # LOSS FUNCTION
    # Define the objective function minimized during training
    # For this project:
    # L2/MSE measures the difference between: predicted trajectories and reference Hodgkin-Huxley trajectories
    myloss = get_loss(Loss)

    # TRAINER
    # The Training class manages: training loop, testing loop, loss computation, optimizer updates and scheduler updatesç

    # This keeps the main script cleaner
    trainer = Training(
        model=model,
        epochs=epochs,
        optimizer=optimizer,
        schedulerName=schedulerName,
        scheduler=scheduler,
        loss=myloss,
        ntrain=X_train.shape[0],
        ntest=X_test.shape[0],
        train_loader=train_loader,
        test_loader=test_loader,
        x_train=x_train,
        x_test=x_test,
        device=mydevice,
        show_every=show_every,
    )

    # TRAIN MODEL
    # Start DeepONet optimization
    # The returned dictionary contains the evolution of training and testing losses
    import numpy as np
    losses = trainer.train()
    # Save losses as NumPy files
    # These can later be loaded to visualize convergence without retraining the model
    np.save(
        "train_loss.npy",
        np.array(losses["train"])
    )

    np.save(
        "test_loss.npy",
        np.array(losses["test"])
    )

    # Print information to verify that the loss history has been correctly stored
    print(type(losses))
    print(losses.keys())
    print(len(losses["train"]))
    print(len(losses["test"]))
    print(losses["train"][:5])
    print(losses["test"][:5])

    # PLOT TRAINING HISTORY
    # Visualize how the error changes during training
    # A decreasing training and testing loss indicates that the operator is being learned successfully
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(12,4)
    )

    ax.plot(
        losses["train"],
        label="Train"
    )

    ax.plot(
        losses["test"],
        label="Test"
    )

    ax.set_xlabel("Epoch")

    ax.set_ylabel("Loss")

    ax.legend()

    ax.grid(True)

    # Display figure inside notebook environments
    display(fig)