# Import PyTorch libraries
import torch
import torch.nn as nn

# Import the available neural network architectures
from .architectures import (
    FNN,
    FNN_BN,
    FNN_LN,
    FourierFeatures,
    AdaptFF,
)

# Import the activation function selector
from .architectures import activation


# DeepONet class
class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet) architecture

    The model consists of two subnetworks:
    - Branch network: processes the input function
    - Trunk network: processes the evaluation coordinates
    """

    def __init__(self, layers, activation_str, kernel_initializer,
                 arc_b, arc_t, adapt_actfun=False):
        """Initialize the branch and trunk networks"""
        super().__init__()

        # Store the network configuration
        self.layer_b = layers["branch"]
        self.layer_t = layers["trunk"]

        self.act_b = activation_str["branch"]
        self.act_t = activation_str["trunk"]

        self.init_b = kernel_initializer["branch"]
        self.init_t = kernel_initializer["trunk"]

        self.arc_b = arc_b
        self.arc_t = arc_t
        self.adapt = adapt_actfun

        
        # Build the branch network
        if self.arc_b == "FNN":
            self.branch = FNN(self.layer_b, self.act_b, self.init_b, self.adapt)

        elif self.arc_b == "FNN_BN":
            self.branch = FNN_BN(self.layer_b, self.act_b, self.init_b, self.adapt)

        elif self.arc_b == "FNN_LN":
            self.branch = FNN_LN(self.layer_b, self.act_b, self.init_b, self.adapt)

        else:
            raise NotImplementedError(
                "Architecture for branch not implemented yet"
            )

        # Build the trunk network
        if self.arc_t == "FNN":
            self.trunk = FNN(self.layer_t, self.act_t, self.init_t, self.adapt)

        elif self.arc_t == "FNN_BN":
            self.trunk = FNN_BN(self.layer_t, self.act_t, self.init_t, self.adapt)

        elif self.arc_t == "FNN_LN":
            self.trunk = FNN_LN(self.layer_t, self.act_t, self.init_t, self.adapt)

        elif self.arc_t == "FourierFeatures":

            # Fourier feature encoding followed by an FNN
            self.mapping_size = 10
            self.scale = 1

            self.trunk = nn.Sequential(
                FourierFeatures(self.scale, self.mapping_size),
                FNN_LN(self.layer_t, self.act_t, self.init_t, self.adapt),
            )

        elif self.arc_t == "AdaptFF":

            # Learnable Fourier feature encoding followed by an FNN
            self.mapping_size = 10

            self.trunk = nn.Sequential(
                AdaptFF(self.mapping_size),
                FNN_LN(self.layer_t, self.act_t, self.init_t, self.adapt),
            )

        else:
            raise NotImplementedError(
                "Architecture for trunk not implemented yet."
            )

        # Learnable output bias
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """
        Computes the DeepONet prediction.
        """

        # Split the inputs into branch and trunk data
        b_in = x[0]
        t_in = x[1]

        # Compute the latent representations
        b_in = self.branch(b_in)
        t_in = self.trunk(t_in)

        # Ensure that both subnetworks produce the same latent dimension
        if b_in.shape[1] != t_in.shape[1]:
            raise ValueError(
                f"Branch output has shape {b_in.shape}, "
                f"but trunk output has shape {t_in.shape}. "
                "Both must have the same latent dimension."
            )

        # Compute the DeepONet output as the inner product
        # between the branch and trunk representations
        out = torch.einsum("ij,kj->ik", b_in, t_in)

        # Add the learnable bias
        out += self.b

        return out
