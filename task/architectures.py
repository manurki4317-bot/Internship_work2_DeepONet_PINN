# IMPORT LIBRARIES
import torch
import torch.nn as nn # PyTorch module containing neural network layers


# DEEPONET CLASS
class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet)
    """

    def __init__(
        self,
        layers,
        activation_str,
        kernel_initializer,
        arc_b,
        arc_t,
        adapt_actfun=False
    ):
        """
        Constructor of the DeepONet
        """

        # Initialize the parent PyTorch class
        super().__init__()

        # Store network configuration
        # Layer sizes for the branch network
        self.layer_b = layers["branch"]

        # Layer sizes for the trunk network
        self.layer_t = layers["trunk"]

        # Activation functions
        self.act_b = activation_str["branch"]
        self.act_t = activation_str["trunk"]

        # Weight initialization methods
        self.init_b = kernel_initializer["branch"]
        self.init_t = kernel_initializer["trunk"]

        # Architecture names
        self.arc_b = arc_b
        self.arc_t = arc_t

        # Whether adaptive activations are enabled
        self.adapt = adapt_actfun

        # BUILD BRANCH NETWORK

        # The branch network processes the input parameters

        if self.arc_b == "FNN":

            # Standard fully-connected neural network
            self.branch = FNN(
                self.layer_b,
                self.act_b,
                self.init_b,
                self.adapt
            )

        elif self.arc_b == "FNN_BN":

            # Fully-connected network with Batch Normalization
            self.branch = FNN_BN(
                self.layer_b,
                self.act_b,
                self.init_b,
                self.adapt
            )

        elif self.arc_b == "FNN_LN":

            # Fully-connected network with Layer Normalization
            self.branch = FNN_LN(
                self.layer_b,
                self.act_b,
                self.init_b,
                self.adapt
            )

        else:

            # Raise an error if the requested architecture has not been implemented
            raise NotImplementedError(
                "Architecture for branch not implemented yet"
            )

        # BUILD TRUNK NETWORK
        # The trunk network receives the coordinates where the solution is evaluated
        # In this case, these coordinates correspond to the simulation time

        if self.arc_t == "FNN":

            self.trunk = FNN(
                self.layer_t,
                self.act_t,
                self.init_t,
                self.adapt
            )

        elif self.arc_t == "FNN_BN":

            self.trunk = FNN_BN(
                self.layer_t,
                self.act_t,
                self.init_t,
                self.adapt
            )

        elif self.arc_t == "FNN_LN":

            self.trunk = FNN_LN(
                self.layer_t,
                self.act_t,
                self.init_t,
                self.adapt
            )

        elif self.arc_t == "FourierFeatures":

            # Number of Fourier frequencies
            self.mapping_size = 10

            # Scaling of the Fourier projection
            self.scale = 1

            # The trunk becomes:
            
            # Time, Fourier Features and Fully Connected Network
            self.trunk = nn.Sequential(

                FourierFeatures(
                    self.scale,
                    self.mapping_size
                ),

                FNN_LN(
                    self.layer_t,
                    self.act_t,
                    self.init_t,
                    self.adapt
                )
            )

        elif self.arc_t == "AdaptFF":

            # Adaptive Fourier Features

            self.mapping_size = 10

            self.trunk = nn.Sequential(

                AdaptFF(self.mapping_size),

                FNN_LN(
                    self.layer_t,
                    self.act_t,
                    self.init_t,
                    self.adapt
                )
            )

        else:

            raise NotImplementedError(
                "Architecture for trunk not implemented yet."
            )

        # OUTPUT BIAS

        # Learnable scalar bias added to every prediction
        
        # During training this value is optimized together with the network weights
        self.b = nn.parameter.Parameter(
            torch.tensor(0.0)
        )

    # FORWARD PASS
    def forward(self, x):
        """
        Computes one forward pass
        """

        # Separate branch and trunk inputs

        b_in = x[0]

        t_in = x[1]

        # Process branch input

        # Encodes the input parameters into a latent vector
        b_in = self.branch(b_in)

        # Process trunk input

        # Encodes every time coordinate into another latent vector
        
        # Unlike the branch network, an activation function is also applied after the last layer
        t_in = activation(self.act_t)(
            self.trunk(t_in)
        )

        # Combine branch and trunk representations
        out = torch.einsum(
            "ij,kj->ik",
            b_in,
            t_in
        )

        # Add learnable bias
        out += self.b

        # Return predicted voltage traces
        return out