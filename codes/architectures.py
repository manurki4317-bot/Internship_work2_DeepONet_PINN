# Import the PyTorch library for tensor operations and automatic differentiation
import torch
# Import commonly used neural network functions
import torch.nn.functional as F
# Import PyTorch neural network modules and layers
import torch.nn as nn

#########################################
# Utilities
#########################################

def activation(act_fun):
    """
    Returns the activation function specified by the user
    """
    # Convert the name to lowercase for consistency
    act_fun = act_fun.lower()
    # Available activation functions
    act_dict = {
        "relu"     : F.relu,
        "tanh"     : torch.tanh,
        "gelu"     : F.gelu,
        "sigmoid"  : torch.sigmoid,
        "sin"      : lambda x: torch.sin(2*torch.pi*x),
    }
    # Check that the activation function exists
    if act_fun not in act_dict:
        raise ValueError(f"Unknown activation function: {act_fun}")
    return act_dict[act_fun]

def initializer(initial):
    """
    Returns the selected weight initialization method
    """
    # Convert the name to lowercase for consistency
    initial = initial.lower()
    # Available initialization methods
    initial_dict = {
        "glorot normal": torch.nn.init.xavier_normal_,
        "glorot uniform": torch.nn.init.xavier_uniform_,
        "he normal": torch.nn.init.kaiming_normal_,
        "he uniform": torch.nn.init.kaiming_uniform_,
        "zeros": torch.nn.init.zeros_,
    }
    # Check that the initializer exists
    if initial not in initial_dict:
        raise ValueError(f"Unknown initializer: {initial}")

    return initial_dict[initial]


def get_optimizer(model, lr, schedulerName, epochs, ntrain, batch_size, name="AdamW"):
    """
    Creates the optimizer and the optional learning rate scheduler
    """
    # Select the optimizer
    if name == "L-BFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)

    elif name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    elif name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    else:
        raise ValueError(f"Unknown optimizer: {name}")

    # Configure the learning rate scheduler (if requested)
    scheduler = None
    if schedulerName is not None:
        if schedulerName.lower() == "steplr":
            # Reduce the learning rate every fixed number of epochs
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma = 0.5)
        elif schedulerName.lower() == "cosineannealinglr":
            # Apply cosine annealing over all training iterations
            iterations = epochs*(ntrain//batch_size)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = iterations)
        elif schedulerName.lower() == "reduceonplateau":
            # Reduce the learning rate when the validation loss stops improving
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95,
                                        patience=10, threshold=0.001, threshold_mode='rel', cooldown=0,
                                        min_lr=2e-4, eps=1e-08)
        else:
            raise ValueError(f"Unknown scheduler: {schedulerName}")
    else:
        schedulerName = "None"
    return optimizer, schedulerName, scheduler

# Fourier Features
class FourierFeatures(nn.Module):
    """
    Maps the input coordinates to a higher-dimensional Fourier feature space.
    """
    def __init__(self, scale, mapping_size):
        super().__init__()
        # Number of Fourier features
        self.mapping_size = mapping_size
        # The scale must be non-zero
        if scale == 0:
            raise ValueError("Scale cannot be zero.")
        self.scale = scale
        # Random projection matrix (fixed during training)
        self.register_buffer("B",self.scale * torch.randn((self.mapping_size, 1)))

    def forward(self, x):
        # Input must have shape (N, 1)
        if x.ndim != 2:
          raise ValueError("Input must have shape (N, 1)")
        # Project the input coordinates
        x_proj = torch.matmul((2. * torch.pi * x), self.B.T)
        # Generate sine and cosine Fourier features
        inp = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

        return inp

# Adaptive Fourier Features
class AdaptFF(nn.Module):
    """
    Fourier Feature layer with learnable projection matrix.
    """
    def __init__(self, mapping_size):
        super().__init__()
        # Number of Fourier features
        self.mapping_size = mapping_size
        # Learnable projection matrix
        self.B = nn.Parameter(torch.randn(self.mapping_size,1))

    def forward(self, x):
        # Input must have shape (N, 1)
        if x.ndim != 2:
          raise ValueError("Input must have shape (N, 1)")
        # Project the input coordinates
        x_proj = torch.matmul((2. * torch.pi * x), self.B.T)
        # Generate sine and cosine Fourier features
        inp = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

        return inp

# Adaptive Linear
class AdaptiveLinear(nn.Linear):
    """
    Linear layer with optional adaptive activation scaling.
    """
    def __init__(self, in_features, out_features, bias=True, adaptive_rate=None, adaptive_rate_scaler=None):
        # Initialize the standard linear layer
        super(AdaptiveLinear, self).__init__(in_features, out_features, bias)
        self.adaptive_rate = adaptive_rate
        self.adaptive_rate_scaler = adaptive_rate_scaler
        # Create learnable adaptive scaling parameters
        if self.adaptive_rate is not None:
            self.A = nn.Parameter(self.adaptive_rate * torch.ones(self.in_features))
            # Default scaling factor
            if not self.adaptive_rate_scaler:
                self.adaptive_rate_scaler = 10.0

    def forward(self, input):
        # Apply adaptive scaling if enabled
        if self.adaptive_rate is not None:
            return nn.functional.linear(self.adaptive_rate_scaler * self.A * input, self.weight, self.bias)
        # Otherwise, perform a standard linear transformation
        return nn.functional.linear(input, self.weight, self.bias)


# Loss functions
class L2relLoss():
    """
    Relative L2 loss function.
    """
    def __init__(self):
        self.name = "L2_rel"

    def get_name(self):
        return self.name

    def rel(self, x, y):
      """
      Computes the mean relative L2 error.
      """
      # L2 norm of the prediction error
      diff_norms = torch.norm(x - y, dim=1)
      # L2 norm of the reference solution
      y_norms = torch.norm(y, dim=1)
      # Mean relative error
      return torch.mean(diff_norms / (y_norms + 1e-8))

    def __call__(self, x, y):
        return self.rel(x, y)



# Loss functions
class MSE():
    """
    Mean Squared Error (MSE) loss function.
    """
    def __init__(self):
        self.name = "mse"

    def get_name(self):
        return self.name

    def mse(self, x, y):
        """
        Computes the mean squared error between predictions and targets.
        """
        # Compute the squared difference
        diff = torch.square(x - y)
        # Return the average error
        return torch.mean(diff)

    def __call__(self, x, y):
        return self.mse(x, y)

# Previous implementation of the relative H1 loss
# It has been replaced by the Fourier-based formulation below
class H1relLoss_fourier():
    
    def __init__(self):
        self.name = "H1_rel"

    def get_name(self):
        return self.name

    def rel(self, x, y, size_mean):
        """
        Computes the relative H1 error
        """
        num_examples = x.size(0)
        # Compute the L2 norms of the error and reference solution
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), 2, 1)
        # check division by zero
        if torch.any(y_norms <= 1e-5):
            raise ValueError("Division by zero")
        # Return either the mean or the sum of the relative errors
        if size_mean:
            return torch.mean(diff_norms/y_norms)
        else:
            return torch.sum(diff_norms/y_norms)

    def __call__(self, x, y, beta = 1, size_mean = False):
        # Number of time samples
        n_t = x.size(1)
        # Frequency indices used for Fourier weighting
        k = torch.cat((torch.arange(start = 0, end = n_t//2, step = 1),
                       torch.arange(start = -n_t//2, end = 0, step = 1)),
                       0)
        k = torch.abs(k).reshape(1, n_t)

        # Transform both signals into the Fourier domain
        x = torch.abs(torch.fft.fft(x, dim = 1))
        y = torch.abs(torch.fft.fft(y, dim = 1))
        # Weight higher frequencies more strongly
        weight = 1 + beta*k**2
        weight = torch.sqrt(weight)
        # Compute the weighted relative error
        loss = self.rel(x*weight, y*weight, size_mean)

        return loss

def get_loss(Loss):
    """
    Returns the selected loss function.
    """
    # Select the desired loss function
    if Loss == "L2":
        myloss = L2relLoss()

    elif Loss == "mse":
        myloss = MSE()

    elif Loss == "H1":
        myloss = H1relLoss_fourier()

    else:
        raise ValueError("Invalid Loss type provided.")

    return myloss

# Feed-Forward Neural Networks

class FNN(nn.Module):
    """
    Standard fully connected feed-forward neural network.
    """
    def __init__(self, layer_sizes, activation_str, kernel_initializer, adapt_actfun=False):
        super().__init__()

        # Store the network configuration
        self.layers = layer_sizes
        self.activation = activation(activation_str)
        self.initializer = initializer(kernel_initializer)
        self.linears = nn.ModuleList()
        self.adapt_rate = None

        # Enable adaptive activation scaling if requested
        if adapt_actfun:
            self.adapt_rate = 0.1

        # Build the network layer by layer
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                AdaptiveLinear(
                    layer_sizes[i - 1],
                    layer_sizes[i],
                    adaptive_rate=self.adapt_rate,
                )
            )

            # Initialize weights and biases
            self.initializer(self.linears[-1].weight)
            initializer("zeros")(self.linears[-1].bias)

    def forward(self, x):

        # Apply hidden layers with activation
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))

        # Output layer (no activation)
        x = self.linears[-1](x)

        return x

# FNN with Batch Normalization

class FNN_BN(nn.Module):
    """
    Feed-forward neural network with Batch Normalization.
    """

    def __init__(self, layer_sizes, activation_str, kernel_initializer, adapt_actfun=False):
        super().__init__()

        self.layers = layer_sizes
        self.activation = activation(activation_str)
        self.initializer = initializer(kernel_initializer)
        self.linears = nn.ModuleList()
        self.batch_layer = nn.ModuleList()
        self.adapt_rate = None

        # Enable adaptive activation scaling if requested
        if adapt_actfun:
            self.adapt_rate = 0.1

        # Compute the initialization gain
        if activation_str.lower() in ["tanh", "relu", "leaky_relu"]:
            gain = nn.init.calculate_gain(activation_str.lower())
        else:
            gain = 1

        # Build the network
        for i in range(1, len(layer_sizes)):

            self.linears.append(
                AdaptiveLinear(
                    layer_sizes[i - 1],
                    layer_sizes[i],
                    adaptive_rate=self.adapt_rate,
                )
            )

            # Batch normalization layer
            self.batch_layer.append(nn.BatchNorm1d(layer_sizes[i]))

            # Initialize weights and biases
            self.initializer(self.linears[-1].weight, gain)
            initializer("zeros")(self.linears[-1].bias)

    def forward(self, x):

        # Hidden layers: Linear -> BatchNorm -> Activation
        for i in range(len(self.linears) - 1):
            x = self.linears[i](x)
            x = self.batch_layer[i](x)
            x = self.activation(x)

        # Output layer
        x = self.linears[-1](x)

        return x

# FNN with Layer Normalization

class FNN_LN(nn.Module):
    """
    Feed-forward neural network with Layer Normalization.
    """

    def __init__(self, layers, activation_str, initialization_str, adapt_actfun=False):
        super().__init__()

        # Store the network configuration
        self.layers = layers
        self.activation_str = activation_str
        self.initialization_str = initialization_str
        self.adapt_rate = None

        # Enable adaptive activation scaling if requested
        if adapt_actfun:
            self.adapt_rate = 0.1

        # Create the linear layers
        self.linears = nn.ModuleList([
            AdaptiveLinear(
                self.layers[i],
                self.layers[i + 1],
                adaptive_rate=self.adapt_rate,
            )
            for i in range(len(self.layers) - 1)
        ])

        # Layer normalization is applied only to the hidden layers
        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(self.layers[i])
            for i in range(1, len(self.layers) - 2)
        ])

        # Initialize the network parameters
        self.linears.apply(self.param_initialization)

    def param_initialization(self, m):
        """
        Initializes the weights and biases of each linear layer.
        """

        if isinstance(m, nn.Linear):

            # Compute the initialization gain
            if self.activation_str in ["tanh", "relu"]:
                gain = nn.init.calculate_gain(self.activation_str)
                a = 0

            elif self.activation_str == "leaky_relu":
                gain = nn.init.calculate_gain(self.activation_str, 0.01)
                a = 0.01

            else:
                gain = 1
                a = 0.01

            # Initialize the weights
            if self.initialization_str == "glorot uniform":
                torch.nn.init.xavier_uniform_(m.weight.data, gain=gain)

            elif self.initialization_str == "glorot normal":
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)

            elif self.initialization_str == "he uniform":
                torch.nn.init.kaiming_uniform_(
                    m.weight.data,
                    a=a,
                    nonlinearity=self.activation_str,
                )

            elif self.initialization_str == "he normal":
                torch.nn.init.kaiming_normal_(
                    m.weight.data,
                    a=a,
                    nonlinearity=self.activation_str,
                )

            # Initialize the bias to zero
            torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):

        # First hidden layer
        x = activation(self.activation_str)(self.linears[0](x))

        # Remaining hidden layers with Layer Normalization
        for i in range(1, len(self.layers) - 2):
            x = activation(self.activation_str)(
                self.linears[i](self.layer_norm[i - 1](x))
            )

        # Output layer
        return self.linears[-1](x)