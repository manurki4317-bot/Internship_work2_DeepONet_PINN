# Import a high-resolution timer to measure training time
from timeit import default_timer

# Import PyTorch
import torch

# Import custom loss functions
from .architectures import L2relLoss, MSE


class Training():
    """
    Training class responsible for:
        - Training the neural network
        - Evaluating it on the test dataset
        - Recording the training and test losses
    """

    def __init__(self,
                 model,
                 epochs,
                 optimizer,
                 schedulerName,
                 scheduler,
                 loss,
                 ntrain,
                 ntest,
                 train_loader,
                 test_loader,
                 x_train,
                 x_test,
                 device='cpu',
                 show_every=100):
        """
        Store all objects and hyperparameters required during training

        """

        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.schedulerName = schedulerName
        self.scheduler = scheduler
        self.loss = loss

        self.ntrain = ntrain
        self.ntest = ntest

        self.train_loader = train_loader
        self.test_loader = test_loader

        # Move coordinate tensors to the selected device
        self.x_train = x_train.to(device)
        self.x_test = x_test.to(device)

        self.device = device
        self.show_every = show_every

        # Display the optimizer being used
        print(f"Optimizer: {self.optimizer.__class__.__name__}")


    def single_train_step(self, ep, t1):
        """
        Perform one complete training epoch followed by evaluation
        """

        # TRAINING PHASE
        # Enable training mode (Dropout, BatchNorm, etc.)
        self.model.train()

        train_loss = 0.0
        # Iterate through all mini-batches
        for X, V in self.train_loader:
            # Move data to CPU/GPU
            X = X.to(self.device)
            V = V.to(self.device)

            # Special treatment for LBFGS optimizer

            if self.optimizer.__class__.__name__ == "LBFGS":

                # LBFGS requires a closure that reevaluates the model
                def closure():

                    # Reset gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    out = self.model((X, self.x_train))

                    # Compute loss
                    loss = self.loss(out, V)

                    # Backpropagation
                    loss.backward()

                    return loss

                # Optimizer internally calls the closure several times
                loss = self.optimizer.step(closure)

            # Standard optimizers (Adam, AdamW...)

            else:

                # Clear previous gradients
                self.optimizer.zero_grad()

                # Forward propagation
                out = self.model((X, self.x_train))

                # Compute loss
                loss = self.loss(out, V)

                # Compute gradients
                loss.backward()

                # Update network parameters
                self.optimizer.step()

            # Accumulate batch loss
            train_loss += loss.item()

            # Update CosineAnnealing scheduler

            # CosineAnnealingLR is updated every batch
            if (
                self.scheduler is not None
                and self.schedulerName.lower() == "cosineannealinglr"
            ):
                self.scheduler.step()

        # Update StepLR scheduler

        # StepLR is updated once per epoch
        if (
            self.scheduler is not None
            and self.schedulerName.lower() == "steplr"
        ):
            self.scheduler.step()

        # TEST PHASE
        # Disable training-specific layers
        self.model.eval()

        test_l2 = 0.0
        test_mse = 0.0

        # No gradient computation during evaluation
        with torch.no_grad():

            # Iterate through test batches
            for X, V in self.test_loader:

                X = X.to(self.device)
                V = V.to(self.device)

                # Forward pass
                out = self.model((X, self.x_test))

                # Compute evaluation metrics
                test_l2 += L2relLoss()(out, V).item()
                test_mse += MSE()(out, V).item()

        # Compute average losses
        train_loss /= len(self.train_loader)
        test_l2 /= len(self.test_loader)
        test_mse /= len(self.test_loader)

        # Measure elapsed time
        t2 = default_timer()

        # Print progress
        if ep % self.show_every == 0:

            print(
                f"Epoch:{ep}  "
                f"Time:{t2-t1:.2f}  "
                f"Train_loss_{self.loss.get_name()}:{train_loss:.5f}  "
                f"Test_loss_L2:{test_l2:.5f}  "
                f"Test_MSE:{test_mse:.5f}"
            )

        return train_loss, test_l2


    def train(self):
        """
        Execute the complete training process
        """

        # Start timer
        t1 = default_timer()

        train_losses = []
        test_losses = []

        # Loop through all epochs
        for ep in range(self.epochs + 1):

            train_loss, test_loss = self.single_train_step(ep, t1)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

        # Return complete training history
        return {
            "train": train_losses,
            "test": test_losses
        }
