# IMPORT LIBRARIES
from timeit import default_timer # Used to measure the execution time of the training
import torch

# TRAINING CLASS

class Training():
    """
    Class responsible for training and evaluating a neural network
    """

    def __init__(
        self,
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
        show_every=100
    ):
        """
        Store all training parameters
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
        self.x_train = x_train
        self.x_test = x_test
        self.device = device
        self.show_every = show_every

        # Display optimizer being used
        print(self.optimizer.__class__.__name__)


    # ONE TRAINING EPOCH
    def single_train_step(self, ep, t1):
        """
        Perform one complete epoch.
        """

        # Closure function

        # Some optimizers require reevaluating the model multiple times
        
        # This function performs:
        
        # 1) zero gradients
        # 2) forward pass
        # 3) loss computation
        # 4) backpropagation
        
        def closure():

            # Remove gradients from the previous iteration
            self.optimizer.zero_grad()

            # Forward pass

            # DeepONet receives:
            # (branch input, trunk input)
            out = self.model((v, self.x_train))

            # Compute training loss
            loss = self.loss(out, u)

            # Compute gradients
            loss.backward()

            return loss

        # TRAINING MODE
        # Enable training behaviour
    
        # Layers such as BatchNorm or Dropout behave differently during training
        self.model.train()

        # Accumulator for epoch loss
        train_loss = 0

        # Iterate over every mini-batch
        for v, u in self.train_loader:

            # Move data to CPU or GPU
            v = v.to(self.device)
            u = u.to(self.device)

            # Special case: LBFGS optimizer

            if self.optimizer.__class__.__name__ == "LBFGS":

                # LBFGS internally calls closure() several times
                self.optimizer.step(closure)

                # Compute final prediction
                out = self.model((v, self.x_train))

                # Final loss
                loss = self.loss(out, u)

            # Standard optimizers (Adam, SGD, ...)

            else:

                # Forward + backward
                loss = closure()

                # Update network weights
                self.optimizer.step()

            # Accumulate loss
            train_loss += loss.item()

            # CosineAnnealing scheduler

            # This scheduler updates after every batch
            if self.schedulerName.lower() == "cosineannealinglr":

                self.scheduler.step()

        # StepLR scheduler

        # StepLR updates once every epoch
        if self.schedulerName.lower() == "steplr":

            self.scheduler.step()

        # EVALUATION

        # Disable training behaviour
        self.model.eval()

        # Accumulators
        test_l2 = 0.0
        test_mse = 0.0

        # No gradients are required during evaluation
        with torch.no_grad():

            for v, u in self.test_loader:

                # Move data to the selected device
                v = v.to(self.device)
                u = u.to(self.device)

                # Forward pass
                out = self.model((v, self.x_test))

                # Relative L2 error
                test_l2 += L2relLoss()(out, u).item()

                # Mean Squared Error
                test_mse += MSE()(out, u).item()

        # AVERAGE METRICS

        train_loss /= self.ntrain

        test_l2 /= self.ntest

        test_mse /= self.ntest

        # Compute elapsed time
        t2 = default_timer()

        # DISPLAY TRAINING PROGRESS

        if ep % self.show_every == 0:

            print(

                f"Epoch:{ep}  "

                f"Time:{t2 - t1:.2f}  "

                f"Train_loss:{train_loss:.5f}  "

                f"Test_L2:{test_l2:.5f}  "

                f"Test_MSE:{test_mse:.5f}"

            )


    # COMPLETE TRAINING LOOP
    def train(self):
        """
        Train the neural network
        """

        # Record starting time
        t1 = default_timer()

        # Optional PyTorch profiler   

        # The following block can be enabled to profile:
        
        # CPU usage
        # GPU usage
        # Memory consumption
        # Execution time
        

        # my_schedule = schedule(...)
        
        # with torch.profiler.profile(...) as prof:

        # Iterate through every epoch
        for ep in range(self.epochs + 1):

            self.single_train_step(ep, t1)
