# Import the PyTorch library
import torch

# Fixed parameters
C = 1.0
VNa = 60.0
VK = -90.0
VL = -80.0
gL = 8.0
V12m = -20.0
km = 15.0
V12n = -25.0
kn = 5.0
gNa0 = 20.0

# Activation functions
def m_inf(V):
    """
    Steady-state sodium activation function
    """
    return 1.0 / (1.0 + torch.exp((V12m - V) / km))


def n_inf(V):
    """
    Steady-state potassium activation function.
    """
    return 1.0 / (1.0 + torch.exp((V12n - V) / kn))

def rhs(V, n, gNa, gK, gK0, tau_n, eps, Iapp):
    """
    Computes the right-hand side (RHS) of the simplified Hodgkin-Huxley bursting model

    Each returned quantity corresponds to the time derivative of one state variable
    """

    # Membrane voltage equation. This equation represents the balance of electrical currents

    # The net current divided by the membrane capacitance gives the rate of change of the membrane potential
    dV = (
        Iapp
        - gL * (V - VL)
        - gNa * m_inf(V) * (V - VNa)
        - gK * n * (V - VK)
    ) / C

    # Potassium activation dynamics
    # The gating variable n relaxes exponentially toward its steady-state value n_inf(V) tau_n controls how quickly this relaxation occurs:

    dn = (n_inf(V) - n) / tau_n

    # Slow sodium conductance dynamics 
    # The sodium conductance evolves on a much slower timescale
    # It is driven by the difference between the reference potassium conductance (gK0) and the current potassium conductance (gK)
    
    # eps is typically a small parameter, ensuring these dynamics are much slower than the voltage dynamics
    dgNa = eps * (gK0 - gK)

    # Slow potassium conductance dynamics

    # The potassium conductance evolves according to the difference between the current sodium conductance and its reference value (gNa0)
    # Together, dgNa and dgK form a slow feedback mechanism that produces bursting oscillations in the model

    dgK = eps * (gNa - gNa0)

    # Return the four time derivatives that define the complete dynamical system
    return dV, dn, dgNa, dgK