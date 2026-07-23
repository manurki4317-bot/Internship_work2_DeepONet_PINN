#IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS
C = 1.0 # Membrane capacitance (controls how quickly voltage changes)
Iapp = 4.3 # External applied current
VNa = 60.0 # Sodium reversal potential (mV)
VK  = -90.0 # Potassium reversal potential (mV)
VL  = -80.0 # Leak reversal potential (mV)
gL = 8.0 # Leak conductance
tau_n = 0.17 # Time constant for the potassium activation variable n
V12m = -20.0 # Half-activation voltage for sodium activation
km = 15.0 # Slope of the sodium activation curve
V12n = -25.0 # Half-activation voltage for potassium activation
kn = 5.0 # Slope of the potassium activation curve
eps = 0.005 # Slow adaptation rate of sodium and potassium conductances
gNa0 = 20.0 # Reference sodium conductance


# ACTIVATION FUNCTIONS
def m_inf(V):
    """
    Steady-state activation function for sodium channels
    """
    return 1.0 / (1.0 + np.exp((V12m - V)/km))

def n_inf(V):
    """
    Steady-state activation function for potassium channels
    """
    return 1.0 / (1.0 + np.exp((V12n - V)/kn))

# ODE SYSTEM
def rhs(state, gK0):
    """
    Computes the right-hand side of the differential equations
    """

    V, n, gNa, gK = state     # Unpack the state variables

    # Membrane potential equation
    # Total current = Applied current - Leak current - Sodium current - Potassium current
    # Dividing by C gives dV/dt
    dV = (Iapp - gL * (V - VL) - gNa * m_inf(V) * (V - VNa) - gK * n * (V - VK)) / C

    # Potassium gating variable dynamics
    dn = (n_inf(V) - n) / tau_n

    # Slow sodium conductance adaptation
    dgNa = eps * (gK0 - gK)

    # Slow potassium conductance adaptation
    dgK = eps * (gNa - gNa0)

    # Return all derivatives as a vector
    return np.array([dV, dn, dgNa, dgK])


# RK4 STEP
def rk4_step(state, dt, gK0):
    """
    Performs one Runge-Kutta 4th-order integration step
    """

    # Slope at the beginning of the interval
    k1 = rhs(state, gK0)

    # Slope at the midpoint using k1
    k2 = rhs(state + 0.5 * dt * k1, gK0)

    # Another midpoint estimate using k2
    k3 = rhs(state + 0.5 * dt * k2, gK0)

    # Slope at the end of the interval
    k4 = rhs(state + dt * k3, gK0)

    # Weighted average of the four slopes
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# SIMULATION FUNCTION
def simulate(gK0, T=1000, dt=0.05):
    """
    Simulates the neuron for a given bursting regime
    """

    # Number of integration steps
    N = int(T / dt)

    # Allocate memory for time values
    time = np.zeros(N + 1)

    # Allocate memory for membrane potential
    V_trace = np.zeros(N + 1)

    # Initial state:
    # V = membrane voltage
    # n = potassium activation
    # gNa = sodium conductance
    # gK  = potassium conductance
    state = np.array([
        -60.0,      # Initial voltage
        0.0,        # Initial gating variable
        gNa0 + 1,   # Initial sodium conductance
        gK0         # Initial potassium conductance
    ])

    # Store the initial voltage
    V_trace[0] = state[0]

    # Time integration loop
    for i in range(N):

        # Advance the system one RK4 step
        state = rk4_step(state, dt, gK0)

        # Store voltage
        V_trace[i + 1] = state[0]

        # Update simulation time
        time[i + 1] = time[i] + dt

    return time, V_trace

# RUN DIFFERENT BURSTING REGIMES
# Potassium conductance values corresponding
gK0_values = {
    "Parabolic (11.0)": 11.0,
    "Triangular (9.75)": 9.75,
    "Square-wave (9.0)": 9.0
}

# Create a figure with enough space for three subplots
plt.figure(figsize=(12, 8))

# Simulate each bursting regime
for i, (label, gK0) in enumerate(gK0_values.items()):

    # Run the numerical simulation
    t, V = simulate(gK0)

    # Create one subplot for this regime
    plt.subplot(3, 1, i + 1)

    # Plot membrane potential versus time
    plt.plot(t, V)

    # Set subplot title
    plt.title(label)

    # Label y-axis
    plt.ylabel("V (mV)")

# Label x-axis
plt.xlabel("Time")

# Improve spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()