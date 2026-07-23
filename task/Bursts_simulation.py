# IMPORT LIBRARIES
import numpy as np
from scipy.io import savemat # Function to save variables into a MATLAB (.mat) file
import plotly.graph_objects as go # To create interactive figures


np.random.seed(42)# Fix random seed



# FIXED PARAMETERS OF THE MODEL
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


# ACTIVATION FUNCTIONS
def m_inf(V):
    """
    Steady-state activation function for sodium channels
    """
    return 1.0 / (1.0 + np.exp((V12m - V) / km))


def n_inf(V):
    """
    Steady-state activation function for potassium channels
    """
    return 1.0 / (1.0 + np.exp((V12n - V) / kn))


# ODE SYSTEM
def rhs(state, gK0, tau_n, eps, Iapp):
    """
    Computes the derivatives of the dynamical system
    """
    V, n, gNa, gK = state    # Extract each state variable

    dV = (     # Computes membrane potential variation

        Iapp
        - gL * (V - VL)
        - gNa * m_inf(V) * (V - VNa)
        - gK * n * (V - VK)
    ) / C
    dn = (n_inf(V) - n) / tau_n     # Potassium gating variable dynamics
    dgNa = eps * (gK0 - gK)     # Slow sodium conductance dynamics
    dgK = eps * (gNa - gNa0)    # Slow potassium conductance dynamics
    return np.array([dV, dn, dgNa, dgK])    # Return all derivatives as a vector


# RUNGE-KUTTA 4 INTEGRATOR
def rk4_step(state, dt, gK0, tau_n, eps, Iapp):
    """
    Integration step using the fourth-order Runge-Kutta method
    Their weighted average gives an accurate approximation
    """
    k1 = rhs(state, gK0, tau_n, eps, Iapp)
    k2 = rhs(state + 0.5 * dt * k1, gK0, tau_n, eps, Iapp)
    k3 = rhs(state + 0.5 * dt * k2, gK0, tau_n, eps, Iapp)
    k4 = rhs(state + dt * k3, gK0, tau_n, eps, Iapp)

    # RK4 weighted average
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# SIMULATION FUNCTION
def simulate(gK0, tau_n, eps, Iapp, T=1000, dt=0.05):
    """
    Simulates one burst using the RK4 solver
    """

    N = int(T / dt)     # Number of integration steps

    time = np.zeros(N + 1, dtype=np.float32)     # Allocate memory for time vector
    V_trace = np.zeros(N + 1, dtype=np.float32)  # Allocate memory for voltage

    # Initial state
    state = np.array([
        -60.0,          # Initial voltage
        0.0,            # Initial gating variable
        gNa0 + 1.0,     # Initial sodium conductance
        gK0             # Initial potassium conductance
    ])

    V_trace[0] = state[0]     # Store initial voltage
    # Time integration loop
    for i in range(N):
        # Advance one RK4 step
        state = rk4_step(
            state,
            dt,
            gK0,
            tau_n,
            eps,
            Iapp
        )
        V_trace[i + 1] = state[0]         # Save voltage
        time[i + 1] = time[i] + dt        # Update time
    return time, V_trace

# DATASET PARAMETERS
# Number of simulations per burst class
Nsim_class = 500

# Fixed physiological parameters
tau_n = 0.17
eps = 0.0016
Iapp = 2.0

# Central value of gK0 for each burst class

# label 0 -> Parabolic bursting
# label 1 -> Triangular bursting
# label 2 -> Square-wave bursting
gK0_values = [
    (11.0, 0),
    (9.75, 1),
    (9.0, 2)
]


# GENERATE DATASET
X_all = [] # Lists that will contain all samples
V_all = [] # Voltage traces
labels = [] # Corresponding labels

for gK0_center, label in gK0_values: # Loop over bursting classes
    print(f"Generating class {label}")
    for _ in range(Nsim_class):     # Generate Nsim_class simulations
        gK0 = gK0_center 

        t, V = simulate(        # Simulate one burst

            gK0=gK0,
            tau_n=tau_n,
            eps=eps,
            Iapp=Iapp
        )

        X_all.append([        # Store input parameters

            gK0,
            Iapp,
            tau_n,
            eps
        ])

        V_all.append(V)        # Store voltage trace

        labels.append(label)         # Store class label


# CONVERT TO NUMPY ARRAYS
X = np.array(X_all, dtype=np.float32)# Input parameters matrix

# Voltage matrix
U = np.array(V_all, dtype=np.float32)

labels = np.array(labels, dtype=np.int32) # Labels vector



# DATASET INFORMATION
print("Dataset generated without normalization.")

print("Voltage range:")
print("Min:", U.min())
print("Max:", U.max())


# SAVE DATASET
# Save everything in MATLAB format
# X = input parameters
# time = common time vector
# V = voltage traces
# label = burst class
savemat(
    "dataset_raw.mat",
    {
        "X": X,
        "time": t,
        "V": U,
        "label": labels,
    }
)

print("\nDataset generated successfully\n")

print("X :", X.shape)
print("V :", U.shape)
print("Labels :", labels.shape)

# VISUALIZATION (Not necessary, just in case to have an idea about simulated bursts)

fig = go.Figure() # Create an empty Plotly figure

for i in range(min(5, len(U))): # Display up to five simulated bursts

    fig.add_trace(
        go.Scatter(

            x=t,            # Horizontal axis
            y=U[i],         # Voltage signal
            mode="lines",   # Draw a continuous line
            name=f"Sim {i} (label={labels[i]})" # Legend
        )
    )

fig.update_layout( # Improve appearance
    title="Burst Dataset",
    xaxis_title="Time",
    yaxis_title="Voltage (mV)",
    template="plotly_white"
)
fig.show()# Open the interactive figure