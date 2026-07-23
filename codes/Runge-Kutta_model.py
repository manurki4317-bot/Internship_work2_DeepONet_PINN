import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PARAMETERS
# =====================================================

C = 1.0
Iapp = 4.3

VNa = 60.0
VK  = -90.0
VL  = -80.0

gL = 8.0

tau_n = 0.17

V12m = -20.0
km = 15.0

V12n = -25.0
kn = 5.0

eps = 0.005

gNa0 = 20.0

# =====================================================
# ACTIVATION FUNCTIONS
# =====================================================

def m_inf(V):
    return 1.0 / (1.0 + np.exp((V12m - V)/km))

def n_inf(V):
    return 1.0 / (1.0 + np.exp((V12n - V)/kn))

# =====================================================
# ODE SYSTEM
# =====================================================

def rhs(state, gK0):

    V, n, gNa, gK = state

    dV = (Iapp -gL * (V-VL) - gNa * (m_inf(V)) * (V - VNa) - gK * n * (V - VK)) / C

    dn = (n_inf(V) - n) / tau_n

    dgNa = eps * (gK0 - gK)
    dgK  = eps * (gNa - gNa0)

    return np.array([dV, dn, dgNa, dgK])

# =====================================================
# RK4 STEP
# =====================================================

def rk4_step(state, dt, gK0):

    k1 = rhs(state, gK0)
    k2 = rhs(state + 0.5 * dt * k1, gK0)
    k3 = rhs(state + 0.5 * dt * k2, gK0)
    k4 = rhs(state + dt * k3, gK0)

    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# =====================================================
# SIMULATION FUNCTION
# =====================================================

def simulate(gK0, T=1000, dt=0.05):

    N = int(T/dt)

    time = np.zeros(N+1)
    V_trace = np.zeros(N+1)

    # initial conditions
    state = np.array([-60.0, 0.0, gNa0+1, gK0])

    V_trace[0] = state[0]

    for i in range(N):

        state = rk4_step(state, dt, gK0)

        V_trace[i+1] = state[0]
        time[i+1] = time[i] + dt

    return time, V_trace

# =====================================================
# RUN DIFFERENT BURSTING REGIMES
# =====================================================

gK0_values = {
    "Parabolic (11.0)": 11.0,
    "Triangular (9.75)": 9.75,
    "Square-wave (9.0)": 9.0
}

plt.figure(figsize=(12,8))

for i, (label, gK0) in enumerate(gK0_values.items()):

    t, V = simulate(gK0)

    plt.subplot(3,1,i+1)
    plt.plot(t, V)
    plt.title(label)
    plt.ylabel("V (mV)")

plt.xlabel("Time")
plt.tight_layout()
plt.show()