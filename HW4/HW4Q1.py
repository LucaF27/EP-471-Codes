# This code is authored by Luca Farmer, 03/19/2026
# Assignment: Homework 4, Problem 1 a & b

import numpy as np
import matplotlib.pyplot as plt

# Parameters and discretization 
L = 10.0
alpha = 0.1
dTdx_0 = -200.0  
dTdx_L = 200.0   
T_initial = 300.0

dx = 0.5
nx = int(L / dx) + 1
xmesh = np.linspace(0, L, nx)

# Stability and Time Stepping
dt_max = (dx**2) / (2 * alpha)
dt = 0.5 * dt_max  # dt = 0.625 s
r = alpha * dt / (dx**2)

t_end = 200.0
nt = int(np.round(t_end / dt)) 

# Dictionaries to store temperature profiles at specific times
eval_times = [0.0, 100.0, 200.0]
results_ftcs = {}
results_btcs = {}

# Initialize arrays
Tk_ftcs = np.full(nx, T_initial)
Tk_btcs = np.full(nx, T_initial)

# Store t=0 states
results_ftcs[0.0] = Tk_ftcs.copy()
results_btcs[0.0] = Tk_btcs.copy()

# Construct BTCS Matrix A

A = np.zeros((nx, nx))
for i in range(1, nx - 1):
    A[i, i-1] = -r
    A[i, i]   = 1 + 2 * r
    A[i, i+1] = -r

# Left boundary (Neumann)
A[0, 0] = 1 + 2 * r
A[0, 1] = -2 * r

# Right boundary (Neumann)
A[-1, -2] = -2 * r
A[-1, -1] = 1 + 2 * r

# Time Marching Loop

current_time = 0.0

for k in range(1, nt + 1):
    current_time += dt
    
    # FTCS Update
    Tk1_ftcs = np.zeros(nx)
    Tk1_ftcs[0] = (1 - 2*r)*Tk_ftcs[0] + 2*r*Tk_ftcs[1] - 2*r*dx*dTdx_0
    
    for i in range(1, nx - 1):
        Tk1_ftcs[i] = Tk_ftcs[i] + r * (Tk_ftcs[i-1] - 2 * Tk_ftcs[i] + Tk_ftcs[i+1])
        
    Tk1_ftcs[-1] = (1 - 2*r)*Tk_ftcs[-1] + 2*r*Tk_ftcs[-2] + 2*r*dx*dTdx_L
    Tk_ftcs = Tk1_ftcs.copy()
    
    # BTCS Update
    B = Tk_btcs.copy()
    B[0]  -= 2 * r * dx * dTdx_0
    B[-1] += 2 * r * dx * dTdx_L
    Tk_btcs = np.linalg.solve(A, B)
    
    # Save results at target times
    # Using a small tolerance for floating point comparisons
    for t_target in [100.0, 200.0]:
        if abs(current_time - t_target) < 1e-5:
            results_ftcs[t_target] = Tk_ftcs.copy()
            results_btcs[t_target] = Tk_btcs.copy()

# Plotting Results

plt.figure(figsize=(10, 6))

# Plot FTCS (Solid lines with circles)
plt.plot(xmesh, results_ftcs[0.0], 'bo-', label='FTCS (t=0s)', markevery=2)
plt.plot(xmesh, results_ftcs[100.0], 'go-', label='FTCS (t=100s)', markevery=2)
plt.plot(xmesh, results_ftcs[200.0], 'ro-', label='FTCS (t=200s)', markevery=2)

# Plot BTCS (Dashed lines with squares)
plt.plot(xmesh, results_btcs[0.0], 'bs--', label='BTCS (t=0s)', markevery=2, mfc='none')
plt.plot(xmesh, results_btcs[100.0], 'gs--', label='BTCS (t=100s)', markevery=2, mfc='none')
plt.plot(xmesh, results_btcs[200.0], 'rs--', label='BTCS (t=200s)', markevery=2, mfc='none')

plt.title('Temperature Evolution in Beam (FTCS vs BTCS)')
plt.xlabel('Position x (m)')
plt.ylabel('Temperature T (K)')
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()