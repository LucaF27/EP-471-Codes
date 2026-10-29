# This code is authored by Luca Farmer, 03/26/2026
# Assignment: Homework 5 Problem 1d

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

def solve_heat_conduction(dy):
    dx = 0.1
    nx = int(3 / dx) + 1
    ny = int(4 / dy) + 1
    nxy = nx * ny

    y_coords = np.linspace(-2, 2, ny)
    A = np.zeros((nxy, nxy))
    b = np.zeros(nxy)

    for i in range(0, nx):
        for j in range(1, ny-1):
            k = j * nx + i
            y_val = y_coords[j]
            Q = 10 * np.exp(-y_val**2)
            
            if i == nx - 1:
                A[k,k] = 1
                A[k,k - (nx-1)] = -1
                b[k] = 0
            else:
                A[k, k]    = (2/dx**2 + 2/dy**2)
                A[k, k-nx] = -1/dy**2
                A[k, k+nx] = -1/dy**2
                b[k]       = Q
                
                A[k, k+1] = -1/dx**2
                if i == 0:
                    A[k, k + nx - 2] = -1/dx**2 
                else:
                    A[k, k-1] = -1/dx**2

    for i in range(nx):
        for j in [0, ny-1]:
            k = j * nx + i
            A[k, k] = 1
            b[k] = 300

    Tsolution = la.solve(A, b)
    Z = np.reshape(Tsolution, (ny, nx))
    
    # Extract temperature at x=1.5, y=0
    i_target = int(1.5 / dx)
    j_target = int(2.0 / dy)
    return Z[j_target, i_target]

# Compute Reference Solution
T_ref = solve_heat_conduction(dy=0.025)

# Compute errors for different dy
dy_values = np.array([0.05, 0.1, 0.2, 0.4])
errors = []

for dy in dy_values:
    T_val = solve_heat_conduction(dy)
    error = np.abs(T_val - T_ref)
    errors.append(error)

errors = np.array(errors)

# Calculate order of truncation error (slope of log-log line)
slope, intercept = np.polyfit(np.log(dy_values), np.log(errors), 1)
print(f"Order of truncation error (slope): {slope:.2f}")

# Plot
plt.figure()
plt.loglog(dy_values, errors, 'bo-', label='Computed Error')
plt.loglog(dy_values, np.exp(intercept) * dy_values**slope, 'r--', label=f'Fit Line (slope={slope:.2f})')
plt.xlabel(r'$\Delta y$ (m)')
plt.ylabel('Absolute Error (K)')
plt.title('Truncation Error Analysis')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()