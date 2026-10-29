# This code is authored by Luca Farmer, 03/25/2026
# Assignment: Homework 5 Problem 1b

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from matplotlib import cm

# Parameters and discretization

x_range = [0, 3]
y_range = [-2, 2]
dx, dy = 0.1, 0.1

nx = int((x_range[1] - x_range[0]) / dx) + 1
ny = int((y_range[1] - y_range[0]) / dy) + 1
nxy = nx * ny

x_coords = np.linspace(x_range[0], x_range[1], nx)
y_coords = np.linspace(y_range[0], y_range[1], ny)

A = np.zeros((nxy, nxy))
b = np.zeros(nxy)

# Interior Nodes
for i in range(1, nx-1):
    for j in range(1, ny-1):
        k = j * nx + i
        y_val = y_coords[j]
        Q = 10 * np.exp(-y_val**2)
        
        A[k, k]    = (2/dx**2 + 2/dy**2)
        A[k, k-1]  = -1/dx**2
        A[k, k+1]  = -1/dx**2
        A[k, k-nx] = -1/dy**2
        A[k, k+nx] = -1/dy**2
        b[k]       = Q

# Boundary conditions
for i in range(nx):
    for j in range(ny):
        k = j * nx + i
        
        # Left and Right: Dirichlet
        if i == 0 or i == nx-1:
            A[k, :] = 0
            A[k, k] = 1
            b[k] = 300
            
        # Bottom: Neumann (Forward Difference)
        elif j == 0:
            if i != 0 and i != nx-1:
                A[k, :] = 0
                A[k, k] = 1
                A[k, k+nx] = -1 
                b[k] = 0
                
        # Top: Neumann (Backward Difference)
        elif j == ny-1:
            if i != 0 and i != nx-1:
                A[k, :] = 0
                A[k, k] = 1
                A[k, k-nx] = -1
                b[k] = 0

Tsolution = la.solve(A, b)
Z = np.reshape(Tsolution, (ny, nx))
X, Y = np.meshgrid(x_coords, y_coords)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(9, 7))
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=True, edgecolor='none')

ax.set_xlabel('X (m)', labelpad=10)
ax.set_ylabel('Y (m)', labelpad=10)
ax.set_zlabel('Temperature (K)', labelpad=10)
ax.set_title('Part B: Dirichlet(x) and Neumann(y) BCs') 
# Set consistent limits for visual comparison
ax.set_zlim(300, 360) 
ax.view_init(elev=30, azim=-60)

fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1, label='Temp (K)')

plt.tight_layout()
plt.show()