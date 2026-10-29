# This code is authored by Luca Farmer, 03/20/2026
# Assignment: Homework 4, Problem 2

import numpy as np
import matplotlib.pyplot as plt

# Parameters and Discretization
L = 1.0           # length of domain 
te = 0.3          # target time for comparison 
a = 2.0           # wave speed derived from PDE and analytical form 
A = 1.0           # constant derived from initial velocity 
B = 1.0           # constant derived from initial displacement 

dx = 0.1          # spatial step 
dt = 0.05         # time step 
nx = int(L/dx + 1)
nt = int(te/dt + 1)
CFL = a * dt / dx

xmesh = np.linspace(0, L, nx)

# Analytical Solution Function

def analytical_u(x, t):
    # u = sin(2*pi*x) * (A*sin(4*pi*t) + B*cos(4*pi*t)) 
    return np.sin(2 * np.pi * x) * (A * np.sin(4 * np.pi * t) + B * np.cos(4 * np.pi * t))


# Numerical Solution Initialization

# arrays for Method 1 (1st-order at t1) and Method 2 (2nd-order at t1)
u1 = np.zeros((nt, nx))
u2 = np.zeros((nt, nx))

# Initial Conditions (t = 0) 
for i in range(nx):
    xi = i * dx
    u1[0, i] = np.sin(2 * np.pi * xi)
    u2[0, i] = np.sin(2 * np.pi * xi)


# Timestep 1 Calculation

for i in range(1, nx - 1):
    xi = i * dx
    v0 = 4 * np.pi * np.sin(2 * np.pi * xi) # initial velocity 
    
    # Method 1: First-order forward time scheme 
    u1[1, i] = u1[0, i] + dt * v0
    
    # Method 2: Second-order correction 
    # Includes d2u/dt2 = a^2 * d2u/dx2
    d2udx2 = (u2[0, i+1] - 2*u2[0, i] + u2[0, i-1]) / dx**2
    u2[1, i] = u2[0, i] + dt * v0 + 0.5 * (dt**2) * (a**2 * d2udx2)

# Boundary Conditions (Dirichlet) 
u1[:, 0] = u1[:, -1] = 0
u2[:, 0] = u2[:, -1] = 0

# Main Time-Stepping Loop (Central Difference)

for k in range(1, nt - 1):
    for i in range(1, nx - 1):
        # standard 2nd-order leapfrog scheme
        u1[k+1, i] = 2*(1 - CFL**2)*u1[k, i] + CFL**2*(u1[k, i+1] + u1[k, i-1]) - u1[k-1, i]
        u2[k+1, i] = 2*(1 - CFL**2)*u2[k, i] + CFL**2*(u2[k, i+1] + u2[k, i-1]) - u2[k-1, i]

# Plotting Results for Parts (b) and (c)
plt.figure(figsize=(8, 5))
u_exact = analytical_u(xmesh, te)

plt.plot(xmesh, u_exact, 'k-', label='Analytical')
plt.plot(xmesh, u1[-1, :], 'ro--', label='1st-order at t1 (Part b)')
plt.plot(xmesh, u2[-1, :], 'bs--', label='2nd-order at t1 (Part c)')

plt.xlabel('position x (-)')
plt.ylabel('displacement u (-)')
plt.legend()
plt.grid(True)
plt.show()

# Part (d): Error Analysis
dts = [0.005, 0.0025, 0.001, 0.0005] # test timesteps 
ratio = dt / dx # fixed ratio 

errors1 = []
errors2 = []

for dt_test in dts:
    dx_test = dt_test / ratio
    nx_t = int(L/dx_test + 1)
    nt_t = int(te/dt_test + 1)
    CFL_t = a * dt_test / dx_test
    
    x_t = np.linspace(0, L, nx_t)
    ua1 = np.zeros((nt_t, nx_t))
    ua2 = np.zeros((nt_t, nx_t))
    
    # Initial Condition
    ua1[0, :] = ua2[0, :] = np.sin(2 * np.pi * x_t)
    
    # Timestep 1
    v0_t = 4 * np.pi * np.sin(2 * np.pi * x_t)
    for i in range(1, nx_t - 1):
        # Method 1
        ua1[1, i] = ua1[0, i] + dt_test * v0_t[i]
        # Method 2
        d2x = (ua2[0, i+1] - 2*ua2[0, i] + ua2[0, i-1]) / dx_test**2
        ua2[1, i] = ua2[0, i] + dt_test * v0_t[i] + 0.5 * dt_test**2 * a**2 * d2x
        
    # Time Loop
    for k in range(1, nt_t - 1):
        for i in range(1, nx_t - 1):
            ua1[k+1, i] = 2*(1-CFL_t**2)*ua1[k, i] + CFL_t**2*(ua1[k, i+1] + ua1[k, i-1]) - ua1[k-1, i]
            ua2[k+1, i] = 2*(1-CFL_t**2)*ua2[k, i] + CFL_t**2*(ua2[k, i+1] + ua2[k, i-1]) - ua2[k-1, i]
            
    # Max error at end of simulation 
    u_exact_t = analytical_u(x_t, (nt_t-1)*dt_test)
    errors1.append(np.max(np.abs(ua1[-1, :] - u_exact_t)))
    errors2.append(np.max(np.abs(ua2[-1, :] - u_exact_t)))

# Log-Log Plot 
plt.figure(figsize=(8, 5))
plt.loglog(dts, errors1, 'ro-', label='Method (b): 1st-order start')
plt.loglog(dts, errors2, 'bs-', label='Method (c): 2nd-order start')
plt.xlabel('Timestep (dt)')
plt.ylabel('Max Error in u')
plt.title('Error Analysis at t = 0.3s (Log-Log)')
plt.legend()
plt.grid(True, which="both", ls="-")
plt.show()