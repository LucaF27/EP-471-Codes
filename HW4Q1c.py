# This code is authored by Luca Farmer, 03/20/2026
# Assignment: Homework 4, Problem 1c

import numpy as np
import matplotlib.pyplot as plt

L = 10.0
k_cond = 0.1
q0 = -200.0
qL = 200.0
T_inf = 300.0
dx = 0.5
t_end = 200.0
dt = 0.625 

x = np.arange(0, L + dx, dx)
nx = len(x)
nt = int(t_end / dt)
r = k_cond * dt / (dx**2)

h_values = [0.0, 0.025, 0.05]
results = {}

for h in h_values:
    A = np.zeros((nx, nx))
    diag_val = 1 + 2*r + h*dt
    
    for i in range(1, nx - 1):
        A[i, i-1] = -r
        A[i, i] = diag_val
        A[i, i+1] = -r
        
    A[0, 0] = diag_val
    A[0, 1] = -2*r
    A[-1, -2] = -2*r
    A[-1, -1] = diag_val
    
    T = np.full(nx, 300.0)
    
    for n in range(nt):
        B = T.copy() + h*dt*T_inf
        B[0] -= 2*r*dx*q0
        B[-1] += 2*r*dx*qL
        T = np.linalg.solve(A, B)
        
    results[h] = T.copy()

plt.figure(figsize=(10, 6))
for h, T_final in results.items():
    plt.plot(x, T_final, label=f'h = {h} s⁻¹')

plt.title('BTCS Temperature Distribution at t = 200 s with Convection')
plt.xlabel('Position x (m)')
plt.ylabel('Temperature T (K)')
plt.legend()
plt.grid(True)
plt.show()