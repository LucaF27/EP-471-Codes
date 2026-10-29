# This code is authored by Luca Farmer, 05/01/2026
# Assignment: Final Project
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

#  PHYSICAL PARAMETERS & CONSTANTS
n_base = 1.0e20      
chi_base = 1.0       
a = 1.0              
S0 = 1.0e6           
k_conv = 1.602e-16   # 1 keV = 1.602e-16 J
w_verify = 0.4 * a   # Standard width for verification

#  CORE NUMERICAL SOLVER & ANALYTICAL SOLUTIONS

def solve_plasma_temp(S_profile, N, n, chi, a_val):
    """Steady-state 1D solver using Central Finite Differences."""
    x = np.linspace(-a_val, a_val, N)
    dx = x[1] - x[0]
    A = np.zeros((N, N))
    B = np.zeros(N)

    # Internal Nodes
    for i in range(1, N - 1):
        A[i, i-1], A[i, i], A[i, i+1] = 1.0, -2.0, 1.0
        B[i] = -S_profile[i] * (dx**2) / (n * chi * k_conv)

    # Boundary Conditions: T = 0 at edges
    A[0, 0], B[0] = 1.0, 0.0
    A[N-1, N-1], B[N-1] = 1.0, 0.0
    
    return x, np.linalg.solve(A, B)

def get_analytical_gaussian(x, S0_val, w_val, n, chi, a_val):
    """Exact solution for a Gaussian source S = S0 * exp(-x^2/w^2)."""
    K = S0_val / (n * chi * k_conv)
    # The double integral of exp(-x^2/w^2) involves the erf function
    term_a = a_val * erf(a_val/w_val) + (w_val/np.sqrt(np.pi)) * np.exp(-(a_val/w_val)**2)
    term_x = x * erf(x/w_val) + (w_val/np.sqrt(np.pi)) * np.exp(-(x/w_val)**2)
    return K * (np.sqrt(np.pi) * w_val / 2.0) * (term_a - term_x)

#  DATA GENERATION
# Verification Data
N_verify = 100
x_v = np.linspace(-a, a, N_verify)
S_verify = S0 * np.exp(-(x_v / w_verify)**2)
_, T_num_verify = solve_plasma_temp(S_verify, N_verify, n_base, chi_base, a)
T_ana_verify = get_analytical_gaussian(x_v, S0, w_verify, n_base, chi_base, a)

# Convergence Study Data 
N_list = [10, 20, 40, 80, 160, 320, 640]
errors, dx_list = [], []
for N_test in N_list:
    x_t = np.linspace(-a, a, N_test)
    S_t = S0 * np.exp(-(x_t / w_verify)**2)
    _, T_n = solve_plasma_temp(S_t, N_test, n_base, chi_base, a)
    T_e = get_analytical_gaussian(x_t, S0, w_verify, n_base, chi_base, a)
    dx_list.append(x_t[1] - x_t[0])
    errors.append(np.max(np.abs(T_n - T_e)))

# Sensitivity Study Data (N=100) 
S_uni = np.full(N_verify, S0)
# Peaked source
S_pk = S0 * np.exp(-(x_v / (0.2 * a))**2)
S_pk *= np.sum(S_uni) / np.sum(S_pk) # Normalize total power
# Off-axis source
S_off = S0 * (np.exp(-((x_v - 0.5*a)/0.1)**2) + np.exp(-((x_v + 0.5*a)/0.1)**2))
S_off *= np.sum(S_uni) / np.sum(S_off) # Normalize total power

_, T_uni = solve_plasma_temp(S_uni, N_verify, n_base, chi_base, a)
_, T_peaked = solve_plasma_temp(S_pk, N_verify, n_base, chi_base, a)
_, T_off_axis = solve_plasma_temp(S_off, N_verify, n_base, chi_base, a)

# FIGURE 1: Verification
plt.figure(1, figsize=(7, 5))
plt.plot(x_v, T_ana_verify, 'k-', lw=4, alpha=0.3, label='Analytical (Exact)')
plt.plot(x_v, T_num_verify, 'r--', lw=2, label='Numerical (FDM)')
plt.title('Figure 1: Numerical Solver Verification\n(Gaussian Source, w=0.4a)')
plt.xlabel('Position x [m]'); plt.ylabel('Temperature T [keV]'); plt.legend(); plt.grid(True, ls=':')

# FIGURE 2: Convergence
plt.figure(2, figsize=(7, 5))
plt.loglog(dx_list, errors, 'bo-', label='Measured Error')
plt.loglog(dx_list, [errors[0]*(d/dx_list[0])**2 for d in dx_list], 'k--', label='Theoretical $O(\Delta x^2)$')
plt.title('Figure 2: Grid Convergence Analysis'); plt.xlabel('Grid Spacing $\Delta x$ [m]'); plt.ylabel('Max Absolute Error ($L_\infty$) [keV]'); plt.legend(); plt.grid(True, which="both", ls=":")

# FIGURE 3: Heating Profile Comparison
plt.figure(3, figsize=(7, 5))
plt.plot(x_v, T_uni, 'gray', label='Uniform Heating')
plt.plot(x_v, T_peaked, 'crimson', label='On-Axis Peaked')
plt.plot(x_v, T_off_axis, 'teal', label='Off-Axis (Double-Hump)')
plt.title('Figure 3: Impact of Heating Profiles (Constant Power)'); plt.xlabel('x [m]'); plt.ylabel('T [keV]'); plt.legend(); plt.grid(True, ls=':')

# FIGURE 4: Chi Sensitivity
plt.figure(4, figsize=(7, 5))
for m in [0.5, 1.0, 2.0, 4.0]:
    c = chi_base * m
    _, T_val = solve_plasma_temp(S_uni, N_verify, n_base, c, a)
    plt.plot(x_v, T_val, label=f'chi = {c}')
plt.title('Figure 4: Sensitivity to Thermal Diffusivity'); plt.xlabel('x [m]'); plt.ylabel('T [keV]'); plt.legend(); plt.grid(True, ls=':')

plt.show()