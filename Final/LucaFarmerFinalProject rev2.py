import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PHYSICAL PARAMETERS & CONSTANTS
# =============================================================================
n_base = 1.0e20      # Plasma density (m^-3)
chi_base = 1.0       # Thermal diffusivity (m^2/s)
a = 1.0              # Slab half-width (m)
S0 = 1.0e6           # Baseline volumetric heating source (W/m^3)

# 1 keV = 1.602e-19 J/eV * 1000 eV/keV = 1.602e-16 J
k_conv = 1.602e-16   

# =============================================================================
# 2. CORE NUMERICAL SOLVER
# =============================================================================
def solve_plasma_temp(S_profile, N, n, chi, a_val):
    """
    Solves the steady-state 1D energy balance equation:
    -d/dx (n * chi * dT/dx) = S(x)
    Returns: x (grid), T (Temperature in keV)
    """
    x = np.linspace(-a_val, a_val, N)
    dx = x[1] - x[0]
    
    # Constructing the Tridiagonal Matrix A
    A = np.zeros((N, N))
    B = np.zeros(N)

    # Finite Difference Approximation (Central Difference)
    for i in range(1, N - 1):
        A[i, i - 1] = 1.0
        A[i, i]     = -2.0
        A[i, i + 1] = 1.0
        # Source term normalized by density, transport, and keV conversion
        B[i] = -S_profile[i] * (dx**2) / (n * chi * k_conv)

    # Dirichlet Boundary Conditions: T(-a) = T(a) = 0
    A[0, 0] = 1.0
    B[0] = 0.0
    A[N-1, N-1] = 1.0
    B[N-1] = 0.0

    T_keV = np.linalg.solve(A, B)
    return x, T_keV

def get_analytical(x, S, n, chi, a_val):
    """Analytical parabolic solution for uniform heating cases."""
    T0 = (S * a_val**2) / (2 * n * chi * k_conv)
    return T0 * (1 - (x**2 / a_val**2))

# =============================================================================
# 3. ANALYSIS DATA GENERATION
# =============================================================================

# --- 3a. Convergence Study Data ---
N_list = [10, 20, 40, 80, 160, 320]
errors = []
dx_list = []
for N_test in N_list:
    S_uni = np.full(N_test, S0)
    x_c, T_num = solve_plasma_temp(S_uni, N_test, n_base, chi_base, a)
    T_exact = get_analytical(x_c, S0, n_base, chi_base, a)
    
    dx_list.append(x_c[1] - x_c[0])
    errors.append(np.max(np.abs(T_num - T_exact)))

# --- 3b. Sensitivity Study Data (N=100) ---
N_fixed = 100
x_f = np.linspace(-a, a, N_fixed)
S_uni = np.full(N_fixed, S0)

# Case 1: Peaked Heating
S_peak = S0 * np.exp(-(x_f / (0.25 * a))**2)
S_peak *= np.sum(S_uni) / np.sum(S_peak) # Maintain power balance

# Case 2: Off-Axis Heating (Double Gaussian)
S_off = S0 * (np.exp(-((x_f - 0.5*a) / 0.15)**2) + np.exp(-((x_f + 0.5*a) / 0.15)**2))
S_off *= np.sum(S_uni) / np.sum(S_off) # Maintain power balance

_, T_uni = solve_plasma_temp(S_uni, N_fixed, n_base, chi_base, a)
_, T_peak = solve_plasma_temp(S_peak, N_fixed, n_base, chi_base, a)
_, T_off = solve_plasma_temp(S_off, N_fixed, n_base, chi_base, a)

# =============================================================================
# 4. FIGURES FOR REPORT
# =============================================================================

# FIGURE 1: Solver Verification
plt.figure(1, figsize=(7, 5))
plt.plot(x_f, get_analytical(x_f, S0, n_base, chi_base, a), 'k-', lw=4, alpha=0.3, label='Analytical')
plt.plot(x_f, T_uni, 'r--', lw=2, label='Numerical (N=100)')
plt.title('Figure 1: Numerical vs Analytical Verification')
plt.xlabel('Position x [m]')
plt.ylabel('Temperature T [keV]')
plt.grid(True, ls=':', alpha=0.6)
plt.legend()

# FIGURE 2: Convergence Study
plt.figure(2, figsize=(7, 5))
plt.loglog(dx_list, errors, 'bo-', label='Max Error ($L_\infty$)')
plt.loglog(dx_list, [errors[0]*(d/dx_list[0])**2 for d in dx_list], 'k--', label='Theoretical $\mathcal{O}(\Delta x^2)$')
plt.title('Figure 2: Grid Convergence Analysis')
plt.xlabel('Grid Spacing $\Delta x$ [m]')
plt.ylabel('Maximum Absolute Error [keV]')
plt.grid(True, which="both", ls=":", alpha=0.6)
plt.legend()

# FIGURE 3: Heating Profile Sensitivity
plt.figure(3, figsize=(7, 5))
plt.plot(x_f, T_uni, label='Uniform Heating', color='gray')
plt.plot(x_f, T_peak, label='On-Axis (Peaked)', color='crimson')
plt.plot(x_f, T_off, label='Off-Axis (Double-Hump)', color='teal')
plt.title('Figure 3: Impact of Heat Deposition Shape')
plt.xlabel('Position x [m]')
plt.ylabel('Temperature T [keV]')
plt.grid(True, ls=':', alpha=0.6)
plt.legend()

# FIGURE 4: Chi Sensitivity
plt.figure(4, figsize=(7, 5))
for mult in [0.5, 1.0, 2.0, 4.0]:
    chi_val = chi_base * mult
    _, T_val = solve_plasma_temp(S_uni, N_fixed, n_base, chi_val, a)
    plt.plot(x_f, T_val, label=f'$\chi$ = {chi_val} $m^2/s$')
plt.title('Figure 4: Sensitivity to Thermal Diffusivity ($\chi$)')
plt.xlabel('Position x [m]')
plt.ylabel('Temperature T [keV]')
plt.grid(True, ls=':', alpha=0.6)
plt.legend()

# PRINT DATA SUMMARY TABLE FOR RESULTS SECTION
print(f"{'Heating Profile':<20} | {'Peak T [keV]':<15}")
print("-" * 40)
print(f"{'Uniform':<20} | {np.max(T_uni):.4f}")
print(f"{'Peaked (On-Axis)':<20} | {np.max(T_peak):.4f}")
print(f"{'Off-Axis':<20} | {np.max(T_off):.4f}")

plt.show()