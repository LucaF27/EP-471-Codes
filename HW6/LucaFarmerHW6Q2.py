# This code is authored by Luca Farmer, 04/20/2026
# Assignment: Homework 6, Problem 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import gamma
from scipy.stats import gaussian_kde

# Helper Functions
def get_lognormal_params(mean, std):
    """Converts standard mean and std to lognormal underlying mu and sigma."""
    var = std**2
    mu_log = np.log(mean**2 / np.sqrt(var + mean**2))
    sigma_log = np.sqrt(np.log(1 + var / mean**2))
    return mu_log, sigma_log

def get_weibull_params(mean, std):
    """Solves for Weibull shape (k) and scale (lambda) given mean and std."""
    cov = std / mean
    # Variance/Mean^2 relationship for Weibull
    def objective(k):
        return np.sqrt(gamma(1 + 2/k) / (gamma(1 + 1/k)**2) - 1) - cov
    
    # Solve for shape parameter k
    k_opt = fsolve(objective, 1.0)[0]
    # Solve for scale parameter lambda
    lam_opt = mean / gamma(1 + 1/k_opt)
    return k_opt, lam_opt

# Distribution Parameters Conversion
# Lognormal parameters (L, b, h)
mu_L_log, sig_L_log = get_lognormal_params(18.0, 0.9)
mu_b_log, sig_b_log = get_lognormal_params(1.0, 0.05)
mu_h_log, sig_h_log = get_lognormal_params(1.5, 0.075)

# Weibull parameters (E)
k_E, lam_E = get_weibull_params(30000.0, 3000.0)

# Normal parameters (Q, Delta_L)
mu_Q, sig_Q = 1.888, 0.283
mu_DL, sig_DL = 0.75, 0.042

# Problem 2 a&b
print("- Initial Monte Carlo Simulation:")
N = int(1e6)

# Generate Samples
L_samp = np.random.lognormal(mu_L_log, sig_L_log, N)
b_samp = np.random.lognormal(mu_b_log, sig_b_log, N)
h_samp = np.random.lognormal(mu_h_log, sig_h_log, N)
E_samp = lam_E * np.random.weibull(k_E, N)
Q_samp = np.random.normal(mu_Q, sig_Q, N)
DL_samp = np.random.normal(mu_DL, sig_DL, N)

# Compute Deflection
Delta_samp = (4 * Q_samp * L_samp**3) / (E_samp * b_samp * h_samp**3)

# Limit State Function
g_X = DL_samp - Delta_samp

# Estimate Probability of Failure
N_failed = np.sum(g_X <= 0)
P_f = N_failed / N

print(f"Number of samples (N): {N}")
print(f"Number of failures: {N_failed}")
print(f"Estimated Probability of Failure (P_f): {P_f:.6f}\n")

# Problem 2 c&d
print("- Sample Size Requirements:")
# From lecture: c.o.v.(P_f) = sqrt((1 - P_f) / (N * P_f))
# Rearrange for N

# Part c
error_c = 0.01
N_req_c = (1 - P_f) / (P_f * error_c**2)
print(f"Required samples for 1% error (based on estimated P_f = {P_f:.6f}): {int(np.ceil(N_req_c)):,}")

# Part d
P_f_hypothetical = 1e-10
error_d = 0.02
N_req_d = (1 - P_f_hypothetical) / (P_f_hypothetical * error_d**2)
print(f"Required samples for 2% error if P_f = 1e-10: {int(np.ceil(N_req_d)):,}\n")

# Problem 2e
N_fixed = int(1e4)
N_sim_cases = [10, 100, 1000]

# Prepare a figure for the histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, N_sim in enumerate(N_sim_cases):
    Pf_estimates = np.zeros(N_sim)
    
    for j in range(N_sim):
        # We sample strictly what affects g(X) inside the loop
        L = np.random.lognormal(mu_L_log, sig_L_log, N_fixed)
        b = np.random.lognormal(mu_b_log, sig_b_log, N_fixed)
        h = np.random.lognormal(mu_h_log, sig_h_log, N_fixed)
        E = lam_E * np.random.weibull(k_E, N_fixed)
        Q = np.random.normal(mu_Q, sig_Q, N_fixed)
        DL = np.random.normal(mu_DL, sig_DL, N_fixed)
        
        Delta = (4 * Q * L**3) / (E * b * h**3)
        Pf_estimates[j] = np.sum(DL - Delta <= 0) / N_fixed

    # Stats
    mean_Pf = np.mean(Pf_estimates)
    std_Pf = np.std(Pf_estimates)
    cov_Pf = std_Pf / mean_Pf
    sampling_error = std_Pf / np.sqrt(N_sim)
    
    print(f"- N_sim = {N_sim}:")
    print(f"Mean P_f: {mean_Pf:.6f}")
    print(f"Std Dev of P_f: {std_Pf:.6f}")
    print(f"C.O.V. of P_f: {cov_Pf:.4f}")
    print(f"Sampling Error: {sampling_error:.6f}\n")
    
    # Plotting Histogram and KDE
    ax = axes[i]
    ax.hist(Pf_estimates, bins=20, density=True, color='gray', alpha=0.6, edgecolor='k')
    
    # KDE representation
    x_vals = np.linspace(np.min(Pf_estimates)*0.8, np.max(Pf_estimates)*1.2, 200)
    # Check to prevent KDE crash if all estimates are exactly the same
    if std_Pf > 0:
        kde = gaussian_kde(Pf_estimates)
        ax.plot(x_vals, kde(x_vals), 'r-', linewidth=2)
    
    ax.set_title(f'$N_{{sim}}={N_sim}$')
    ax.set_xlabel('Estimated $P_f$')
    ax.set_ylabel('Density')

plt.suptitle(f'Distribution of Monte Carlo Estimates for $P_f$ (N={N_fixed})')
plt.tight_layout()
plt.show()

# Problem 2f
N_test_values = [100, 1000, 10000, 100000]
N_trials = 100
cov_empirical = []

for N_test in N_test_values:
    Pf_trials = np.zeros(N_trials)
    for j in range(N_trials):
        L = np.random.lognormal(mu_L_log, sig_L_log, N_test)
        b = np.random.lognormal(mu_b_log, sig_b_log, N_test)
        h = np.random.lognormal(mu_h_log, sig_h_log, N_test)
        E = lam_E * np.random.weibull(k_E, N_test)
        Q = np.random.normal(mu_Q, sig_Q, N_test)
        DL = np.random.normal(mu_DL, sig_DL, N_test)
        
        Delta = (4 * Q * L**3) / (E * b * h**3)
        Pf_trials[j] = np.sum(DL - Delta <= 0) / N_test
        
    mean_val = np.mean(Pf_trials)
    std_val = np.std(Pf_trials)
    
    # Avoid division by zero if no failures occur in small N cases
    if mean_val > 0:
        cov_empirical.append(std_val / mean_val)
    else:
        cov_empirical.append(np.nan)

plt.figure(figsize=(8, 5))
plt.plot(N_test_values, cov_empirical, 'o-', color='#0077b6', linewidth=2, markersize=8)
plt.xscale('log')
plt.title('Monte Carlo Convergence: COV of Estimated $P_f$')
plt.xlabel('Number of Samples (N)')
plt.ylabel('COV of Estimated P_f')
plt.grid(True, which="both", ls="--")
plt.show()