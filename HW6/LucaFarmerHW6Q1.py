# This code is authored by Luca Farmer, 04/16/2026
# Assignment: Homework 6, Problem 1
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Problem 1 a&b
t_half = 60.55  # half-life in minutes
lam = np.log(2) / t_half  # decay constant 

# Analytical mean (expected lifetime) and standard deviation
mu_analytical = 1 / lam
std_analytical = 1 / lam

print("Analytical Results:")
print(f"Decay constant (lambda): {lam:.4f} min^-1")
print(f"Expected Lifetime (Mean): {mu_analytical:.2f} min")
print(f"Standard Deviation: {std_analytical:.2f} min\n")

def pdf_exponential(t):
    return lam * np.exp(-lam * t)

# Problem 1c

t_vals = np.linspace(0, 500, 1000)
pdf_vals = pdf_exponential(t_vals)

plt.figure(figsize=(8, 5))
plt.plot(t_vals, pdf_vals, 'k-', linewidth=2, label='Analytical PDF')
plt.axvline(t_half, color='b', linestyle='--', label=f'Half-life ($t_{{1/2}}$) = {t_half:.2f} min')
plt.axvline(mu_analytical, color='r', linestyle='--', label=f'Expected Life ($\mu$) = {mu_analytical:.2f} min')
plt.title('Analytical PDF of Bismuth-212 Radioactive Decay')
plt.xlabel('Time (minutes)')
plt.ylabel('Probability Density $f(t)$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
#%%
# Problem 1 (d): Inverse Transform Sampling
N_samples = int(1e5)

# Generate uniform random numbers u ~ U(0,1)
u = np.random.rand(N_samples)

# Apply inverse CDF
# F(t) = 1 - exp(-lambda * t)  =>  t = -ln(1-u) / lambda
t_inv_transform = - (1 / lam) * np.log(1 - u)

mu_inv = np.mean(t_inv_transform)
std_inv = np.std(t_inv_transform)

print("Inverse Transform Sampling:")
print(f"Estimated Mean: {mu_inv:.2f} min")
print(f"Estimated Std Dev: {std_inv:.2f} min\n")

# Plot Inverse Transform Results
plt.figure(figsize=(8, 5))
plt.hist(t_inv_transform, bins=100, density=True, color='blue', alpha=0.4, label='Inverse Transform Samples')
kde_inv = gaussian_kde(t_inv_transform)
plt.plot(t_vals, kde_inv(t_vals), 'b-', linewidth=1.5, label='Estimated PDF (KDE)')
plt.plot(t_vals, pdf_vals, 'r--', linewidth=2, label='Analytical PDF')
plt.title('Inverse Transform Sampling: Exponential Decay')
plt.xlabel('Time (minutes)')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
#%%
# Problem 1e
t_max = 1000  # Chosen large enough to encompass the decay curve
f_max = lam   # Maximum of the exponential PDF occurs at t = 0
g_pdf = 1 / t_max
M = f_max / g_pdf  # Ensures f(t) <= M * g(t)

samples_rej = []
total_generated = 0
total_accepted = 0

# Vectorized rejection sampling for performance
while total_accepted < N_samples:
    chunk_size = N_samples
    
    t_candidate = np.random.uniform(0, t_max, chunk_size)
    
    
    p_accept = pdf_exponential(t_candidate) / f_max
    
    # Draw uniform random numbers for acceptance test
    u_test = np.random.rand(chunk_size)
    
    
    accepted = t_candidate[u_test <= p_accept]
    samples_rej.extend(accepted)
    
    total_generated += chunk_size
    total_accepted += len(accepted)

# Trim to exactly N_samples
t_rejection = np.array(samples_rej[:N_samples])

mu_rej = np.mean(t_rejection)
std_rej = np.std(t_rejection)
acceptance_rate = N_samples / total_generated

print("Rejection Sampling:")
print(f"Estimated Mean: {mu_rej:.2f} min")
print(f"Estimated Std Dev: {std_rej:.2f} min")
print(f"Total Candidates Generated: {total_generated}")
print(f"Acceptance Rate: {acceptance_rate:.4f}")

# Plot Rejection Sampling Results
plt.figure(figsize=(8, 5))
plt.hist(t_rejection, bins=100, density=True, color='green', alpha=0.4, label='Rejection Samples')
kde_rej = gaussian_kde(t_rejection)
plt.plot(t_vals, kde_rej(t_vals), 'g-', linewidth=1.5, label='Estimated PDF (KDE)')
plt.plot(t_vals, pdf_vals, 'r--', linewidth=2, label='Analytical PDF')
plt.title('Rejection Sampling: Exponential Decay')
plt.xlabel('Time (minutes)')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# Problem 1f 
# Target relative error for the mean is 1% (0.01)
# Standard Error (SE) = std / sqrt(N). Relative Error = SE / mu
# Relative Error = 1 / sqrt(N)
target_error = 0.01
N_required = int(1 / (target_error**2))

print("Standard Error Analysis:")
print(f"Required samples for 1% relative error: {N_required}")

# Verify via Monte Carlo over various N
N_values = np.logspace(2, 6, 15).astype(int)
relative_errors_empirical = []

for N in N_values:
    # Run multiple times to estimate the empirical standard error of the mean
    means = []
    for _ in range(50):
        u_temp = np.random.rand(N)
        t_temp = - (1 / lam) * np.log(1 - u_temp)
        means.append(np.mean(t_temp))
    
    emp_std_error = np.std(means)
    emp_rel_error = emp_std_error / mu_analytical
    relative_errors_empirical.append(emp_rel_error)

analytical_rel_error = 1 / np.sqrt(N_values)

plt.figure(figsize=(8, 5))
plt.loglog(N_values, analytical_rel_error, 'k-', linewidth=2, label='Analytical $1/\sqrt{N}$')
plt.loglog(N_values, relative_errors_empirical, 'bo', label='Empirical Relative Error')
plt.axhline(target_error, color='r', linestyle='--', label='1% Error Target')
plt.axvline(N_required, color='r', linestyle=':', label=f'N = {N_required}')
plt.title('Relative Error of the Mean vs. Sample Size (N)')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Relative Error')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()