# EP 471: Intermediate Problem Solving for Engineers

Python code and written reports from **EP 471** at the University of Wisconsin–Madison (Spring 2026). The course covers numerical methods for differential equations and Monte Carlo methods for engineering problems. Each folder holds the code and the PDF report for one assignment.

**Author:** Luca Farmer, Nuclear Engineering, UW–Madison

## Repository Contents

| Folder | Topic | Methods |
|---|---|---|
| [`HW4`](HW4) | Parabolic and hyperbolic PDEs | FTCS and BTCS heat conduction, wave equation |
| [`HW5`](HW5) | Elliptic PDEs | 2D Poisson equation with Dirichlet, Neumann and periodic boundaries |
| [`HW6`](HW6) | Monte Carlo methods | Inverse transform and rejection sampling, probability of failure |
| [`Final-Project`](Final-Project) | 1D plasma slab thermal conduction | Finite difference solver, verification and sensitivity studies |

---

## Final Project: Numerical Analysis of 1D Thermal Conduction in a Plasma Slab

A finite difference solver for the steady-state energy balance in a 1D plasma slab (−a ≤ x ≤ a). The model follows the treatment in Freidberg, *Plasma Physics and Fusion Energy* (2007), and is relevant to tokamak power balance and first-wall heat loads.

**Governing equation:** `d/dx (n χ dT/dx) + S = 0`, with `T(±a) = 0`

**What the project does**
- Solves the boundary value problem with second-order central differences and a dense linear solver (`numpy.linalg.solve`)
- **Verifies** the solver against an exact analytical solution (involving the error function) for a Gaussian heating source
- Runs a **grid convergence study** showing second-order (O(Δx²)) accuracy
- Studies how **heating profile shape** (uniform, on-axis peaked, off-axis double-Gaussian) affects peak temperature at constant total power
- Studies the sensitivity to **thermal diffusivity** χ

**Selected results** (n = 1×10²⁰ m⁻³, S₀ = 1 MW/m³)

| Heating profile | Peak temperature (keV) |
|---|---|
| Uniform | 31.21 |
| On-axis peaked | 55.93 |
| Off-axis (double hump) | 31.53 |

| χ (m²/s) | Peak temperature (keV) |
|---|---|
| 0.5 | 62.42 |
| 1.0 | 31.21 |
| 2.0 | 15.61 |
| 4.0 | 7.80 |

Peak temperature scales as 1/χ, as the analytical solution predicts.

---

## Homework 4: Parabolic and Hyperbolic PDEs

**Heat transport in a beam**
- Compares the explicit FTCS and implicit BTCS schemes for 1D heat conduction with fixed heat flux (Neumann) boundaries, implemented with ghost nodes
- Derives the FTCS stability limit (r ≤ 0.5), giving Δt_max = 1.25 s, and uses Δt = 0.625 s
- Adds a convection term to the BTCS scheme and shows how it lowers the beam temperature for h = 0, 0.025 and 0.05 s⁻¹

**Wave equation**
- Derives the analytical solution for a vibrating string (wave speed a = 2)
- Compares first-order and second-order initialization of the first time step
- A log-log error study at constant Courant number shows slopes of about 1 (first-order start) and about 2 (second-order start)

## Homework 5: Elliptic PDEs

Solves the steady-state 2D Poisson equation, `−∇²T = Q`, with a 5-point central difference stencil assembled into a sparse-style linear system `A·T = b`.

- **Part A:** Dirichlet boundaries with a parabolic temperature profile on the side walls
- **Part B:** Mixed boundaries (Dirichlet in x, zero-flux Neumann in y)
- **Part C:** Periodic boundaries in x with Dirichlet in y
- **Part D:** Truncation error study showing a fitted slope of about 2.14, consistent with second-order accuracy

## Homework 6: Monte Carlo Methods

**Radioactive decay of Bismuth-212**
- Derives the CDF, half-life relation, mean lifetime and standard deviation of the exponential distribution
- Generates decay times with **inverse transform sampling** and **rejection sampling**, and compares the estimated PDFs to the analytical solution
- Shows that the relative error of the mean falls as 1/√N, so about 10,000 samples are needed for 1% error

**Probability of failure for a cantilever beam**
- Formulates a limit state function for deflection failure and estimates the probability of failure by Monte Carlo simulation (P_f ≈ 0.030 with 10⁶ samples)
- Derives the sample size needed for a target coefficient of variation, and examines how the spread of repeated estimates narrows as the number of runs increases

---

## Requirements

- Python 3
- NumPy
- SciPy
- Matplotlib

Run a script from its folder, for example:

```
python final_project.py
```

The final project script displays four figures: solver verification, grid convergence, heating profile comparison and diffusivity sensitivity.

## Notes

These are course assignments written for learning and reporting, not production code. The PDF in each folder explains the derivations, assumptions and results in full.
