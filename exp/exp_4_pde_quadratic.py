"""
Experiment 4 — Nonlinear PDE vs. Quadratic Benchmark

Demonstrates that NonlinearPDESolver recovers the analytical QuadraticBenchmark
solution on the interior of the grid, and explores how price-impact parameters
affect the initial capital X_0^*.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pde import QuadraticBenchmark, NonlinearPDESolver

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
S0 = 100.0
T = 1.0
r = 0.05
sigma = 0.20
LAM = 0.10
VARPHI = 0.06
ALPHA = 1e-4
BETA = 0.0
GAMMA = 0.0
N = 1.0

assert 2 * VARPHI >= LAM, "Admissibility violated"

# ── Section 1: Solve QuadraticBenchmark ODE ───────────────────────────────────
bench = QuadraticBenchmark(S0=S0, T=T, r=r, sigma=sigma, lam=LAM,
                            varphi=VARPHI, alpha=ALPHA, beta=BETA,
                            gamma=GAMMA, N=N)
bench.solve_ode()

t_plot = np.linspace(0.0, T, 200)
a_vals = np.array([float(bench._a_func(t)) for t in t_plot])
b_vals = np.array([float(bench._b_func(t)) for t in t_plot])
c_vals = np.array([float(bench._c_func(t)) for t in t_plot])

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
fig.suptitle("Quadratic Benchmark — ODE Coefficients a(t), b(t), c(t)", fontsize=12)
for ax, vals, label in zip(axes, [a_vals, b_vals, c_vals], ["a(t)", "b(t)", "c(t)"]):
    ax.plot(t_plot, vals, lw=2)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("t")
plt.tight_layout()
fname = os.path.join(RESULTS_DIR, "exp4_ode_coefficients.png")
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname}")

# ── Section 2: PDE solver vs. analytical ─────────────────────────────────────
a_T, b_T, c_T = bench._terminal_coefficients()
term_func = lambda x: a_T * x ** 2 + b_T * x + c_T

# Build PDE solver on [x_min, x_max] with enough room
X_MIN, X_MAX, NX, NT = 50.0, 200.0, 200, 300
pde = NonlinearPDESolver(T=T, r=r, sigma=sigma, lam=LAM, varphi=VARPHI,
                          x_min=X_MIN, x_max=X_MAX, Nx=NX, Nt=NT,
                          terminal_func=term_func)
pde.solve()

# Compare on interior sub-grid (away from boundaries)
x_cmp = np.linspace(75.0, 160.0, 100)
t_cmp_vals = [0.0, 0.25, 0.5, 0.75]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("PDE Solver vs. Analytical Quadratic Benchmark", fontsize=12)

errors = {}
for ax, t_c in zip(axes.flat, t_cmp_vals):
    u_analytical = bench.u(t_c, x_cmp)
    u_numerical = pde.get_u(t_c, x_cmp)
    err = np.abs(u_numerical - u_analytical)
    errors[f"t={t_c}"] = {
        'sup_norm': float(np.max(err)),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
    }
    ax.plot(x_cmp, u_analytical, 'b-', lw=2, label="Analytical")
    ax.plot(x_cmp, u_numerical, 'r--', lw=1.5, label="PDE numerical")
    ax.set_title(f"t = {t_c:.2f}  sup-err = {np.max(err):.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(t,x)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fname = os.path.join(RESULTS_DIR, "exp4_pde_vs_analytical.png")
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname}")

# ── Section 3: X_0^* vs lambda and N ─────────────────────────────────────────
lam_values = np.linspace(0.0, 0.20, 20)
X0_star_lam = []
X0_frictionless_lam = []

for lam_i in lam_values:
    vph_i = max(lam_i / 2.0, VARPHI)  # ensure 2phi >= lambda
    b_i = QuadraticBenchmark(S0=S0, T=T, r=r, sigma=sigma, lam=lam_i,
                              varphi=vph_i, alpha=ALPHA, beta=BETA,
                              gamma=GAMMA, N=N)
    b_i.solve_ode()
    X0_star_lam.append(b_i.initial_capital())
    # frictionless: lam=0, varphi=0
    bf = QuadraticBenchmark(S0=S0, T=T, r=r, sigma=sigma, lam=0.0,
                             varphi=0.0, alpha=ALPHA, beta=BETA,
                             gamma=GAMMA, N=N)
    bf.solve_ode()
    X0_frictionless_lam.append(bf.initial_capital())

N_values = [0.5, 1.0, 2.0, 5.0]
X0_star_N = []
for N_i in N_values:
    b_i = QuadraticBenchmark(S0=S0, T=T, r=r, sigma=sigma, lam=LAM,
                              varphi=VARPHI, alpha=ALPHA, beta=BETA,
                              gamma=GAMMA, N=N_i)
    b_i.solve_ode()
    X0_star_N.append(b_i.initial_capital())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Initial Capital X_0^* for Quadratic Payoff", fontsize=12)

ax1.plot(lam_values, X0_star_lam, 'b-o', ms=4, label="With impact")
ax1.axhline(X0_frictionless_lam[0], color='k', ls='--', label="Frictionless (λ=0)")
ax1.set_xlabel("λ")
ax1.set_ylabel("X_0^*")
ax1.set_title("X_0^* vs λ  (φ = max(λ/2, 0.06))")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.bar([str(n) for n in N_values], X0_star_N, color='steelblue')
ax2.set_xlabel("N (number of contracts)")
ax2.set_ylabel("X_0^*")
ax2.set_title("X_0^* vs N  (λ=0.1, φ=0.06)")
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fname = os.path.join(RESULTS_DIR, "exp4_initial_capital.png")
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname}")

# ── Section 4: Midpoint regime (2*varphi = lambda) ───────────────────────────
bench_mid = QuadraticBenchmark(S0=S0, T=T, r=r, sigma=sigma,
                                lam=LAM, varphi=LAM / 2,
                                alpha=ALPHA, beta=BETA, gamma=GAMMA, N=N)
bench_mid.solve_ode()
X0_mid = bench_mid.initial_capital()

# Frictionless benchmark
bench_0 = QuadraticBenchmark(S0=S0, T=T, r=r, sigma=sigma,
                              lam=0.0, varphi=0.0,
                              alpha=ALPHA, beta=BETA, gamma=GAMMA, N=N)
bench_0.solve_ode()
X0_0 = bench_0.initial_capital()

# ── Summary ───────────────────────────────────────────────────────────────────
bench.solve_ode()   # re-solve baseline for reporting
X0_baseline = bench.initial_capital()

summary = {
    "params": {"S0": S0, "T": T, "r": r, "sigma": sigma,
               "lam": LAM, "varphi": VARPHI, "alpha": ALPHA,
               "beta": BETA, "gamma": GAMMA, "N": N},
    "terminal_coefficients": {"a_T": a_T, "b_T": b_T, "c_T": c_T},
    "pde_vs_analytical_errors": errors,
    "initial_capital": {
        "X0_star": X0_baseline,
        "X0_frictionless": X0_0,
        "X0_midpoint": X0_mid,
        "X0_impact_premium": X0_baseline - X0_0,
    },
    "X0_vs_N": dict(zip([str(n) for n in N_values], X0_star_N)),
}

summary_path = os.path.join(RESULTS_DIR, "exp4_summary.json")
with open(summary_path, "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"Summary written to {summary_path}")
print(json.dumps(summary, indent=2))
