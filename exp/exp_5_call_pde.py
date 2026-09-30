"""
Experiment 5 — Call Option Pricing via Nonlinear PDE

Solves the nonlinear PDE with modified terminal condition for a European call
under linear permanent price impact.  Compares prices and deltas to the
Black-Scholes benchmark, and demonstrates the normalization/scaling properties.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pde import NonlinearPDESolver
from src.terminal import modified_terminal_call, modified_terminal_call_hedge
from src.simulation import PathwiseReplicator

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Shared parameters ─────────────────────────────────────────────────────────
K = 100.0
T = 1.0
r = 0.05
sigma = 0.20
VARPHI = 0.06

S_PLOT = np.linspace(60.0, 140.0, 200)
X_MIN, X_MAX, NX, NT = 40.0, 220.0, 200, 300

CASES = [
    {"N": 1.0, "lam": 0.01, "label": "N=1, λ=0.01"},
    {"N": 1.0, "lam": 0.10, "label": "N=1, λ=0.10"},
    {"N": 10.0, "lam": 0.01, "label": "N=10, λ=0.01"},
    {"N": 10.0, "lam": 0.10, "label": "N=10, λ=0.10"},
]


def solve_call_pde(N: float, lam: float, varphi: float) -> NonlinearPDESolver:
    """Build and solve nonlinear PDE for call with given N, lam, varphi."""
    terminal = lambda x: modified_terminal_call(np.asarray(x), K, N, lam, varphi)
    pde = NonlinearPDESolver(T=T, r=r, sigma=sigma, lam=lam, varphi=varphi,
                              x_min=X_MIN, x_max=X_MAX, Nx=NX, Nt=NT,
                              terminal_func=terminal)
    pde.solve()
    return pde


# ── BS reference ──────────────────────────────────────────────────────────────
bs_prices = np.array([PathwiseReplicator.bs_price(s, K, T, r, sigma)
                      for s in S_PLOT])

# Compute BS delta on S_PLOT
from scipy.stats import norm as _norm
def bs_delta_curve(S_arr):
    d1 = (np.log(S_arr / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return _norm.cdf(d1)

bs_deltas = bs_delta_curve(S_PLOT)

# ── Run experiments ───────────────────────────────────────────────────────────
results = {}
pde_solutions = {}

for case in CASES:
    N, lam, label = case["N"], case["lam"], case["label"]
    print(f"Solving PDE: {label} ...")
    assert 2 * VARPHI >= lam, f"Admissibility violated for {label}"
    pde = solve_call_pde(N, lam, VARPHI)
    pde_solutions[label] = pde

    u0 = pde.get_u(0.0, S_PLOT)
    ux0 = pde.get_ux(0.0, S_PLOT)

    price_per_contract = u0 / N
    delta_per_contract = ux0 / N
    X0_star = pde.initial_capital(100.0)

    results[label] = {
        "N": N, "lam": lam, "varphi": VARPHI,
        "X0_star": float(X0_star),
        "X0_star_per_N": float(X0_star / N),
        "u0_at_S0": float(pde.get_u(0.0, np.array([100.0]))[0]),
        "ux0_at_S0": float(pde.get_ux(0.0, np.array([100.0]))[0]),
        "price_above_bs_at_S0": float(pde.get_u(0.0, np.array([100.0]))[0] / N
                                      - PathwiseReplicator.bs_price(100.0, K, T, r, sigma)),
    }

# ── Figure 1: Price per contract vs S0 ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
fig.suptitle("Call PDE Price per Contract vs Black-Scholes", fontsize=12)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

for ax, case, color in zip(axes.flat, CASES, colors):
    label = case["label"]
    N = case["N"]
    pde = pde_solutions[label]
    price_per_contract = pde.get_u(0.0, S_PLOT) / N
    ax.plot(S_PLOT, bs_prices, 'k--', lw=1.5, label="Black-Scholes")
    ax.plot(S_PLOT, price_per_contract, color=color, lw=2, label=f"PDE ({label})")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("S₀")
    ax.set_ylabel("Price / N")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(K, color='gray', lw=0.8, ls=':')

plt.tight_layout()
fname = os.path.join(RESULTS_DIR, "exp5_price_per_contract.png")
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname}")

# ── Figure 2: Delta per contract vs S0 ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
fig.suptitle("Call PDE Delta per Contract vs Black-Scholes Delta", fontsize=12)

for ax, case, color in zip(axes.flat, CASES, colors):
    label = case["label"]
    N = case["N"]
    pde = pde_solutions[label]
    delta_per_contract = pde.get_ux(0.0, S_PLOT) / N
    ax.plot(S_PLOT, bs_deltas, 'k--', lw=1.5, label="BS Δ")
    ax.plot(S_PLOT, delta_per_contract, color=color, lw=2, label=f"PDE Δ ({label})")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("S₀")
    ax.set_ylabel("Δ / N")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axvline(K, color='gray', lw=0.8, ls=':')
    ax.axhline(0.0, color='gray', lw=0.5)
    ax.axhline(1.0, color='gray', lw=0.5)

plt.tight_layout()
fname = os.path.join(RESULTS_DIR, "exp5_delta_per_contract.png")
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname}")

# ── Figure 3: Normalization — same N*lambda => same normalized curve ───────────
# Test midpoint regime (2*varphi = lambda)
# Case A: N=1, lam=0.10, varphi=0.05  => N*lam = 0.10
# Case B: N=2, lam=0.05, varphi=0.025 => N*lam = 0.10
print("Solving normalization comparison PDEs ...")
lam_A, N_A, vph_A = 0.10, 1.0, 0.05
lam_B, N_B, vph_B = 0.05, 2.0, 0.025

pde_A = solve_call_pde(N_A, lam_A, vph_A)
pde_B = solve_call_pde(N_B, lam_B, vph_B)

u_A_norm = pde_A.get_u(0.0, S_PLOT) / N_A
u_B_norm = pde_B.get_u(0.0, S_PLOT) / N_B
ux_A_norm = pde_A.get_ux(0.0, S_PLOT) / N_A
ux_B_norm = pde_B.get_ux(0.0, S_PLOT) / N_B

norm_price_diff = float(np.max(np.abs(u_A_norm - u_B_norm)))
norm_delta_diff = float(np.max(np.abs(ux_A_norm - ux_B_norm)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Midpoint Regime Normalization: N*λ=0.10 for both cases", fontsize=11)
ax1.plot(S_PLOT, u_A_norm, 'b-', lw=2, label=f"N={N_A}, λ={lam_A}, φ={vph_A}")
ax1.plot(S_PLOT, u_B_norm, 'r--', lw=2, label=f"N={N_B}, λ={lam_B}, φ={vph_B}")
ax1.plot(S_PLOT, bs_prices, 'k:', lw=1.5, label="BS")
ax1.set_xlabel("S₀")
ax1.set_ylabel("u(0,S₀)/N")
ax1.set_title(f"Price/N  (max diff = {norm_price_diff:.4f})")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.plot(S_PLOT, ux_A_norm, 'b-', lw=2, label=f"N={N_A}, λ={lam_A}")
ax2.plot(S_PLOT, ux_B_norm, 'r--', lw=2, label=f"N={N_B}, λ={lam_B}")
ax2.plot(S_PLOT, bs_deltas, 'k:', lw=1.5, label="BS Δ")
ax2.set_xlabel("S₀")
ax2.set_ylabel("u_x(0,S₀)/N")
ax2.set_title(f"Delta/N  (max diff = {norm_delta_diff:.4f})")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fname = os.path.join(RESULTS_DIR, "exp5_normalization_midpoint.png")
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname}")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = {
    "params": {"K": K, "T": T, "r": r, "sigma": sigma, "varphi": VARPHI},
    "cases": results,
    "normalization_check": {
        "midpoint_price_max_diff": norm_price_diff,
        "midpoint_delta_max_diff": norm_delta_diff,
        "cases": [
            {"N": N_A, "lam": lam_A, "varphi": vph_A, "N_lam": N_A * lam_A},
            {"N": N_B, "lam": lam_B, "varphi": vph_B, "N_lam": N_B * lam_B},
        ],
    },
    "bs_benchmark_at_S0": PathwiseReplicator.bs_price(100.0, K, T, r, sigma),
}

summary_path = os.path.join(RESULTS_DIR, "exp5_summary.json")
with open(summary_path, "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"Summary written to {summary_path}")
print(json.dumps(summary, indent=2))
