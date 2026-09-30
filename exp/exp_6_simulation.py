"""
Experiment 6 — Pathwise Replication: Convergence as n_steps Increases

Simulates discrete-time hedging under linear price impact using the PDE hedge
field u_x(t, S_t).  Shows that residuals X_T - V(P_T) converge to zero as the
rebalancing frequency increases.
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
from src.terminal import modified_terminal_call
from src.simulation import PathwiseReplicator

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.20
LAM = 0.10
VARPHI = 0.06
N = 1.0
N_PATHS = 1000
STEPS_LIST = [100, 1000, 10000]

# ── Solve PDE once ────────────────────────────────────────────────────────────
print("Solving nonlinear PDE for call option ...")
terminal_func = lambda x: modified_terminal_call(np.asarray(x), K, N, LAM, VARPHI)
pde = NonlinearPDESolver(T=T, r=r, sigma=sigma, lam=LAM, varphi=VARPHI,
                          x_min=40.0, x_max=220.0, Nx=200, Nt=300,
                          terminal_func=terminal_func)
pde.solve()

V_func = lambda p: N * np.maximum(np.asarray(p, dtype=float) - K, 0.0)
replicator = PathwiseReplicator(
    S0=S0, T=T, r=r, sigma=sigma,
    lam=LAM, varphi=VARPHI,
    K=K, N=N,
    pde_u_func=pde.get_u,
    pde_ux_func=pde.get_ux,
    V_func=V_func,
)

X0_star = pde.initial_capital(S0)
bs_price = PathwiseReplicator.bs_price(S0, K, T, r, sigma)
print(f"X_0^* = {X0_star:.6f},  BS price = {bs_price:.6f}")

# ── Run simulation for each n_steps ──────────────────────────────────────────
convergence = {}

for n_steps in STEPS_LIST:
    print(f"Simulating {N_PATHS} paths, n_steps={n_steps} ...")
    res = replicator.run(n_steps=n_steps, n_paths=N_PATHS, seed=42)
    residuals = res['residuals']
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mean_res = float(np.mean(residuals))
    max_abs = float(np.max(np.abs(residuals)))
    q01 = float(np.quantile(residuals, 0.01))
    q99 = float(np.quantile(residuals, 0.99))
    convergence[n_steps] = {
        'mean': mean_res,
        'mae': mae,
        'rmse': rmse,
        'max_abs': max_abs,
        'q01': q01,
        'q99': q99,
    }
    print(f"  n={n_steps}: mean={mean_res:.5f}, MAE={mae:.5f}, "
          f"RMSE={rmse:.5f}, max|e|={max_abs:.5f}")

    # ── Scatter plot: (P_T, X_T) with payoff curve ───────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Replication Quality: n_steps={n_steps}", fontsize=12)

    P_T = res['P_T']
    X_T = res['X_T']
    payoff = res['payoff']

    p_line = np.linspace(max(60.0, P_T.min() - 2), min(200.0, P_T.max() + 2), 300)
    payoff_line = N * np.maximum(p_line - K, 0.0)

    ax1.scatter(P_T, X_T, s=4, alpha=0.4, c='steelblue', label="Portfolio X_T")
    ax1.plot(p_line, payoff_line, 'r-', lw=2, label="Payoff V(P_T)")
    ax1.set_xlabel("P_T (impacted terminal price)")
    ax1.set_ylabel("X_T (portfolio value)")
    ax1.set_title("(P_T, X_T) scatter")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.hist(res['residuals'], bins=50, color='steelblue', edgecolor='white',
             density=True, alpha=0.8)
    ax2.axvline(0.0, color='r', lw=1.5, ls='--')
    ax2.set_xlabel("Residual X_T - V(P_T)")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Residual distribution  RMSE={rmse:.5f}")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"exp6_scatter_n{n_steps}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

# ── Convergence plot ──────────────────────────────────────────────────────────
steps_arr = np.array(STEPS_LIST)
rmse_arr = np.array([convergence[n]['rmse'] for n in STEPS_LIST])
mae_arr = np.array([convergence[n]['mae'] for n in STEPS_LIST])

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(steps_arr, rmse_arr, 'b-o', ms=7, lw=2, label="RMSE")
ax.loglog(steps_arr, mae_arr, 'r-s', ms=7, lw=2, label="MAE")
# Reference line: O(1/sqrt(n))
ref = rmse_arr[0] * np.sqrt(steps_arr[0] / steps_arr)
ax.loglog(steps_arr, ref, 'k--', lw=1, label=r"$O(1/\sqrt{n})$ reference")
ax.set_xlabel("Number of rebalancing steps")
ax.set_ylabel("Replication error")
ax.set_title("Convergence of Replication Error vs. Number of Steps")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
fname = os.path.join(RESULTS_DIR, "exp6_convergence.png")
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname}")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = {
    "params": {"S0": S0, "K": K, "T": T, "r": r, "sigma": sigma,
               "lam": LAM, "varphi": VARPHI, "N": N, "n_paths": N_PATHS},
    "X0_star": float(X0_star),
    "bs_price": float(bs_price),
    "convergence": {str(k): v for k, v in convergence.items()},
}

summary_path = os.path.join(RESULTS_DIR, "exp6_summary.json")
with open(summary_path, "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"Summary written to {summary_path}")
print(json.dumps(summary, indent=2))
