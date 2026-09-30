"""
Experiment 7 — ITM Event Analysis

Under permanent price impact, the impacted terminal price P_T = S_T + λ*Δ_T
can exceed the strike K even when the fundamental price S_T < K.  This
experiment identifies such paths, measures their frequency, and plots
representative time series of S_t, P_t^Δ, and Δ_t.
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
N_STEPS = 1000

# ── Solve PDE once ────────────────────────────────────────────────────────────
print("Solving nonlinear PDE ...")
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

# ── Run simulation ────────────────────────────────────────────────────────────
print(f"Simulating {N_PATHS} paths, n_steps={N_STEPS} ...")
res = replicator.run(n_steps=N_STEPS, n_paths=N_PATHS, seed=42)

S_T = res['S_T']
P_T = res['P_T']
delta_T = res['delta_T']
paths = res['paths']     # (n_paths, n_steps+1)

# ── Identify ITM event paths: S_T < K < P_T ───────────────────────────────────
itm_event_mask = (S_T < K) & (P_T > K)
n_events = int(np.sum(itm_event_mask))
event_freq = n_events / N_PATHS
print(f"ITM event: S_T < K < P_T")
print(f"  Count: {n_events} / {N_PATHS}  (frequency = {event_freq:.4f})")

# Also compute the theoretical expected frequency (gap region K - lam*N < S_T < K)
K_eff = K - LAM * N
# P(K_eff < S_T < K) under GBM: approx (log(K) - log(K_eff)) / (sigma * sqrt(T))
from scipy.stats import norm as _norm
d_K = (np.log(K / S0) - (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
d_K_eff = (np.log(K_eff / S0) - (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
theoretical_freq = _norm.cdf(d_K) - _norm.cdf(d_K_eff)
print(f"  Theoretical (GBM approx): {theoretical_freq:.4f}")

# ── Select representative paths ───────────────────────────────────────────────
event_indices = np.where(itm_event_mask)[0]
if len(event_indices) == 0:
    # If no ITM events found, choose paths near the boundary
    near_K = np.abs(S_T - K)
    event_indices = np.argsort(near_K)[:3]
    print("No exact ITM events found; using paths nearest K")

# Choose up to 3 representative paths
n_rep = min(3, len(event_indices))
rep_indices = event_indices[:n_rep]

# ── Reconstruct hedge path for representative paths ───────────────────────────
t_grid = np.linspace(0.0, T, N_STEPS + 1)

def reconstruct_hedge_path(path_idx):
    """Re-run a single path to get the full Delta_t series."""
    S_path = paths[path_idx]
    deltas = np.zeros(N_STEPS + 1)
    deltas[0] = float(pde.get_ux(0.0, np.array([S_path[0]]))[0])
    for i in range(1, N_STEPS):
        t = t_grid[i]
        deltas[i] = float(pde.get_ux(t, np.array([S_path[i]]))[0])
    deltas[N_STEPS] = deltas[N_STEPS - 1]  # terminal
    return deltas

def bs_delta_path(S_path):
    """Black-Scholes delta along a path."""
    deltas = np.zeros(N_STEPS + 1)
    for i in range(N_STEPS):
        t = t_grid[i]
        tau = max(T - t, 1e-10)
        S = S_path[i]
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        deltas[i] = _norm.cdf(d1)
    deltas[N_STEPS] = deltas[N_STEPS - 1]
    return deltas

# ── Plotting representative paths ─────────────────────────────────────────────
# Sub-sample for plotting speed (show every k-th step)
k_sub = max(1, N_STEPS // 500)
t_sub = t_grid[::k_sub]

colors_rep = ['tab:blue', 'tab:orange', 'tab:green']

fig, axes = plt.subplots(n_rep, 2, figsize=(14, 4 * n_rep))
if n_rep == 1:
    axes = axes[np.newaxis, :]  # ensure 2-D indexing

fig.suptitle(
    f"Representative ITM-Event Paths  (S_T < K < P_T, λ={LAM}, φ={VARPHI})",
    fontsize=12
)

for row, (pidx, col) in enumerate(zip(rep_indices, colors_rep)):
    S_path = paths[pidx]
    delta_path = reconstruct_hedge_path(pidx)
    bs_d_path = bs_delta_path(S_path)
    P_path = S_path + LAM * delta_path

    S_sub = S_path[::k_sub]
    P_sub = P_path[::k_sub]
    delta_sub = delta_path[::k_sub]
    bs_sub = bs_d_path[::k_sub]

    ax_price = axes[row, 0]
    ax_delta = axes[row, 1]

    ax_price.plot(t_sub, S_sub, color=col, lw=1.5, label=f"S_t (path {pidx})")
    ax_price.plot(t_sub, P_sub, color=col, lw=1.5, ls='--', label=f"P_t = S_t + λΔ_t")
    ax_price.axhline(K, color='k', lw=1.0, ls=':', label=f"Strike K={K}")
    ax_price.set_xlabel("t")
    ax_price.set_ylabel("Price")
    ax_price.set_title(
        f"Path {pidx}: S_T={S_T[pidx]:.2f}, P_T={P_T[pidx]:.2f}"
        f"  [ITM: {itm_event_mask[pidx]}]"
    )
    ax_price.legend(fontsize=8)
    ax_price.grid(True, alpha=0.3)

    ax_delta.plot(t_sub, delta_sub, color=col, lw=1.5, label="PDE Δ_t")
    ax_delta.plot(t_sub, bs_sub, 'k--', lw=1.0, label="BS Δ_t")
    ax_delta.set_xlabel("t")
    ax_delta.set_ylabel("Delta")
    ax_delta.set_title(f"Hedge comparison (path {pidx})")
    ax_delta.legend(fontsize=8)
    ax_delta.grid(True, alpha=0.3)

plt.tight_layout()
fname = os.path.join(RESULTS_DIR, "exp7_representative_path.png")
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname}")

# ── Overview scatter: S_T vs P_T with event region highlighted ────────────────
fig, ax = plt.subplots(figsize=(8, 6))
non_event = ~itm_event_mask
ax.scatter(S_T[non_event], P_T[non_event], s=3, alpha=0.3, c='steelblue',
           label="Normal paths")
if itm_event_mask.any():
    ax.scatter(S_T[itm_event_mask], P_T[itm_event_mask], s=12, c='red',
               zorder=5, label=f"ITM event (n={n_events})")
ax.axhline(K, color='k', lw=1.0, ls='--', label=f"K={K}")
ax.axvline(K, color='k', lw=1.0, ls='--')
ax.axvline(K - LAM * N, color='gray', lw=0.8, ls=':', label=f"K-λN={K-LAM*N:.1f}")
ax.plot([60, 160], [60, 160], 'g-', lw=0.8, label="P_T = S_T (no impact)")
ax.set_xlabel("S_T (fundamental)")
ax.set_ylabel("P_T (impacted)")
ax.set_title(f"Terminal Price Scatter: ITM events highlighted\n"
             f"Freq={event_freq:.4f}, theory≈{theoretical_freq:.4f}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fname = os.path.join(RESULTS_DIR, "exp7_scatter.png")
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname}")

# ── Summary ───────────────────────────────────────────────────────────────────
summary = {
    "params": {"S0": S0, "K": K, "T": T, "r": r, "sigma": sigma,
               "lam": LAM, "varphi": VARPHI, "N": N,
               "n_paths": N_PATHS, "n_steps": N_STEPS},
    "itm_event": {
        "count": n_events,
        "frequency": event_freq,
        "theoretical_approx": theoretical_freq,
        "K_eff": K_eff,
        "gap_width": float(K - K_eff),
    },
    "representative_paths": {
        str(int(pidx)): {
            "S_T": float(S_T[pidx]),
            "P_T": float(P_T[pidx]),
            "delta_T": float(delta_T[pidx]),
            "itm_event": bool(itm_event_mask[pidx]),
        }
        for pidx in rep_indices
    },
}

freq_path = os.path.join(RESULTS_DIR, "exp7_event_frequency.json")
with open(freq_path, "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"Summary written to {freq_path}")
print(json.dumps(summary, indent=2))
