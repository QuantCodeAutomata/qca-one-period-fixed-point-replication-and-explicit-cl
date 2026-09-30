"""
Experiment 3 — Continuous-Time Terminal Condition Modification

Demonstrates the modified terminal condition tilde_V for a call option under
price-impact constraints, following the paper's explicit piecewise construction.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.terminal import (
    modified_terminal_call,
    modified_terminal_call_hedge,
    modified_terminal_put,
    validate_implicit_relation,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Global parameters ────────────────────────────────────────────────────────
K = 100.0
LAM = 0.1
VARPHI = 0.06
N_VALUES = [1, 2]
X_GRID = np.linspace(80, 120, 2000)


# ── Helper ───────────────────────────────────────────────────────────────────
def call_payoff(x: np.ndarray, strike: float, n: float) -> np.ndarray:
    return n * np.maximum(x - strike, 0.0)


def put_payoff(x: np.ndarray, strike: float, n: float) -> np.ndarray:
    return n * np.maximum(strike - x, 0.0)


# ── Section 1: main comparison plots (N = 1 and N = 2) ───────────────────────
summary = {}

for N in N_VALUES:
    thr_low = K - 2 * VARPHI * N
    thr_high = K - LAM * N

    V_orig = call_payoff(X_GRID, K, N)
    V_shift = call_payoff(X_GRID, K - LAM * N, N)
    tilde_V = modified_terminal_call(X_GRID, K, N, LAM, VARPHI)
    tilde_Vp = modified_terminal_call_hedge(X_GRID, K, N, LAM, VARPHI)

    # Validate implicit relation
    V_func = lambda xi, _N=N: call_payoff(xi, K, _N)
    val = validate_implicit_relation(
        X_GRID, tilde_V, tilde_Vp, V_func, LAM, VARPHI
    )
    summary[f"N{N}_call"] = val

    # ── Figure: payoffs + hedge ──────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(
        f"Modified Terminal Condition — Call (N={N}, K={K}, λ={LAM}, φ={VARPHI})",
        fontsize=12,
    )

    # Upper panel: payoff comparison
    ax1.plot(X_GRID, V_orig, "k--", lw=1.4, label=r"Original $V(x)=N(x-K)^+$")
    ax1.plot(
        X_GRID,
        V_shift,
        color="tab:orange",
        lw=1.4,
        ls="--",
        label=r"Shifted benchmark $N(x-(K-\lambda N))^+$",
    )
    ax1.plot(
        X_GRID,
        tilde_V,
        color="tab:blue",
        lw=2,
        label=r"Modified $\tilde{V}(x)$",
    )
    ax1.axvline(thr_low, color="gray", lw=0.9, ls=":", label=f"$K-2φN={thr_low:.2f}$")
    ax1.axvline(thr_high, color="dimgray", lw=0.9, ls=":", label=f"$K-λN={thr_high:.2f}$")
    ax1.set_ylabel("Value")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("Payoff comparison", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Lower panel: terminal hedge
    ax2.plot(
        X_GRID,
        tilde_Vp,
        color="tab:red",
        lw=2,
        label=r"Terminal hedge $\tilde{V}'(x)$",
    )
    ax2.axhline(N, color="tab:green", lw=1, ls="--", label=f"$N={N}$ (max)")
    ax2.axvline(thr_low, color="gray", lw=0.9, ls=":")
    ax2.axvline(thr_high, color="dimgray", lw=0.9, ls=":")
    ax2.set_xlabel("Underlying price $x$")
    ax2.set_ylabel("Hedge ratio")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_title("Terminal hedge (derivative)", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"exp3_call_N{N}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[N={N}] Saved {fname}")
    print(
        f"[N={N}] Implicit-relation validation — "
        f"max_residual={val['max_residual']:.2e}, satisfied={val['satisfied']}"
    )


# ── Section 2: midpoint regime (2*varphi = lambda → smoothing collapses) ────
VARPHI_MID = LAM / 2.0  # 2*varphi = lambda
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"Midpoint Regime: 2φ = λ (φ={VARPHI_MID}, λ={LAM}) — Smoothing Region Collapses",
    fontsize=11,
)

for idx, N in enumerate(N_VALUES):
    ax = axes[idx]
    thr_low_mid = K - 2 * VARPHI_MID * N   # = K - lambda*N = thr_high_mid
    thr_high_mid = K - LAM * N              # same point

    V_orig = call_payoff(X_GRID, K, N)
    tilde_V_mid = modified_terminal_call(X_GRID, K, N, LAM, VARPHI_MID)
    tilde_V_nom = modified_terminal_call(X_GRID, K, N, LAM, VARPHI)

    ax.plot(X_GRID, V_orig, "k--", lw=1.2, label=r"Original $V(x)$")
    ax.plot(
        X_GRID,
        tilde_V_nom,
        color="tab:blue",
        lw=1.8,
        label=f"Normal φ={VARPHI} (smooth region width={(2*VARPHI-LAM)*N:.2f})",
    )
    ax.plot(
        X_GRID,
        tilde_V_mid,
        color="tab:red",
        lw=1.8,
        ls="--",
        label=f"Midpoint φ={VARPHI_MID} (collapsed)",
    )
    ax.axvline(thr_high_mid, color="gray", lw=0.9, ls=":", label=f"$K-λN={thr_high_mid:.2f}$")
    ax.set_xlabel("Underlying price $x$")
    ax.set_ylabel("Value")
    ax.set_title(f"N={N}", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(K - 3 * N - 1, K + 2 * N + 1)

    val_mid = validate_implicit_relation(
        X_GRID,
        tilde_V_mid,
        modified_terminal_call_hedge(X_GRID, K, N, LAM, VARPHI_MID),
        lambda xi, _N=N: call_payoff(xi, K, _N),
        LAM,
        VARPHI_MID,
    )
    summary[f"N{N}_midpoint"] = val_mid
    print(
        f"[Midpoint N={N}] max_residual={val_mid['max_residual']:.2e}, "
        f"satisfied={val_mid['satisfied']}"
    )

plt.tight_layout()
fname_mid = os.path.join(RESULTS_DIR, "exp3_midpoint_regime.png")
fig.savefig(fname_mid, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname_mid}")


# ── Section 3: put payoff ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"Modified Terminal Condition — Put (K={K}, λ={LAM}, φ={VARPHI})",
    fontsize=12,
)

for idx, N in enumerate(N_VALUES):
    ax = [ax1, ax2][idx]
    thr_low_put = K + LAM * N
    thr_high_put = K + 2 * VARPHI * N

    V_put = put_payoff(X_GRID, K, N)
    tilde_V_put = modified_terminal_put(X_GRID, K, N, LAM, VARPHI)

    ax.plot(X_GRID, V_put, "k--", lw=1.4, label=r"Original put $V(x)=N(K-x)^+$")
    ax.plot(
        X_GRID,
        tilde_V_put,
        color="tab:purple",
        lw=2,
        label=r"Modified put $\tilde{V}(x)$",
    )
    ax.axvline(thr_low_put, color="gray", lw=0.9, ls=":", label=f"$K+λN={thr_low_put:.2f}$")
    ax.axvline(thr_high_put, color="dimgray", lw=0.9, ls=":", label=f"$K+2φN={thr_high_put:.2f}$")
    ax.set_xlabel("Underlying price $x$")
    ax.set_ylabel("Value")
    ax.set_title(f"N={N}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(K - 2 * N - 1, K + 3 * N + 1)

    # validate put implicit relation
    two_phi_minus_lam = 2 * VARPHI - LAM
    # For put: tilde_V_put'(x) = derivative
    # Low mask: <= K+lam*N → N*(K-x) + ..., derivative = -N
    # Mid mask: (K+lam*N, K+2*phi*N] → -(K+2phi*N-x)/(2phi-lam)
    # High mask: > K+2*phi*N → 0
    tilde_Vp_put = np.zeros_like(X_GRID)
    low_m = X_GRID <= thr_low_put
    mid_m = (X_GRID > thr_low_put) & (X_GRID <= thr_high_put)
    tilde_Vp_put[low_m] = -N
    if two_phi_minus_lam > 1e-14:
        tilde_Vp_put[mid_m] = -(K + 2 * VARPHI * N - X_GRID[mid_m]) / two_phi_minus_lam

    V_put_func = lambda xi, _N=N: put_payoff(xi, K, _N)
    val_put = validate_implicit_relation(
        X_GRID, tilde_V_put, tilde_Vp_put, V_put_func, LAM, VARPHI
    )
    summary[f"N{N}_put"] = val_put
    print(
        f"[Put N={N}] max_residual={val_put['max_residual']:.2e}, "
        f"satisfied={val_put['satisfied']}"
    )

plt.tight_layout()
fname_put = os.path.join(RESULTS_DIR, "exp3_put.png")
fig.savefig(fname_put, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname_put}")


# ── Section 4: summary overview figure ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Modified Terminal Condition — Overview", fontsize=13)

titles = [
    (f"Call N=1 (φ={VARPHI}, λ={LAM})", 1, VARPHI, LAM, "call"),
    (f"Call N=2 (φ={VARPHI}, λ={LAM})", 2, VARPHI, LAM, "call"),
    (f"Call N=1 midpoint (φ={VARPHI_MID}=λ/2)", 1, VARPHI_MID, LAM, "call"),
    (f"Put  N=1 (φ={VARPHI}, λ={LAM})", 1, VARPHI, LAM, "put"),
]

for ax, (title, N, varphi, lam, ptype) in zip(axes.flat, titles):
    if ptype == "call":
        V_base = call_payoff(X_GRID, K, N)
        tV = modified_terminal_call(X_GRID, K, N, lam, varphi)
        ax.plot(X_GRID, V_base, "k--", lw=1.2, label="Original")
        ax.plot(X_GRID, tV, color="tab:blue", lw=2, label=r"$\tilde{V}$")
        ax.axvline(K - 2 * varphi * N, color="gray", lw=0.8, ls=":")
        ax.axvline(K - lam * N, color="dimgray", lw=0.8, ls=":")
    else:
        V_base = put_payoff(X_GRID, K, N)
        tV = modified_terminal_put(X_GRID, K, N, lam, varphi)
        ax.plot(X_GRID, V_base, "k--", lw=1.2, label="Original put")
        ax.plot(X_GRID, tV, color="tab:purple", lw=2, label=r"$\tilde{V}$ put")
        ax.axvline(K + lam * N, color="gray", lw=0.8, ls=":")
        ax.axvline(K + 2 * varphi * N, color="dimgray", lw=0.8, ls=":")

    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("Value")
    xl = K - 3 * N - 0.5 if ptype == "call" else K - 2 * N - 0.5
    xr = K + 2 * N + 0.5 if ptype == "call" else K + 3 * N + 0.5
    ax.set_xlim(xl, xr)

plt.tight_layout()
fname_overview = os.path.join(RESULTS_DIR, "exp3_overview.png")
fig.savefig(fname_overview, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname_overview}")


# ── Save JSON summary ────────────────────────────────────────────────────────
summary_path = os.path.join(RESULTS_DIR, "exp3_summary.json")
with open(summary_path, "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"Summary written to {summary_path}")
print(json.dumps(summary, indent=2))
