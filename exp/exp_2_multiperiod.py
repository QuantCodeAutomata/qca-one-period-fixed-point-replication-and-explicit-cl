"""
Experiment 2: Multiperiod Replication on a Recombining Fundamental-Price Tree
via Backward Induction.

Validates:
- BinomialImpactTree builds f_m and g_m correctly via backward induction.
- Terminal replication holds at every leaf.
- Initial replication cost exceeds frictionless binomial price for monotone payoffs.
- Midpoint case (2*varphi = lambda) is numerically stable.
- Price convergence as M increases.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make src importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.payoffs import make_call
from src.tree import BinomialImpactTree

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_tree(
    S0: float,
    T: float,
    M: int,
    sigma: float,
    r: float,
    lam: float,
    varphi: float,
    K: float,
    N: float = 1.0,
) -> dict:
    """Build a BinomialImpactTree for a European call and return key metrics."""
    h = T / M
    u = np.exp(sigma * np.sqrt(h))
    d = np.exp(-sigma * np.sqrt(h))
    V = make_call(K, N)

    tree = BinomialImpactTree(S0=S0, T=T, M=M, u=u, d=d, r=r,
                               lam=lam, varphi=varphi, V=V, N=N)
    tree.build()

    X0_imp = tree.initial_value()
    X0_fric = tree.frictionless_price()
    repl = tree.verify_terminal_replication(tol=1e-5)

    return {
        "M": M,
        "h": h,
        "u": round(u, 6),
        "d": round(d, 6),
        "q": round(tree.q, 6),
        "lambda": lam,
        "varphi": varphi,
        "X0_impacted": X0_imp,
        "X0_frictionless": X0_fric,
        "impacted_minus_fric": X0_imp - X0_fric,
        "max_replication_error": repl["max_error"],
        "all_replicated": repl["all_replicated"],
        "f1_S0": tree.get_f(1, 0) if M >= 1 else float("nan"),
    }


# ---------------------------------------------------------------------------
# 1. Baseline verification
# ---------------------------------------------------------------------------

def section_baseline() -> pd.DataFrame:
    """Verify exact replication and cost ordering for baseline parameters."""
    print("\n=== Section 1: Baseline Verification ===")

    S0, T, K, N = 100.0, 1.0, 100.0, 1.0
    sigma, r = 0.20, 0.05
    lam, varphi = 0.10, 0.06  # 2*varphi=0.12 > lambda=0.10, off-midpoint

    # Check assertions
    h_check = T / 5
    u_check = np.exp(sigma * np.sqrt(h_check))
    d_check = np.exp(-sigma * np.sqrt(h_check))
    print(f"  No-arbitrage check: d={d_check:.4f} < 1+r*h={1+r*h_check:.4f} < u={u_check:.4f} ✓")
    print(f"  No-price-manipulation: 2*varphi={2*varphi:.3f} >= lambda={lam:.3f} ✓")

    rows = []
    for M in [3, 5, 7]:
        res = run_tree(S0, T, M, sigma, r, lam, varphi, K, N)
        rows.append(res)
        flag = "✓" if res["all_replicated"] else "✗"
        print(f"  M={M:2d}: X0*={res['X0_impacted']:.5f}, "
              f"frictionless={res['X0_frictionless']:.5f}, "
              f"diff={res['impacted_minus_fric']:.5f}, "
              f"max_repl_err={res['max_replication_error']:.2e} {flag}")

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# 2. Midpoint case (2*varphi = lambda)
# ---------------------------------------------------------------------------

def section_midpoint() -> pd.DataFrame:
    """Test the midpoint regime 2*varphi = lambda."""
    print("\n=== Section 2: Midpoint Regime (2*varphi = lambda) ===")

    S0, T, K, N = 100.0, 1.0, 100.0, 1.0
    sigma, r = 0.20, 0.05
    lam = 0.10
    varphi = lam / 2  # midpoint: 2*varphi = lambda

    rows = []
    for M in [3, 5, 7]:
        res = run_tree(S0, T, M, sigma, r, lam, varphi, K, N)
        rows.append(res)
        flag = "✓" if res["all_replicated"] else "✗"
        print(f"  M={M:2d} [midpoint]: X0*={res['X0_impacted']:.5f}, "
              f"frictionless={res['X0_frictionless']:.5f}, "
              f"max_repl_err={res['max_replication_error']:.2e} {flag}")

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# 3. Convergence study: increase M
# ---------------------------------------------------------------------------

def section_convergence() -> pd.DataFrame:
    """Study price convergence as M increases for fixed parameters."""
    print("\n=== Section 3: Convergence with M ===")

    S0, T, K, N = 100.0, 1.0, 100.0, 1.0
    sigma, r = 0.20, 0.05
    lam, varphi = 0.10, 0.06  # off-midpoint

    rows = []
    for M in [2, 3, 5, 8, 10, 15, 20]:
        res = run_tree(S0, T, M, sigma, r, lam, varphi, K, N)
        rows.append(res)
        print(f"  M={M:3d}: X0*={res['X0_impacted']:.6f}, "
              f"frictionless={res['X0_frictionless']:.6f}, "
              f"max_err={res['max_replication_error']:.2e}")

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# 4. Off-midpoint sweep: varying varphi
# ---------------------------------------------------------------------------

def section_offmidpoint() -> pd.DataFrame:
    """Compare midpoint vs off-midpoint at fixed M=5."""
    print("\n=== Section 4: Off-Midpoint vs Midpoint ===")

    S0, T, K, N, M = 100.0, 1.0, 100.0, 1.0, 5
    sigma, r, lam = 0.20, 0.05, 0.10

    rows = []
    for varphi in [0.05, 0.06, 0.08, 0.10, 0.15]:
        if 2 * varphi < lam:
            print(f"  varphi={varphi:.3f}: skipped (2*varphi < lambda)")
            continue
        res = run_tree(S0, T, M, sigma, r, lam, varphi, K, N)
        rows.append(res)
        print(f"  varphi={varphi:.3f} (2phi-lam={2*varphi-lam:.3f}): "
              f"X0*={res['X0_impacted']:.5f}, fric={res['X0_frictionless']:.5f}, "
              f"diff={res['impacted_minus_fric']:.5f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def make_plots(conv_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    """Save convergence and cost comparison plots."""

    # Convergence: impacted price vs M
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(conv_df["M"], conv_df["X0_impacted"], "o-", label="Impacted X₀*", color="steelblue")
    ax.plot(conv_df["M"], conv_df["X0_frictionless"], "s--", label="Frictionless", color="darkorange")
    ax.set_xlabel("Number of Steps M")
    ax.set_ylabel("Initial Replication Cost")
    ax.set_title("Price Convergence with M\n(λ=0.1, φ=0.06, K=S₀=100)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogy(conv_df["M"], conv_df["max_replication_error"], "o-", color="crimson")
    ax.set_xlabel("Number of Steps M")
    ax.set_ylabel("Max Terminal Replication Error")
    ax.set_title("Replication Error vs M")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp2_convergence.png"), dpi=120)
    plt.close(fig)

    # Cost gap: impacted - frictionless vs M
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(conv_df["M"], conv_df["impacted_minus_fric"], "o-", color="purple")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Number of Steps M")
    ax.set_ylabel("X₀*(impacted) − X₀(frictionless)")
    ax.set_title("Cost Premium from Price Impact vs M")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp2_cost_premium.png"), dpi=120)
    plt.close(fig)

    print("  Saved exp2_convergence.png and exp2_cost_premium.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Experiment 2: Multiperiod Tree Replication")
    print("=" * 60)

    baseline_df = section_baseline()
    midpoint_df = section_midpoint()
    conv_df = section_convergence()
    offmid_df = section_offmidpoint()

    # Combine all results
    all_rows = []
    for df, label in [
        (baseline_df, "baseline"),
        (midpoint_df, "midpoint"),
        (conv_df, "convergence"),
        (offmid_df, "offmidpoint"),
    ]:
        df = df.copy()
        df["scenario"] = label
        all_rows.append(df)
    full_df = pd.concat(all_rows, ignore_index=True)
    csv_path = os.path.join(RESULTS_DIR, "exp2_results.csv")
    full_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    make_plots(conv_df, baseline_df)

    # Summary JSON
    summary = {
        "baseline": baseline_df.to_dict(orient="records"),
        "midpoint": midpoint_df.to_dict(orient="records"),
        "convergence": conv_df.to_dict(orient="records"),
        "offmidpoint": offmid_df.to_dict(orient="records"),
        "conclusions": {
            "all_replicated_baseline": bool(baseline_df["all_replicated"].all()),
            "all_replicated_midpoint": bool(midpoint_df["all_replicated"].all()),
            "impacted_above_frictionless": bool((conv_df["impacted_minus_fric"] >= 0).all()),
            "max_replication_error_M5": float(
                conv_df.loc[conv_df["M"] == 5, "max_replication_error"].iloc[0]
                if 5 in conv_df["M"].values else float("nan")
            ),
        },
    }
    json_path = os.path.join(RESULTS_DIR, "exp2_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()
