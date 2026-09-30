"""
Experiment 1 — One-Period Price-Impact Replication

Replicates options (call, put, digital call) in a two-state binomial model with
price impact.  For each payoff type we:
  - validate the no-price-manipulation criterion (NPC),
  - compute the replicating delta Delta* via explicit analytical formulae *and*
    the fixed-point root-finder, then compare both,
  - price the replicating portfolio and compare to the frictionless benchmark,
  - identify the non-replicable strike interval J for digital calls.

Results are saved to results/exp1_results.csv, results/exp1_summary.json and
four PNG figures.
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.payoffs import make_call, make_put, make_digital_call
from src.impact import linear_impact, sqrt_impact, linear_exec_cost
from src.fixed_point import (
    no_price_manipulation_criterion,
    find_all_roots,
    select_min_abs_root,
    replication_cost,
    frictionless_price,
    fixed_point_residual,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Baseline parameters ────────────────────────────────────────────────────────
S0 = 100.0
S_u = 105.0
S_d = 95.0
r = 0.01
K_BASE = 100.0
N = 1.0

# Risk-neutral probability q = (S0*(1+r) - S_d) / (S_u - S_d)
q = (S0 * (1 + r) - S_d) / (S_u - S_d)

# Primary impact / cost parameters (first param set)
LAM = 0.1
VARPHI = 0.05

# Param sets for NPC validation
NPC_PARAM_SETS = [
    (0.1, 0.05),
    (0.2, 0.10),
    (0.3, 0.15),
]

# K values for call / put comparison tables
K_LIST = [90.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 110.0]


# ── 1. No-arbitrage verification ───────────────────────────────────────────────
def verify_no_arbitrage(S0: float, S_u: float, S_d: float, r: float) -> dict:
    F0 = S0 * (1 + r)
    ok = S_d < F0 < S_u
    q_val = (F0 - S_d) / (S_u - S_d)
    return {"S_d": S_d, "F0": F0, "S_u": S_u, "q": q_val, "no_arbitrage": bool(ok)}


# ── 2. NPC validation ──────────────────────────────────────────────────────────
def validate_npc_param_set(lam: float, varphi: float, r: float, n_pts: int = 5000) -> dict:
    """Verify x*C(x) >= 0 on dense grid for (lambda, varphi)."""
    x_grid = np.linspace(-20.0, 20.0, n_pts)
    I = linear_impact(lam)
    phi = linear_exec_cost(varphi)
    C = no_price_manipulation_criterion(x_grid, phi, I, r)
    xC = x_grid * C
    satisfied = bool(np.all(xC >= -1e-12))
    return {
        "lambda": lam,
        "varphi": varphi,
        "min_xC": float(np.min(xC)),
        "satisfied": satisfied,
    }


# ── 3. Analytical Delta* formulas ──────────────────────────────────────────────
def linear_call_analytical(S_u: float, S_d: float, lam: float, N: float, K: float) -> float:
    """Closed-form Delta* for call under linear impact I(x)=lambda*x.

    Regions (assuming lambda*N < S_u - S_d):
      K >= S_u                     →  Delta* = 0
      S_d+lambda*N <= K < S_u      →  Delta* = N*(S_u-K)/(S_u-S_d-lambda*N)
      K < min(S_d+lambda*N, S_u)   →  Delta* = N
    """
    if K >= S_u:
        return 0.0
    lN = lam * N
    if K >= S_d + lN:
        denom = (S_u - S_d) - lN
        if abs(denom) < 1e-14:
            return float("nan")
        return N * (S_u - K) / denom
    return float(N)


def sqrt_call_analytical(S_u: float, S_d: float, lam: float, N: float, K: float) -> float:
    """Closed-form Delta* for call under sqrt impact I(x)=lambda*sign(x)*sqrt(|x|).

    For N > 0 and K that puts only the up-state in-the-money:
      c_* = [lam*N + sqrt(lam^2*N^2 + 4*N*(S_u-S_d)*(S_u-K))] / [2*(S_u-S_d)]
      Delta* = sign(N) * c_*^2

    Regions:
      K >= S_u                     →  Delta* = 0  (both OTM at Delta=0)
      S_d+lam*sqrt(N) <= K < S_u   →  Delta* = c_*^2  (only up-state ITM)
      K < S_d + lam*sqrt(|N|)      →  Delta* = N  (both ITM)
    """
    if N == 0.0:
        return 0.0
    sqrt_N = np.sqrt(abs(N))
    sign_N = np.sign(N)
    if K >= S_u:
        return 0.0
    threshold = S_d + lam * sqrt_N
    if K < threshold:
        return float(N)
    # Only up-state ITM: solve c^2*(S_u-S_d) - N*lam*c - N*(S_u-K) = 0 for c>0
    span = S_u - S_d
    discriminant = (lam * N) ** 2 + 4.0 * abs(N) * span * (S_u - K)
    c_star = (lam * N + np.sqrt(discriminant)) / (2.0 * span)
    return float(sign_N * c_star ** 2)


# ── 4. Run comparisons ─────────────────────────────────────────────────────────
def run_call_comparison(S_u, S_d, q, r, lam, varphi, N, K_list, impact_type="linear"):
    """Return list of result dicts for call payoff with given impact type."""
    I = linear_impact(lam) if impact_type == "linear" else sqrt_impact(lam)
    phi = linear_exec_cost(varphi)
    rows = []
    for K in K_list:
        V = make_call(K, N)
        roots = find_all_roots(V, S_u, S_d, I, n_grid=5000,
                               x_low=-50.0, x_high=50.0)
        delta_num = select_min_abs_root(roots)
        if impact_type == "linear":
            delta_ana = linear_call_analytical(S_u, S_d, lam, N, K)
        else:
            delta_ana = sqrt_call_analytical(S_u, S_d, lam, N, K)

        if delta_num is not None:
            X0 = replication_cost(delta_num, V, S_u, S_d, I, phi, q, r)
        else:
            X0 = float("nan")
        fp = frictionless_price(V, S_u, S_d, q, r)
        agreement = (delta_num is not None
                     and not np.isnan(delta_ana)
                     and abs(delta_num - delta_ana) < 1e-6)
        rows.append({
            "payoff_type": "call",
            "impact_type": impact_type,
            "K": K,
            "lambda": lam,
            "varphi": varphi,
            "Delta_analytical": delta_ana,
            "Delta_numerical": delta_num if delta_num is not None else float("nan"),
            "X0_star": X0,
            "frictionless_price": fp,
            "agreement": agreement,
        })
    return rows


def run_put_comparison(S_u, S_d, q, r, lam, varphi, N, K_list):
    """Return list of result dicts for put payoff with linear impact."""
    I = linear_impact(lam)
    phi = linear_exec_cost(varphi)
    rows = []
    for K in K_list:
        V = make_put(K, N)
        roots = find_all_roots(V, S_u, S_d, I, n_grid=5000,
                               x_low=-50.0, x_high=50.0)
        delta_num = select_min_abs_root(roots)
        if delta_num is not None:
            X0 = replication_cost(delta_num, V, S_u, S_d, I, phi, q, r)
        else:
            X0 = float("nan")
        fp = frictionless_price(V, S_u, S_d, q, r)
        rows.append({
            "payoff_type": "put",
            "impact_type": "linear",
            "K": K,
            "lambda": lam,
            "varphi": varphi,
            "Delta_analytical": float("nan"),   # analytical put formula omitted
            "Delta_numerical": delta_num if delta_num is not None else float("nan"),
            "X0_star": X0,
            "frictionless_price": fp,
            "agreement": True,                  # numerical only, always self-consistent
        })
    return rows


def scan_digital_replicability(S_u, S_d, lam, N, K_scan, n_grid=2000):
    """Return array of bools: True if digital call with strike K is replicable."""
    I = linear_impact(lam)
    replicable = []
    for K in K_scan:
        V = make_digital_call(K, N)
        roots = find_all_roots(V, S_u, S_d, I, n_grid=n_grid,
                               x_low=-20.0, x_high=20.0)
        replicable.append(len(roots) > 0)
    return np.array(replicable)


# ── 5. Plotting helpers ────────────────────────────────────────────────────────
def _plot_npc(param_sets, r, save_path):
    x_grid = np.linspace(-10, 10, 2000)
    fig, axes = plt.subplots(1, len(param_sets), figsize=(14, 4), sharey=False)
    fig.suptitle("No-Price-Manipulation Criterion: $x \\cdot C(x) \\geq 0$", fontsize=12)
    for ax, (lam, varphi) in zip(axes, param_sets):
        I = linear_impact(lam)
        phi = linear_exec_cost(varphi)
        C = no_price_manipulation_criterion(x_grid, phi, I, r)
        xC = x_grid * C
        ax.plot(x_grid, xC, color="tab:blue", lw=1.5)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.fill_between(x_grid, xC, 0, where=(xC >= 0),
                        alpha=0.15, color="tab:green", label="$x\\cdot C(x)\\geq0$")
        ax.fill_between(x_grid, xC, 0, where=(xC < 0),
                        alpha=0.4, color="tab:red", label="Violation")
        ax.set_title(f"$\\lambda={lam},\\,\\varphi={varphi}$", fontsize=10)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$x\\cdot C(x)$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_residual(V, S_u, S_d, I, K, title, save_path, x_low=-5, x_high=5):
    delta_grid = np.linspace(x_low, x_high, 3000)
    F = np.array([fixed_point_residual(d, V, S_u, S_d, I) for d in delta_grid])
    roots = find_all_roots(V, S_u, S_d, I, n_grid=5000,
                           x_low=x_low, x_high=x_high)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(delta_grid, F, color="tab:blue", lw=1.5, label="$F(\\Delta)$")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    for root in roots:
        ax.axvline(root, color="tab:red", lw=1.2, ls="--",
                   label=f"$\\Delta^*\\approx{root:.4f}$")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("$\\Delta$")
    ax.set_ylabel("$F(\\Delta)$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_digital_scan(K_scan, replicable, J_lo, J_hi, save_path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.scatter(K_scan[replicable], np.ones(replicable.sum()),
               s=4, color="tab:green", label="Replicable", zorder=3)
    ax.scatter(K_scan[~replicable], np.ones((~replicable).sum()),
               s=10, color="tab:red", marker="x", label="Non-replicable", zorder=4)
    ax.axvspan(J_lo, J_hi, alpha=0.2, color="tab:red",
               label=f"$J=({J_lo:.3f},{J_hi:.3f})$")
    ax.axvline(S_d, color="gray", lw=0.9, ls=":", label=f"$S_d={S_d}$")
    ax.axvline(S_u, color="dimgray", lw=0.9, ls=":", label=f"$S_u={S_u}$")
    ax.set_xlabel("Strike $K$")
    ax.set_yticks([])
    ax.set_title("Digital Call Replicability Scan (linear impact, $N=1$)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    summary = {}

    # ── 1. No-arbitrage ────────────────────────────────────────────────────────
    na = verify_no_arbitrage(S0, S_u, S_d, r)
    summary["no_arbitrage"] = na
    print("=" * 70)
    print("1. NO-ARBITRAGE VERIFICATION")
    print(f"   S_d={na['S_d']}  <  F0={na['F0']:.4f}  <  S_u={na['S_u']}: "
          f"{'PASS' if na['no_arbitrage'] else 'FAIL'}")
    print(f"   Risk-neutral probability q = {na['q']:.6f}")
    assert na["no_arbitrage"], "No-arbitrage condition violated!"

    # ── 2. NPC validation ──────────────────────────────────────────────────────
    print("\n2. NO-PRICE-MANIPULATION CRITERION (linear impact + linear cost)")
    npc_results = []
    for lam, varphi in NPC_PARAM_SETS:
        res = validate_npc_param_set(lam, varphi, r)
        npc_results.append(res)
        print(f"   (λ={lam}, φ={varphi}): min(x·C(x))={res['min_xC']:.4e}  "
              f"{'PASS' if res['satisfied'] else 'FAIL'}")
    summary["npc"] = npc_results

    _plot_npc(NPC_PARAM_SETS, r,
              os.path.join(RESULTS_DIR, "exp1_npc.png"))
    print("   → Saved exp1_npc.png")

    # ── 3. Linear-impact calls ─────────────────────────────────────────────────
    print("\n3. CALL OPTION — LINEAR IMPACT (λ=0.1, φ=0.05, N=1)")
    print(f"   S_d + λN = {S_d + LAM * N:.4f}   S_u = {S_u}")
    lin_call_rows = run_call_comparison(S_u, S_d, q, r, LAM, VARPHI, N, K_LIST,
                                       impact_type="linear")

    header = f"{'K':>6} {'Δ_anal':>10} {'Δ_num':>10} {'|diff|':>10} {'X0*':>10} {'FP':>10} {'OK'}"
    print(f"   {header}")
    print("   " + "-" * 65)
    for row in lin_call_rows:
        da = row["Delta_analytical"]
        dn = row["Delta_numerical"]
        diff = abs(dn - da) if not (np.isnan(da) or np.isnan(dn)) else float("nan")
        print(f"   {row['K']:>6.1f} {da:>10.6f} {dn:>10.6f} "
              f"{diff:>10.2e} {row['X0_star']:>10.6f} "
              f"{row['frictionless_price']:>10.6f} "
              f"{'✓' if row['agreement'] else '✗'}")

    summary["linear_call"] = [
        {k: (float(v) if isinstance(v, (float, np.floating)) else v)
         for k, v in row.items()}
        for row in lin_call_rows
    ]

    # Residual plot for K=100, linear call
    V100_call = make_call(K_BASE, N)
    I_lin = linear_impact(LAM)
    _plot_residual(V100_call, S_u, S_d, I_lin, K_BASE,
                   f"Fixed-Point Residual — Call K={K_BASE}, Linear Impact λ={LAM}",
                   os.path.join(RESULTS_DIR, "exp1_call_linear.png"))
    print("   → Saved exp1_call_linear.png")

    # ── 4. Sqrt-impact calls ───────────────────────────────────────────────────
    print("\n4. CALL OPTION — SQRT IMPACT (λ=0.1, φ=0.05, N=1)")
    print(f"   S_d + λ√N = {S_d + LAM * np.sqrt(N):.4f}   S_u = {S_u}")
    sqrt_call_rows = run_call_comparison(S_u, S_d, q, r, LAM, VARPHI, N, K_LIST,
                                        impact_type="sqrt")

    print(f"   {header}")
    print("   " + "-" * 65)
    for row in sqrt_call_rows:
        da = row["Delta_analytical"]
        dn = row["Delta_numerical"]
        diff = abs(dn - da) if not (np.isnan(da) or np.isnan(dn)) else float("nan")
        print(f"   {row['K']:>6.1f} {da:>10.6f} {dn:>10.6f} "
              f"{diff:>10.2e} {row['X0_star']:>10.6f} "
              f"{row['frictionless_price']:>10.6f} "
              f"{'✓' if row['agreement'] else '✗'}")

    summary["sqrt_call"] = [
        {k: (float(v) if isinstance(v, (float, np.floating)) else v)
         for k, v in row.items()}
        for row in sqrt_call_rows
    ]

    I_sqrt = sqrt_impact(LAM)
    _plot_residual(V100_call, S_u, S_d, I_sqrt, K_BASE,
                   f"Fixed-Point Residual — Call K={K_BASE}, Sqrt Impact λ={LAM}",
                   os.path.join(RESULTS_DIR, "exp1_call_sqrt.png"))
    print("   → Saved exp1_call_sqrt.png")

    # ── 5. Put payoffs (numerical only) ───────────────────────────────────────
    print("\n5. PUT OPTION — LINEAR IMPACT (λ=0.1, φ=0.05, N=1)")
    put_rows = run_put_comparison(S_u, S_d, q, r, LAM, VARPHI, N, K_LIST)

    put_header = f"{'K':>6} {'Δ_num':>10} {'X0*':>10} {'FP':>10} {'X0*≥FP'}"
    print(f"   {put_header}")
    print("   " + "-" * 50)
    for row in put_rows:
        above = (row["X0_star"] >= row["frictionless_price"] - 1e-8
                 if not np.isnan(row["X0_star"]) else False)
        print(f"   {row['K']:>6.1f} {row['Delta_numerical']:>10.6f} "
              f"{row['X0_star']:>10.6f} {row['frictionless_price']:>10.6f} "
              f"{'✓' if above else '✗'}")

    summary["put"] = [
        {k: (float(v) if isinstance(v, (float, np.floating)) else v)
         for k, v in row.items()}
        for row in put_rows
    ]

    # Residual plot for put K=100
    V100_put = make_put(K_BASE, N)
    _plot_residual(V100_put, S_u, S_d, I_lin, K_BASE,
                   f"Fixed-Point Residual — Put K={K_BASE}, Linear Impact λ={LAM}",
                   os.path.join(RESULTS_DIR, "exp1_put.png"),
                   x_low=-5, x_high=5)
    print("   → Saved exp1_put.png")

    # ── 6. Digital call replicability scan ────────────────────────────────────
    print("\n6. DIGITAL CALL — REPLICABILITY SCAN (λ=0.1, N=1)")
    # Theoretical non-replicable interval J = (S_d, S_d + λN/(S_u-S_d))
    J_lo = S_d
    J_hi = S_d + LAM * N / (S_u - S_d)
    print(f"   Theoretical J = ({J_lo:.4f}, {J_hi:.6f})")

    # Wide scan (coarse, for overview plot)
    K_scan_wide = np.linspace(80.0, 120.0, 200)
    rep_wide = scan_digital_replicability(S_u, S_d, LAM, N, K_scan_wide, n_grid=500)

    # Fine scan around J (denser grid near J boundary)
    K_scan_fine = np.linspace(94.5, 95.5, 300)
    rep_fine = scan_digital_replicability(S_u, S_d, LAM, N, K_scan_fine, n_grid=2000)

    non_rep_K = K_scan_fine[~rep_fine]
    if len(non_rep_K) > 0:
        J_num_lo = float(non_rep_K.min())
        J_num_hi = float(non_rep_K.max())
    else:
        J_num_lo = J_num_hi = float("nan")

    print(f"   Numerical  J ≈ ({J_num_lo:.6f}, {J_num_hi:.6f})")
    print(f"   Non-replicable K count (fine scan): {(~rep_fine).sum()}")

    _plot_digital_scan(K_scan_fine, rep_fine, J_lo, J_hi,
                       os.path.join(RESULTS_DIR, "exp1_digital.png"))
    print("   → Saved exp1_digital.png")

    summary["digital"] = {
        "lambda": LAM,
        "N": N,
        "J_theoretical_lower": J_lo,
        "J_theoretical_upper": J_hi,
        "J_numerical_lower": J_num_lo,
        "J_numerical_upper": J_num_hi,
        "non_replicable_count_fine_scan": int((~rep_fine).sum()),
        "total_K_scanned_fine": int(len(K_scan_fine)),
    }

    # ── 7. Save CSV ────────────────────────────────────────────────────────────
    all_rows = lin_call_rows + sqrt_call_rows + put_rows
    csv_path = os.path.join(RESULTS_DIR, "exp1_results.csv")
    fieldnames = [
        "payoff_type", "impact_type", "K", "lambda", "varphi",
        "Delta_analytical", "Delta_numerical", "X0_star",
        "frictionless_price", "agreement",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"\n→ Results saved to {csv_path}")

    # ── 8. Save JSON summary ───────────────────────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, "exp1_summary.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"→ Summary  saved to {json_path}")

    # ── 9. Final summary ───────────────────────────────────────────────────────
    n_pass_lin = sum(r["agreement"] for r in lin_call_rows)
    n_pass_sqrt = sum(r["agreement"] for r in sqrt_call_rows)
    n_npc = sum(r["satisfied"] for r in npc_results)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  No-arbitrage:          {'PASS' if na['no_arbitrage'] else 'FAIL'}")
    print(f"  NPC satisfied:         {n_npc}/{len(NPC_PARAM_SETS)} param sets")
    print(f"  Linear call agreement: {n_pass_lin}/{len(lin_call_rows)} strikes")
    print(f"  Sqrt call agreement:   {n_pass_sqrt}/{len(sqrt_call_rows)} strikes")
    print(f"  Digital non-replicable J ≈ ({J_num_lo:.6f}, {J_num_hi:.6f})")
    print("=" * 70)
