# Custom — Context7 found no library equivalent (paper Eq. fixed-point)
import numpy as np
from scipy.optimize import brentq
from typing import Callable, List, Tuple, Optional


def no_price_manipulation_criterion(
    x_grid: np.ndarray,
    phi: Callable,
    I: Callable,
    r: float
) -> np.ndarray:
    """Compute C(x) = phi(x)*(1+r) - phi(-x) - I(x) and check x*C(x) >= 0."""
    C = phi(x_grid) * (1 + r) - phi(-x_grid) - I(x_grid)
    return C


def fixed_point_residual(
    Delta: float,
    V: Callable,
    S_u: float,
    S_d: float,
    I: Callable
) -> float:
    """F(Delta) = Delta - [V(S_u+I(Delta)) - V(S_d+I(Delta))] / (S_u - S_d)"""
    return Delta - (V(S_u + I(Delta)) - V(S_d + I(Delta))) / (S_u - S_d)


def find_all_roots(
    V: Callable,
    S_u: float,
    S_d: float,
    I: Callable,
    n_grid: int = 5000,
    x_low: float = -50.0,
    x_high: float = 50.0,
    tol: float = 1e-10
) -> List[float]:
    """Find all roots of the fixed-point equation using sign-change detection on dense grid.

    A residual check (|F(root)| < 1e-6) is applied after brentq to discard
    spurious sign-changes that arise from payoff discontinuities (e.g. digital
    calls) rather than genuine zeros.
    """
    # Include 0 explicitly so tangent zeros (e.g. OTM call at K=S_u) are detected
    grid = np.unique(np.sort(np.append(np.linspace(x_low, x_high, n_grid), [0.0])))
    F = np.array([fixed_point_residual(x, V, S_u, S_d, I) for x in grid])
    roots = []
    for i in range(len(grid) - 1):
        if F[i] * F[i + 1] < 0:
            try:
                root = brentq(fixed_point_residual, grid[i], grid[i + 1],
                              args=(V, S_u, S_d, I), xtol=tol, rtol=tol)
                # Reject spurious roots caused by payoff discontinuities
                if abs(fixed_point_residual(root, V, S_u, S_d, I)) < 1e-6:
                    roots.append(root)
            except Exception:
                pass
        elif abs(F[i]) < tol:
            roots.append(float(grid[i]))
    # Deduplicate
    roots_clean = []
    for r_val in roots:
        if not any(abs(r_val - existing) < tol * 100 for existing in roots_clean):
            roots_clean.append(r_val)
    return roots_clean


def select_min_abs_root(roots: List[float]) -> Optional[float]:
    """Select the minimum-absolute-value root (paper's Delta_min^* selection rule)."""
    if not roots:
        return None
    return min(roots, key=abs)


def replication_cost(
    Delta: float,
    V: Callable,
    S_u: float,
    S_d: float,
    I: Callable,
    phi: Callable,
    q: float,
    r: float
) -> float:
    """X_0^* = (E^Q[V(P_T^Delta)] + Delta*C(Delta)) / (1+r)"""
    P_u = S_u + I(Delta)
    P_d = S_d + I(Delta)
    EQ_V = q * V(P_u) + (1 - q) * V(P_d)
    C_Delta = phi(Delta) * (1 + r) - phi(-Delta) - I(Delta)
    return (EQ_V + Delta * C_Delta) / (1 + r)


def frictionless_price(
    V: Callable,
    S_u: float,
    S_d: float,
    q: float,
    r: float
) -> float:
    """Frictionless benchmark: E^Q[V(S_T)] / (1+r)"""
    return (q * V(S_u) + (1 - q) * V(S_d)) / (1 + r)
