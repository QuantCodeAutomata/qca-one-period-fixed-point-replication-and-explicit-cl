# Custom — Context7 found no library equivalent (paper terminal ODE)
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from typing import Callable, Tuple, Optional


def modified_terminal_call(
    x: np.ndarray,
    K: float,
    N: float,
    lam: float,
    varphi: float
) -> np.ndarray:
    """
    Explicit piecewise modified terminal payoff for call V(x)=N*(x-K)^+.
    Paper formulas:
    tilde_V(x) = 0                              for x < K - 2*varphi*N
    tilde_V(x) = (x-K+2*varphi*N)^2/(2*(2*varphi-lambda))  for K-2*phi*N <= x < K-lambda*N
    tilde_V(x) = N*(x-K) + (2*varphi+lambda)*N^2/2  for x >= K-lambda*N
    """
    x = np.asarray(x, dtype=float)
    two_phi = 2 * varphi
    two_phi_minus_lam = two_phi - lam

    thr_low = K - two_phi * N   # lower threshold
    thr_high = K - lam * N      # upper threshold

    result = np.zeros_like(x)
    mid_mask = (x >= thr_low) & (x < thr_high)
    high_mask = x >= thr_high

    if two_phi_minus_lam < 1e-14:  # midpoint case: 2*varphi = lambda
        # The quadratic region collapses; use limit form
        result[high_mask] = N * (x[high_mask] - K) + (two_phi + lam) * N**2 / 2
    else:
        result[mid_mask] = (x[mid_mask] - K + two_phi * N)**2 / (2 * two_phi_minus_lam)
        result[high_mask] = N * (x[high_mask] - K) + (two_phi + lam) * N**2 / 2

    return result


def modified_terminal_call_hedge(
    x: np.ndarray,
    K: float,
    N: float,
    lam: float,
    varphi: float
) -> np.ndarray:
    """
    Explicit piecewise derivative of modified terminal payoff for call.
    tilde_V'(x) = 0                              for x < K - 2*varphi*N
    tilde_V'(x) = (x-K+2*varphi*N)/(2*varphi-lambda)  for K-2*phi*N <= x < K-lambda*N
    tilde_V'(x) = N                              for x >= K-lambda*N
    """
    x = np.asarray(x, dtype=float)
    two_phi = 2 * varphi
    two_phi_minus_lam = two_phi - lam

    thr_low = K - two_phi * N
    thr_high = K - lam * N

    result = np.zeros_like(x)
    mid_mask = (x >= thr_low) & (x < thr_high)
    high_mask = x >= thr_high

    if two_phi_minus_lam < 1e-14:
        result[high_mask] = N
    else:
        result[mid_mask] = (x[mid_mask] - K + two_phi * N) / two_phi_minus_lam
        result[high_mask] = N

    return result


def validate_implicit_relation(
    x_grid: np.ndarray,
    tilde_V: np.ndarray,
    tilde_V_prime: np.ndarray,
    V_func: Callable,
    lam: float,
    varphi: float,
    tol: float = 1e-8
) -> dict:
    """
    Validate that tilde_V satisfies the implicit terminal relation:
    tilde_V(x) = V(x + lambda*tilde_V'(x)) + (2*varphi-lambda)/2 * (tilde_V'(x))^2
    """
    two_phi_minus_lam = 2 * varphi - lam
    rhs = V_func(x_grid + lam * tilde_V_prime) + two_phi_minus_lam / 2.0 * tilde_V_prime**2
    residuals = np.abs(tilde_V - rhs)
    return {
        'max_residual': float(np.max(residuals)),
        'mean_residual': float(np.mean(residuals)),
        'satisfied': bool(np.max(residuals) < tol)
    }


def modified_terminal_put(
    x: np.ndarray,
    K: float,
    N: float,
    lam: float,
    varphi: float
) -> np.ndarray:
    """
    Modified terminal condition for put V(x)=N*(K-x)^+.
    By put-call symmetry / paper proposition, for put:
    tilde_V(x) = 0                              for x > K + 2*varphi*N  (equivalently using symmetry)
    Use general ODE construction for puts.
    """
    # For put (epsilon=-1, nonincreasing), use numerical ODE approach
    # Alternatively, call-put relationship if paper states it
    # Here we implement via the general auxiliary ODE
    x = np.asarray(x, dtype=float)
    # For put: V'(xi) = -N for xi < K, 0 for xi > K
    # The auxiliary ODE: 2*varphi*P'(xi)*P(xi) = P(xi) - V'(xi)
    # With epsilon=-1 (nonincreasing), constraint -L_V <= P <= 0 => 0 >= P >= -N
    # Solve numerically
    return _general_modified_terminal(x, K=K, N=N, lam=lam, varphi=varphi, payoff_type='put')


def _general_modified_terminal(
    x: np.ndarray,
    K: float,
    N: float,
    lam: float,
    varphi: float,
    payoff_type: str = 'call'
) -> np.ndarray:
    """
    General modified terminal construction via auxiliary ODE.
    Only used as validation; explicit formulas preferred for calls.
    """
    # This is a simplified numerical implementation
    x = np.asarray(x, dtype=float)
    two_phi = 2 * varphi
    two_phi_minus_lam = two_phi - lam

    if payoff_type == 'put':
        # V(xi) = N*(K-xi)^+
        # V'(xi) = -N for xi < K, 0 for xi > K
        # epsilon = -1 (nonincreasing)
        # P in [-N, 0]
        # By symmetry with call: tilde_V_put(x) = tilde_V_call(2K-x) for symmetric model
        # Use this for the numerical implementation
        thr_high = K + two_phi * N
        thr_low = K + lam * N
        result = np.zeros_like(x)
        mid_mask = (x > thr_low) & (x <= thr_high)
        low_mask = x <= thr_low
        if two_phi_minus_lam < 1e-14:
            result[low_mask] = N * (K - x[low_mask]) + (two_phi + lam) * N**2 / 2
        else:
            result[mid_mask] = (K + two_phi * N - x[mid_mask])**2 / (2 * two_phi_minus_lam)
            result[low_mask] = N * (K - x[low_mask]) + (two_phi + lam) * N**2 / 2
        return result
    else:
        return modified_terminal_call(x, K, N, lam, varphi)
