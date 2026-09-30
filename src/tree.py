# Custom — Context7 found no library equivalent (paper-specific f_m, g_m recursion)
"""
Multiperiod replication on a recombining fundamental-price tree with linear
permanent impact I(x)=lambda*x and linear execution cost phi(x)=varphi*x.

Key objects:
  f[m, j]          = f_m(s_j)  where s_j = S0*u^j*d^(m-1-j) is the price at t_{m-1}
                     The hedge position established at time t_m when S_{t_{m-1}} = s_j.
  g[m, j, is_up]   = g_m(s_j, s_child)  where s_j is at t_{m-1} and s_child at t_m.
                     The replication value at t_m.
"""
import warnings
from math import comb
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.optimize import brentq


class BinomialImpactTree:
    """
    Backward-induction replication on a recombining binomial tree with price impact.

    Parameters
    ----------
    S0      : initial fundamental price
    T       : maturity
    M       : number of time steps
    u, d    : up/down multipliers
    r       : risk-free rate (continuously compounded per unit time, or simple)
    lam     : permanent-impact coefficient (I(x) = lam*x)
    varphi  : execution-cost coefficient (phi(x) = varphi*x)
    V       : payoff callable  V(price) -> float
    N       : contract size (used only for fixed-point grid width)
    """

    def __init__(
        self,
        S0: float,
        T: float,
        M: int,
        u: float,
        d: float,
        r: float,
        lam: float,
        varphi: float,
        V: Callable,
        N: float = 1.0,
    ) -> None:
        h = T / M
        assert d < 1 + r * h < u, "No-arbitrage condition violated: d < 1+r*h < u required"
        assert r >= 0, "Interest rate must be non-negative"
        assert 2 * varphi >= lam, "No-price-manipulation requires 2*varphi >= lambda"

        self.S0 = S0
        self.T = T
        self.M = M
        self.h = h
        self.u = u
        self.d = d
        self.r = r
        self.lam = lam
        self.varphi = varphi
        self.V = V
        self.N = N
        self.q = (1 + r * h - d) / (u - d)
        assert 0 < self.q < 1, f"Risk-neutral probability q={self.q:.4f} not in (0,1)"

        # f[(m, j)] = f_m(S0*u^j*d^(m-1-j)), hedge at t_m given parent price at t_{m-1}
        # g[(m, j, is_up)] = g_m(s_parent, s_child), value at t_m
        self._f: Dict[Tuple[int, int], float] = {}
        self._g: Dict[Tuple[int, int, bool], float] = {}
        self.g0: float = float("nan")
        self._built = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def node_price(self, m: int, k: int) -> float:
        """Fundamental price at node (m, k): S0 * u^k * d^(m-k)."""
        return self.S0 * (self.u ** k) * (self.d ** (m - k))

    def build(self) -> None:
        """Run full backward induction and populate f, g, and g0."""
        self._run_backward_induction()
        self._built = True

    def initial_value(self) -> float:
        """Return the initial replication cost g_0(S0)."""
        if not self._built:
            raise RuntimeError("Call build() first.")
        return self.g0

    def frictionless_price(self) -> float:
        """Frictionless benchmark: E^Q[V(S_T)] / (1+r*h)^M."""
        total = 0.0
        M, q = self.M, self.q
        for k in range(M + 1):
            prob = comb(M, k) * (q ** k) * ((1 - q) ** (M - k))
            total += prob * self.V(self.node_price(M, k))
        return total / (1 + self.r * self.h) ** M

    def verify_terminal_replication(self, tol: float = 1e-6) -> Dict:
        """
        Check exact replication at every terminal node.
        Returns dict with max_error, per-node errors, and boolean flag.
        """
        if not self._built:
            raise RuntimeError("Call build() first.")
        M = self.M
        node_errors: Dict[Tuple[int, bool], float] = {}
        for j in range(M):
            # Node (M-1, j) has up-child (M, j+1) and down-child (M, j)
            f_M_j = self._f[(M, j)]
            for is_up in (True, False):
                child_k = j + 1 if is_up else j
                s_T = self.node_price(M, child_k)
                X_T = self._g[(M, j, is_up)]
                payoff = self.V(s_T + self.lam * f_M_j)
                err = abs(X_T - payoff)
                node_errors[(j, is_up)] = err
        max_err = max(node_errors.values())
        return {"max_error": max_err, "node_errors": node_errors, "all_replicated": max_err < tol}

    def get_f(self, m: int, j: int) -> float:
        """Return stored f_m value at parent-node index j."""
        return self._f[(m, j)]

    def get_g(self, m: int, j: int, is_up: bool) -> float:
        """Return stored g_m value for parent index j, up or down child."""
        return self._g[(m, j, is_up)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _penultimate_fixed_point(self, s: float) -> float:
        """
        Solve the nodewise fixed-point at penultimate step:
          x = [V(s*u + lambda*x) - V(s*d + lambda*x)] / (s*u - s*d)
        Returns the min-absolute-value root.
        """
        S_u, S_d = s * self.u, s * self.d
        lam = self.lam
        V = self.V
        span = S_u - S_d

        def F(x: float) -> float:
            return x - (V(S_u + lam * x) - V(S_d + lam * x)) / span

        # Dense grid to detect all sign changes
        half_width = max(abs(self.N) * 20, 50.0)
        grid = np.linspace(-half_width, half_width, 20_000)
        Fvals = F(grid)
        roots = []
        tol = 1e-10
        for i in range(len(grid) - 1):
            if Fvals[i] * Fvals[i + 1] < 0:
                try:
                    root = brentq(F, grid[i], grid[i + 1], xtol=tol, rtol=tol)
                    if abs(F(root)) < 1e-7:
                        roots.append(root)
                except Exception:
                    pass
        # Catch tangential zeros (e.g. OTM call at K=S_u)
        near_zero_mask = np.abs(Fvals) < tol
        for xi in grid[near_zero_mask]:
            if not any(abs(xi - r) < 1e-6 for r in roots):
                roots.append(float(xi))
        if not roots:
            return 0.0
        return min(roots, key=abs)

    def _EQ_g(self, m: int, j: int) -> float:
        """E^Q[g_m(s_j, S_{t_m}) | S_{t_{m-1}}=s_j] using stored g values."""
        return self.q * self._g[(m, j, True)] + (1 - self.q) * self._g[(m, j, False)]

    def _run_backward_induction(self) -> None:
        M = self.M
        q = self.q
        r = self.r
        h = self.h
        lam = self.lam
        varphi = self.varphi
        two_phi_lam = 2 * varphi - lam  # 2*varphi - lambda
        rh = 1 + r * h
        phi_cross = two_phi_lam + varphi * r * h  # (2*varphi-lambda + varphi*r*h)

        # ---- Step 1: Penultimate step m=M ----
        for j in range(M):  # j = 0, ..., M-1  (parent node index at t_{M-1})
            s = self.node_price(M - 1, j)
            f_val = self._penultimate_fixed_point(s)
            self._f[(M, j)] = f_val
            # Terminal value at children
            s_u = s * self.u
            s_d = s * self.d
            self._g[(M, j, True)] = self.V(s_u + lam * f_val)
            self._g[(M, j, False)] = self.V(s_d + lam * f_val)

        # ---- Step 2: Backward induction for m = M-1, ..., 1 ----
        for m in range(M - 1, 0, -1):
            # parent nodes at t_{m-1}: j = 0, ..., m-1
            for j in range(m):
                # Prices at t_m (children of parent j at t_{m-1})
                # up-child index at t_m: j+1 (since s*u = S0*u^{j+1}*d^{m-1-j})
                # down-child index at t_m: j
                f_next_up = self._f[(m + 1, j + 1)]  # f_{m+1}(s_u)
                f_next_dn = self._f[(m + 1, j)]       # f_{m+1}(s_d)

                # Conditional expectations
                EQ_gup = self._EQ_g(m + 1, j + 1)  # E^Q[g_{m+1}(s_u, .)|S_m=s_u]
                EQ_gdn = self._EQ_g(m + 1, j)       # E^Q[g_{m+1}(s_d, .)|S_m=s_d]

                # Node price spread at t_m
                s = self.node_price(m - 1, j)
                s_u = s * self.u
                s_d_node = s * self.d
                spread = s_u - s_d_node  # always positive

                denom = rh * (spread + two_phi_lam * (f_next_up - f_next_dn))
                if abs(denom) < 1e-14:
                    warnings.warn(
                        f"Near-zero denominator at node (m={m}, j={j}); "
                        "check 2*varphi >= lambda and tree parameters."
                    )
                    f_m_j = 0.0
                else:
                    num = (EQ_gup - EQ_gdn
                           + phi_cross * (f_next_up ** 2 - f_next_dn ** 2))
                    f_m_j = num / denom
                self._f[(m, j)] = f_m_j

                # Value functions g_m(parent_j, up-child) and g_m(parent_j, down-child)
                for is_up, f_next, EQ_g_child in [
                    (True,  f_next_up, EQ_gup),
                    (False, f_next_dn, EQ_gdn),
                ]:
                    g_val = (EQ_g_child / rh
                             + f_next * ((phi_cross / rh) * f_next - two_phi_lam * f_m_j))
                    self._g[(m, j, is_up)] = g_val

        # ---- Step 3: Initial value g_0(S0) ----
        # For M=1: penultimate step IS step 1, already computed in Step 1.
        #   f_1(S0) = f[(1, 0)], g[(1, 0, True/False)] already set.
        # For M>=2: f_1 must be computed via the backward formula at m=1, j=0.

        if M == 1:
            f_1 = self._f[(1, 0)]
            EQ_g1 = self._EQ_g(1, 0)
        else:
            # m=1, j=0: parent is S0; children at t_1 are (1, 1) and (1, 0)
            f_1_next_up = self._f[(2, 1)]  # f_2 when S_{t_1}=S0*u
            f_1_next_dn = self._f[(2, 0)]  # f_2 when S_{t_1}=S0*d
            EQ_g2_up = self._EQ_g(2, 1)    # E^Q[g_2(S0*u, .)|S_1=S0*u]
            EQ_g2_dn = self._EQ_g(2, 0)    # E^Q[g_2(S0*d, .)|S_1=S0*d]

            spread_1 = self.S0 * (self.u - self.d)
            denom_1 = rh * (spread_1 + two_phi_lam * (f_1_next_up - f_1_next_dn))
            if abs(denom_1) < 1e-14:
                warnings.warn("Near-zero denominator at initial node.")
                f_1 = 0.0
            else:
                num_1 = (EQ_g2_up - EQ_g2_dn
                         + phi_cross * (f_1_next_up ** 2 - f_1_next_dn ** 2))
                f_1 = num_1 / denom_1
            self._f[(1, 0)] = f_1

            # Store g_1(S0, children) for completeness
            for is_up, f_next, EQ_g_c in [
                (True,  f_1_next_up, EQ_g2_up),
                (False, f_1_next_dn, EQ_g2_dn),
            ]:
                gv = (EQ_g_c / rh
                      + f_next * ((phi_cross / rh) * f_next - two_phi_lam * f_1))
                self._g[(1, 0, is_up)] = gv

            EQ_g1 = self._EQ_g(1, 0)

        # g_0(S0) = [E^Q[g_1(S0, S_{t_1})] + (2phi-lam+phi*r*h)*f_1(S0)^2] / (1+r*h)
        self.g0 = (EQ_g1 + phi_cross * f_1 ** 2) / rh
