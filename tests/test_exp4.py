"""
Tests for Experiment 4 — Quadratic Benchmark and Nonlinear PDE Solver.
"""
import numpy as np
import pytest
from src.pde import QuadraticBenchmark, NonlinearPDESolver

# ── Shared parameters ─────────────────────────────────────────────────────────
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


def make_bench(**kwargs) -> QuadraticBenchmark:
    defaults = dict(S0=S0, T=T, r=r, sigma=sigma, lam=LAM, varphi=VARPHI,
                    alpha=ALPHA, beta=BETA, gamma=GAMMA, N=N)
    defaults.update(kwargs)
    b = QuadraticBenchmark(**defaults)
    b.solve_ode()
    return b


# ── Terminal coefficients ──────────────────────────────────────────────────────
class TestQuadraticTerminalCoefficients:
    def test_consistent_with_self_consistency_equation(self):
        """
        a_T, b_T, c_T must satisfy:
          tilde_V(x) = V(x + lam*tilde_V'(x)) + (2phi-lam)/2*(tilde_V'(x))^2
        at multiple x values.
        """
        b = make_bench()
        a_T, b_T, c_T = b._terminal_coefficients()
        phi0 = 2 * VARPHI - LAM
        lam = LAM

        V = lambda xi: N * (ALPHA * xi ** 2 + BETA * xi + GAMMA)

        x_test = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        tilde_V = a_T * x_test ** 2 + b_T * x_test + c_T
        tilde_Vp = 2 * a_T * x_test + b_T

        rhs = V(x_test + lam * tilde_Vp) + phi0 / 2.0 * tilde_Vp ** 2
        np.testing.assert_allclose(tilde_V, rhs, atol=1e-6, rtol=1e-5)

    def test_a_T_near_N_alpha_when_frictionless(self):
        """With lam→0, varphi→0, a_T → N*alpha."""
        b = QuadraticBenchmark(S0=S0, T=T, r=r, sigma=sigma,
                               lam=1e-6, varphi=1e-6, alpha=ALPHA,
                               beta=BETA, gamma=GAMMA, N=N)
        a_T, _, _ = b._terminal_coefficients()
        assert abs(a_T - N * ALPHA) < 1e-5

    def test_2varphi_ge_lambda_assertion(self):
        """Constructor asserts 2*varphi >= lambda."""
        with pytest.raises(AssertionError):
            QuadraticBenchmark(S0=S0, T=T, r=r, sigma=sigma,
                               lam=0.5, varphi=0.1,   # 2*0.1 < 0.5
                               alpha=ALPHA, beta=BETA, gamma=GAMMA, N=N)


# ── ODE solution properties ────────────────────────────────────────────────────
class TestODESolution:
    def test_a_T_matches_initial_condition(self):
        """a(T) from ODE must equal a_T from terminal condition."""
        b = make_bench()
        a_T, b_T, c_T = b._terminal_coefficients()
        a_ode_T = float(b._a_func(T))
        np.testing.assert_allclose(a_ode_T, a_T, atol=1e-6)

    def test_a_increases_in_backward_time(self):
        """
        da/dtau > 0 for a > 0, so integrating backward from t=T to t=0
        gives a(0) >= a(T) = a_T.
        """
        b = make_bench()
        a0 = float(b._a_func(0.0))
        a_T = b._terminal_coefficients()[0]
        # a(0) >= a(T) because da/dtau > 0 in the τ=T-t convention
        assert a0 >= a_T - 1e-12

    def test_u_shape_quadratic_in_x(self):
        """u(t,x) should be quadratic: u(t,x) - u(t,100) = a*(x-100)*(x+100) + b*(x-100)."""
        b = make_bench()
        x = np.array([90.0, 100.0, 110.0])
        u = b.u(0.5, x)
        # Finite-difference second derivative should equal 2*a(0.5)
        u_xx_fd = (u[0] - 2 * u[1] + u[2]) / 10.0 ** 2
        a05 = float(b._a_func(0.5))
        np.testing.assert_allclose(u_xx_fd, 2 * a05, rtol=1e-4)


# ── PDE solver vs analytical ───────────────────────────────────────────────────
# Wide domain avoids BC contamination; Nx=300,Nt=200 runs in ~0.2 s.
X_MIN_PDE, X_MAX_PDE, NX_PDE, NT_PDE = 20.0, 200.0, 300, 200


class TestPDESolverVsAnalytical:
    @classmethod
    @pytest.fixture(scope='class')
    def solved_pair(cls):
        b = make_bench()
        a_T, b_T, c_T = b._terminal_coefficients()
        term = lambda x: a_T * x ** 2 + b_T * x + c_T
        pde = NonlinearPDESolver(T=T, r=r, sigma=sigma, lam=LAM, varphi=VARPHI,
                                  x_min=X_MIN_PDE, x_max=X_MAX_PDE,
                                  Nx=NX_PDE, Nt=NT_PDE,
                                  terminal_func=term)
        pde.solve()
        return b, pde

    def test_sup_norm_error_lt_threshold(self, solved_pair):
        """sup-norm error on interior sub-grid at t=0 should be < 0.01."""
        b, pde = solved_pair
        # Use PDE grid points to avoid interpolation error
        mask = (pde.x_grid >= 80.0) & (pde.x_grid <= 140.0)
        x_cmp = pde.x_grid[mask]
        u_ana = b.u(0.0, x_cmp)
        u_pde = pde.get_u(0.0, x_cmp)
        sup_err = float(np.max(np.abs(u_pde - u_ana)))
        assert sup_err < 0.01, f"sup-norm error = {sup_err:.5f} >= 0.01"

    def test_rmse_error_lt_threshold(self, solved_pair):
        """RMSE on interior PDE grid at t=0 should be < 0.005."""
        b, pde = solved_pair
        mask = (pde.x_grid >= 80.0) & (pde.x_grid <= 140.0)
        x_cmp = pde.x_grid[mask]
        u_ana = b.u(0.0, x_cmp)
        u_pde = pde.get_u(0.0, x_cmp)
        rmse = float(np.sqrt(np.mean((u_pde - u_ana) ** 2)))
        assert rmse < 0.005, f"RMSE = {rmse:.5f} >= 0.005"

    def test_terminal_condition_exact(self, solved_pair):
        """PDE u(T,x) on grid nodes matches terminal condition to interpolation precision."""
        b, pde = solved_pair
        a_T, b_T, c_T = b._terminal_coefficients()
        # Evaluate at PDE grid nodes so interpolation is exact
        mask = (pde.x_grid >= 80.0) & (pde.x_grid <= 140.0)
        x_cmp = pde.x_grid[mask]
        u_term = pde.get_u(T, x_cmp)
        u_exact = a_T * x_cmp ** 2 + b_T * x_cmp + c_T
        np.testing.assert_allclose(u_term, u_exact, atol=1e-6)


# ── Initial capital ───────────────────────────────────────────────────────────
class TestInitialCapital:
    def test_impact_raises_initial_capital(self):
        """X_0^* with friction >= X_0^* without friction."""
        b_friction = make_bench()
        b_free = make_bench(lam=0.0, varphi=0.0)
        assert b_friction.initial_capital() >= b_free.initial_capital() - 1e-10

    def test_midpoint_simplification(self):
        """
        At 2*varphi = lambda (midpoint), the extra term (2phi-lam)/2 * ux^2 = 0,
        so X_0^* = u(0, S0) exactly.
        """
        b_mid = make_bench(varphi=LAM / 2)
        X0 = b_mid.initial_capital()
        u0 = float(b_mid.u(0.0, np.array([S0]))[0])
        # (2*varphi - lam)/2 = 0 at midpoint, so X0 = u(0,S0)
        np.testing.assert_allclose(X0, u0, atol=1e-10)

    def test_initial_capital_scales_with_N(self):
        """X_0^*(2N) / X_0^*(N) > 2 (superlinear growth due to quadratic payoff)."""
        b1 = make_bench(N=1.0)
        b2 = make_bench(N=2.0)
        ratio = b2.initial_capital() / b1.initial_capital()
        assert ratio > 2.0
