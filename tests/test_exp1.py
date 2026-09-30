"""
Pytest tests for Experiment 1 — one-period price-impact option replication.

All tests are self-contained; baseline parameters are fixed so results can be
verified analytically.

Run with:
    pytest tests/test_exp1.py
"""
import numpy as np
import pytest

from src.payoffs import (
    call_payoff, put_payoff, digital_call_payoff,
    make_call, make_put, make_digital_call,
)
from src.impact import linear_impact, sqrt_impact, linear_exec_cost, fig2_exec_cost
from src.fixed_point import (
    no_price_manipulation_criterion,
    fixed_point_residual,
    find_all_roots,
    select_min_abs_root,
    replication_cost,
    frictionless_price,
)

# ── Shared baseline parameters ────────────────────────────────────────────────
S0 = 100.0
S_u = 105.0
S_d = 95.0
r = 0.01
K_ATM = 100.0
N = 1.0
LAM = 0.1
VARPHI = 0.05
q = (S0 * (1 + r) - S_d) / (S_u - S_d)   # = 0.6


# ── Payoff tests ──────────────────────────────────────────────────────────────
class TestCallPayoffBasic:
    def test_call_payoff_non_negative(self):
        """Call payoff must be non-negative everywhere."""
        x = np.linspace(80, 120, 500)
        result = call_payoff(x, K=K_ATM, N=N)
        assert np.all(result >= 0.0)

    def test_call_payoff_zero_at_and_below_strike(self):
        """Call payoff is exactly 0 for x <= K."""
        x_below = np.array([80.0, 95.0, 99.9, 100.0])
        result = call_payoff(x_below, K=K_ATM, N=N)
        np.testing.assert_array_equal(result, 0.0)

    def test_call_payoff_intrinsic_value(self):
        """Call payoff equals N*(x-K) for x > K."""
        x_above = np.array([101.0, 105.0, 110.0])
        expected = N * (x_above - K_ATM)
        np.testing.assert_allclose(call_payoff(x_above, K_ATM, N), expected, atol=1e-14)

    def test_call_payoff_scaling_with_N(self):
        """Doubling N doubles the payoff."""
        x = np.array([105.0, 110.0])
        assert np.allclose(call_payoff(x, K_ATM, 2.0), 2.0 * call_payoff(x, K_ATM, 1.0))

    def test_make_call_factory(self):
        """make_call factory agrees with call_payoff."""
        V = make_call(K_ATM, N)
        x = np.array([90.0, 100.0, 110.0])
        np.testing.assert_allclose(V(x), call_payoff(x, K_ATM, N), atol=1e-14)


class TestPutPayoffBasic:
    def test_put_payoff_non_negative(self):
        """Put payoff must be non-negative everywhere."""
        x = np.linspace(80, 120, 500)
        result = put_payoff(x, K=K_ATM, N=N)
        assert np.all(result >= 0.0)

    def test_put_payoff_zero_at_and_above_strike(self):
        """Put payoff is exactly 0 for x >= K."""
        x_above = np.array([100.0, 100.1, 105.0, 120.0])
        result = put_payoff(x_above, K=K_ATM, N=N)
        np.testing.assert_array_equal(result, 0.0)

    def test_put_payoff_intrinsic_value(self):
        """Put payoff equals N*(K-x) for x < K."""
        x_below = np.array([90.0, 95.0, 99.0])
        expected = N * (K_ATM - x_below)
        np.testing.assert_allclose(put_payoff(x_below, K_ATM, N), expected, atol=1e-14)

    def test_make_put_factory(self):
        """make_put factory agrees with put_payoff."""
        V = make_put(K_ATM, N)
        x = np.array([90.0, 100.0, 110.0])
        np.testing.assert_allclose(V(x), put_payoff(x, K_ATM, N), atol=1e-14)


class TestDigitalCallBasic:
    def test_digital_call_is_zero_or_N(self):
        """Digital call output is exactly 0 or N at every grid point."""
        x = np.linspace(90, 110, 1000)
        result = digital_call_payoff(x, K=K_ATM, N=N)
        assert np.all((result == 0.0) | (result == N))

    def test_digital_call_strict_inequality(self):
        """Payoff is 0 at strike (strict x > K)."""
        assert digital_call_payoff(np.array([K_ATM]), K=K_ATM, N=N)[0] == 0.0

    def test_digital_call_above_strike(self):
        """Payoff equals N strictly above strike."""
        assert digital_call_payoff(np.array([K_ATM + 0.001]), K=K_ATM, N=N)[0] == N

    def test_digital_call_scaling_with_N(self):
        """Payoff scales linearly with N above strike."""
        x = np.array([110.0])
        assert digital_call_payoff(x, K_ATM, 3.0)[0] == 3.0


# ── Market sanity ─────────────────────────────────────────────────────────────
class TestNoArbitrage:
    def test_no_arbitrage(self):
        """S_d < S0*(1+r) < S_u must hold."""
        F0 = S0 * (1 + r)
        assert S_d < F0 < S_u, (
            f"No-arbitrage violated: S_d={S_d}, F0={F0:.4f}, S_u={S_u}"
        )

    def test_risk_neutral_prob_in_unit_interval(self):
        """Risk-neutral probability q must lie strictly in (0, 1)."""
        assert 0.0 < q < 1.0

    def test_q_exact_value(self):
        """q = (S0*(1+r) - S_d)/(S_u-S_d) = 0.6 for baseline params."""
        assert abs(q - 0.6) < 1e-14


# ── NPC tests ─────────────────────────────────────────────────────────────────
class TestNoPriceManipulation:
    """x·C(x) >= 0 for all x (no-price-manipulation criterion)."""

    PARAM_SETS = [(0.1, 0.05), (0.2, 0.10), (0.3, 0.15)]

    @pytest.mark.parametrize("lam,varphi", PARAM_SETS)
    def test_no_price_manipulation(self, lam, varphi):
        x_grid = np.linspace(-20.0, 20.0, 10_000)
        I = linear_impact(lam)
        phi = linear_exec_cost(varphi)
        C = no_price_manipulation_criterion(x_grid, phi, I, r)
        xC = x_grid * C
        assert np.all(xC >= -1e-12), (
            f"NPC violated for λ={lam}, φ={varphi}: min(x·C(x))={xC.min():.4e}"
        )

    def test_npc_linear_formula(self):
        """For linear phi and I, C(x) = x*(phi*(2+r) - lam); NPC iff phi*(2+r)>=lam."""
        lam, varphi = 0.1, 0.05
        x_grid = np.linspace(-10, 10, 500)
        I = linear_impact(lam)
        phi = linear_exec_cost(varphi)
        C = no_price_manipulation_criterion(x_grid, phi, I, r)
        expected_C = x_grid * (varphi * (2 + r) - lam)
        np.testing.assert_allclose(C, expected_C, atol=1e-12)


# ── Fixed-point solver tests ──────────────────────────────────────────────────
class TestLinearImpactCallAnalyticalVsNumerical:
    """Analytical Delta* formula must match numerical root-finding to 1e-8."""

    def _analytical(self, K):
        """Closed-form Delta* for call under linear impact."""
        if K >= S_u:
            return 0.0
        lN = LAM * N
        if K >= S_d + lN:
            return N * (S_u - K) / ((S_u - S_d) - lN)
        return float(N)

    @pytest.mark.parametrize("K", [90.0, 95.0, 97.0, 100.0, 102.0, 105.0, 110.0])
    def test_linear_impact_call_analytical_vs_numerical(self, K):
        V = make_call(K, N)
        I = linear_impact(LAM)
        roots = find_all_roots(V, S_u, S_d, I, n_grid=5000,
                               x_low=-50.0, x_high=50.0)
        delta_num = select_min_abs_root(roots)
        delta_ana = self._analytical(K)
        assert delta_num is not None, f"No root found for K={K}"
        assert abs(delta_num - delta_ana) < 1e-8, (
            f"K={K}: analytical={delta_ana:.8f}, numerical={delta_num:.8f}"
        )


class TestSqrtImpactCall:
    """c_* formula must match numerical root to 1e-6."""

    def _c_star(self, K):
        span = S_u - S_d
        disc = (LAM * N) ** 2 + 4.0 * abs(N) * span * (S_u - K)
        return (LAM * N + np.sqrt(disc)) / (2.0 * span)

    def _analytical(self, K):
        if K >= S_u:
            return 0.0
        sqrt_N = np.sqrt(abs(N))
        if K < S_d + LAM * sqrt_N:
            return float(N)
        c = self._c_star(K)
        return float(np.sign(N) * c ** 2)

    @pytest.mark.parametrize("K", [96.0, 98.0, 100.0, 102.0, 104.0])
    def test_sqrt_impact_call(self, K):
        V = make_call(K, N)
        I = sqrt_impact(LAM)
        roots = find_all_roots(V, S_u, S_d, I, n_grid=5000,
                               x_low=-50.0, x_high=50.0)
        delta_num = select_min_abs_root(roots)
        delta_ana = self._analytical(K)
        assert delta_num is not None, f"No root found for K={K}"
        assert abs(delta_num - delta_ana) < 1e-6, (
            f"K={K}: analytical={delta_ana:.8f}, numerical={delta_num:.8f}"
        )

    def test_c_star_formula_deep_itm(self):
        """For K < S_d + lambda*sqrt(N), Delta* = N (both states ITM)."""
        K_deep = S_d + LAM * np.sqrt(N) - 0.5   # e.g. 94.6
        delta_ana = self._analytical(K_deep)
        assert abs(delta_ana - N) < 1e-14

    def test_c_star_formula_otm(self):
        """For K >= S_u, Delta* = 0."""
        delta_ana = self._analytical(S_u + 1.0)
        assert delta_ana == 0.0


# ── Replication cost tests ────────────────────────────────────────────────────
class TestReplicationCost:
    def test_replication_cost_above_frictionless_call(self):
        """For a call with positive Delta*, X0* >= frictionless price."""
        V = make_call(K_ATM, N)
        I = linear_impact(LAM)
        phi = linear_exec_cost(VARPHI)
        roots = find_all_roots(V, S_u, S_d, I, n_grid=5000,
                               x_low=-50.0, x_high=50.0)
        delta = select_min_abs_root(roots)
        assert delta is not None
        X0 = replication_cost(delta, V, S_u, S_d, I, phi, q, r)
        fp = frictionless_price(V, S_u, S_d, q, r)
        assert X0 >= fp - 1e-8, (
            f"Replication cost {X0:.8f} < frictionless price {fp:.8f}"
        )

    def test_replication_cost_above_frictionless_put(self):
        """For a put, X0* >= frictionless put price (selling pushes price down)."""
        V = make_put(K_ATM, N)
        I = linear_impact(LAM)
        phi = linear_exec_cost(VARPHI)
        roots = find_all_roots(V, S_u, S_d, I, n_grid=5000,
                               x_low=-50.0, x_high=50.0)
        delta = select_min_abs_root(roots)
        assert delta is not None
        X0 = replication_cost(delta, V, S_u, S_d, I, phi, q, r)
        fp = frictionless_price(V, S_u, S_d, q, r)
        assert X0 >= fp - 1e-8, (
            f"Replication cost {X0:.8f} < frictionless price {fp:.8f}"
        )

    def test_frictionless_call_price_exact(self):
        """Frictionless call price = (q*(S_u-K)^+ + (1-q)*(S_d-K)^+)/(1+r)."""
        V = make_call(K_ATM, N)
        fp = frictionless_price(V, S_u, S_d, q, r)
        expected = (q * max(S_u - K_ATM, 0) + (1 - q) * max(S_d - K_ATM, 0)) / (1 + r)
        assert abs(fp - expected) < 1e-14

    def test_frictionless_price_zero_otm_call(self):
        """Frictionless price of a deep-OTM call is 0."""
        V = make_call(200.0, N)
        fp = frictionless_price(V, S_u, S_d, q, r)
        assert abs(fp) < 1e-14


# ── Fixed-point residual at root ──────────────────────────────────────────────
class TestFixedPointResidualAtRoot:
    @pytest.mark.parametrize("K", [97.0, 100.0, 103.0])
    def test_fixed_point_residual_at_root(self, K):
        """F(Delta*) must be approximately 0 for genuine roots."""
        V = make_call(K, N)
        I = linear_impact(LAM)
        roots = find_all_roots(V, S_u, S_d, I, n_grid=5000,
                               x_low=-50.0, x_high=50.0)
        assert len(roots) > 0, f"No root found for K={K}"
        for root in roots:
            resid = abs(fixed_point_residual(root, V, S_u, S_d, I))
            assert resid < 1e-8, (
                f"K={K}: |F(Delta*={root:.8f})| = {resid:.4e} >= 1e-8"
            )

    def test_fixed_point_residual_at_put_root(self):
        """F(Delta*) ≈ 0 for the put fixed-point solution."""
        V = make_put(K_ATM, N)
        I = linear_impact(LAM)
        roots = find_all_roots(V, S_u, S_d, I, n_grid=5000,
                               x_low=-50.0, x_high=50.0)
        assert len(roots) > 0
        root = select_min_abs_root(roots)
        resid = abs(fixed_point_residual(root, V, S_u, S_d, I))
        assert resid < 1e-8


# ── Digital call non-replicability ────────────────────────────────────────────
class TestDigitalNonReplicability:
    """For K in J = (S_d, S_d + lambda*N/(S_u-S_d)), no fixed-point root exists."""

    J_lo = S_d                                     # 95.0
    J_hi = S_d + LAM * N / (S_u - S_d)            # 95.01

    @pytest.mark.parametrize("K_inner", [95.002, 95.005, 95.008])
    def test_digital_non_replicability(self, K_inner):
        """Strikes strictly inside J have no replicating delta."""
        V = make_digital_call(K_inner, N)
        I = linear_impact(LAM)
        roots = find_all_roots(V, S_u, S_d, I, n_grid=10_000,
                               x_low=-20.0, x_high=20.0)
        assert len(roots) == 0, (
            f"Expected no root for K={K_inner} in J, but found: {roots}"
        )

    def test_digital_replicable_below_J(self):
        """K <= S_d: digital call is in-the-money for both states → Delta*=0."""
        K_below = S_d - 0.5
        V = make_digital_call(K_below, N)
        I = linear_impact(LAM)
        roots = find_all_roots(V, S_u, S_d, I, n_grid=5000,
                               x_low=-20.0, x_high=20.0)
        assert len(roots) > 0, f"Expected a root for K={K_below} < S_d"

    def test_digital_replicable_above_J(self):
        """K in (J_hi, S_u): digital call is replicable with Delta*=N/(S_u-S_d)."""
        K_above = self.J_hi + 0.1   # 95.11 — safely inside (J_hi, S_u)
        V = make_digital_call(K_above, N)
        I = linear_impact(LAM)
        roots = find_all_roots(V, S_u, S_d, I, n_grid=10_000,
                               x_low=-20.0, x_high=20.0)
        assert len(roots) > 0, f"Expected a root for K={K_above} above J"

    def test_J_bounds_correct(self):
        """Verify the theoretical J bounds against the formula."""
        J_hi_expected = S_d + LAM * N / (S_u - S_d)
        assert abs(self.J_hi - J_hi_expected) < 1e-14


# ── Edge case: empty roots ────────────────────────────────────────────────────
class TestEdgeCasesEmptyRoots:
    def test_select_min_abs_root_empty(self):
        """select_min_abs_root returns None when list is empty."""
        result = select_min_abs_root([])
        assert result is None

    def test_select_min_abs_root_single(self):
        """select_min_abs_root of a single element returns that element."""
        assert select_min_abs_root([3.7]) == 3.7

    def test_select_min_abs_root_chooses_smallest_abs(self):
        """select_min_abs_root picks the element closest to 0."""
        roots = [-2.0, 0.5, 1.5]
        assert select_min_abs_root(roots) == 0.5

    def test_find_all_roots_deep_otm_call_returns_zero_root(self):
        """OTM call (K >> S_u): Delta*=0 is the only fixed point."""
        V = make_call(200.0, N)
        I = linear_impact(LAM)
        roots = find_all_roots(V, S_u, S_d, I, n_grid=5000,
                               x_low=-50.0, x_high=50.0)
        assert len(roots) == 1
        assert abs(roots[0]) < 1e-6

    def test_digital_in_J_yields_no_roots(self):
        """Confirm find_all_roots returns [] for digital call inside J."""
        K_J = (S_d + S_d + LAM * N / (S_u - S_d)) / 2  # midpoint of J
        V = make_digital_call(K_J, N)
        I = linear_impact(LAM)
        roots = find_all_roots(V, S_u, S_d, I, n_grid=10_000,
                               x_low=-20.0, x_high=20.0)
        assert roots == [], f"Expected no roots inside J, got {roots}"


# ── Impact function sanity ────────────────────────────────────────────────────
class TestImpactFunctions:
    def test_linear_impact_values(self):
        I = linear_impact(0.1)
        np.testing.assert_allclose(I(np.array([0.0, 1.0, -2.0])),
                                   np.array([0.0, 0.1, -0.2]), atol=1e-14)

    def test_sqrt_impact_sign(self):
        """Sqrt impact has same sign as its argument."""
        I = sqrt_impact(0.1)
        assert I(2.0) > 0.0
        assert I(-2.0) < 0.0
        assert I(0.0) == 0.0

    def test_sqrt_impact_symmetric_magnitude(self):
        """|I(x)| = |I(-x)| for sqrt impact."""
        I = sqrt_impact(0.1)
        x = np.array([1.0, 4.0, 9.0])
        np.testing.assert_allclose(np.abs(I(x)), np.abs(I(-x)), atol=1e-14)

    def test_fig2_exec_cost_continuity(self):
        """fig2_exec_cost is continuous at the threshold."""
        from src.impact import fig2_exec_cost
        lam, varphi = 0.2, 0.05
        phi = fig2_exec_cost(lam, varphi)
        thresh = 4 * varphi**2 / lam**2
        eps = 1e-8
        left = phi(np.array([thresh - eps]))
        right = phi(np.array([thresh + eps]))
        assert abs(left[0] - right[0]) < 1e-4
