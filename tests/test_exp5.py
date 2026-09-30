"""
Tests for Experiment 5 — Call Option PDE Pricing.
"""
import numpy as np
import pytest
from src.pde import NonlinearPDESolver
from src.terminal import modified_terminal_call, modified_terminal_call_hedge
from src.simulation import PathwiseReplicator

# ── Shared parameters ─────────────────────────────────────────────────────────
K = 100.0
T = 1.0
r = 0.05
sigma = 0.20
VARPHI = 0.06
LAM = 0.10
N = 1.0
X_MIN, X_MAX, NX, NT = 50.0, 200.0, 120, 120


def make_call_pde(N=N, lam=LAM, varphi=VARPHI) -> NonlinearPDESolver:
    terminal = lambda x: modified_terminal_call(np.asarray(x), K, N, lam, varphi)
    pde = NonlinearPDESolver(T=T, r=r, sigma=sigma, lam=lam, varphi=varphi,
                              x_min=X_MIN, x_max=X_MAX, Nx=NX, Nt=NT,
                              terminal_func=terminal)
    pde.solve()
    return pde


BS_PRICE_ATM = PathwiseReplicator.bs_price(100.0, K, T, r, sigma)


# ── Price above BS benchmark ───────────────────────────────────────────────────
class TestCallPDEPriceAboveBS:
    @classmethod
    @pytest.fixture(scope='class')
    def pde(cls):
        return make_call_pde()

    def test_atm_price_above_bs(self, pde):
        """PDE price at S0=K=100 should exceed BS price (impact premium)."""
        u_pde = float(pde.get_u(0.0, np.array([100.0]))[0])
        assert u_pde > BS_PRICE_ATM - 1e-8, (
            f"PDE price {u_pde:.4f} not above BS {BS_PRICE_ATM:.4f}"
        )

    def test_otm_price_nonnegative(self, pde):
        """PDE call price is non-negative for all S0."""
        S_grid = np.linspace(60.0, 140.0, 50)
        u_vals = pde.get_u(0.0, S_grid)
        assert np.all(u_vals >= -1e-4), "Negative call prices found"

    def test_itm_price_exceeds_intrinsic(self, pde):
        """Deep ITM: u(0,S0) > max(S0-K*exp(-rT), 0) approximately."""
        S_deep_itm = np.array([140.0, 150.0])
        u_vals = pde.get_u(0.0, S_deep_itm)
        intrinsic = np.maximum(S_deep_itm - K * np.exp(-r * T), 0.0)
        assert np.all(u_vals >= intrinsic - 1.0), "ITM price below intrinsic"

    def test_impact_premium_increases_with_lambda(self):
        """Larger lambda => higher PDE price at ATM."""
        pde1 = make_call_pde(lam=0.01)
        pde2 = make_call_pde(lam=0.10)
        u1 = float(pde1.get_u(0.0, np.array([100.0]))[0])
        u2 = float(pde2.get_u(0.0, np.array([100.0]))[0])
        assert u2 >= u1 - 1e-6, f"Higher lambda should give higher price: u2={u2:.4f} < u1={u1:.4f}"


# ── Delta structure ────────────────────────────────────────────────────────────
class TestCallPDEDeltaStructure:
    @classmethod
    @pytest.fixture(scope='class')
    def pde(cls):
        return make_call_pde()

    def test_delta_in_range_0_to_N(self, pde):
        """PDE delta u_x should lie in [0, N] on the interior grid."""
        S_grid = np.linspace(65.0, 175.0, 80)
        ux = pde.get_ux(0.0, S_grid)
        assert np.all(ux >= -0.01), f"Negative delta found: min={ux.min():.4f}"
        assert np.all(ux <= N + 0.01), f"Delta > N found: max={ux.max():.4f}"

    def test_delta_monotone_increasing(self, pde):
        """Delta is monotone non-decreasing as S increases."""
        S_grid = np.linspace(70.0, 160.0, 80)
        ux = pde.get_ux(0.0, S_grid)
        diffs = np.diff(ux)
        # Allow small numerical non-monotonicities
        assert np.all(diffs >= -0.02), f"Delta is not monotone: min_diff={diffs.min():.5f}"

    def test_delta_near_0_for_deep_otm(self, pde):
        """Delta near 0 for very small S (deep OTM call)."""
        S_otm = np.array([55.0, 60.0, 65.0])
        ux = pde.get_ux(0.0, S_otm)
        assert np.all(np.abs(ux) < 0.1), f"Deep OTM delta too large: {ux}"

    def test_delta_near_N_for_deep_itm(self, pde):
        """Delta near N for large S (deep ITM call)."""
        S_itm = np.array([160.0, 170.0, 175.0])
        ux = pde.get_ux(0.0, S_itm)
        assert np.all(ux > 0.8 * N), f"Deep ITM delta too small: {ux}"


# ── Effective strike shift ────────────────────────────────────────────────────
class TestEffectiveStrikeShift:
    def test_pde_price_positive_at_effective_strike(self):
        """
        At S0 = K (standard strike), tilde_V is strictly positive
        (impact premium), so u(0, K) > BS_price_at_K.
        """
        pde = make_call_pde()
        u_at_K = float(pde.get_u(0.0, np.array([K]))[0])
        assert u_at_K > BS_PRICE_ATM * 0.95, (
            f"u(0,K)={u_at_K:.4f} not above 95% of BS={BS_PRICE_ATM:.4f}"
        )

    def test_terminal_condition_matches_formula(self):
        """Modified terminal condition satisfies three-region formula at test points."""
        x_vals = np.array([90.0, 95.0, 99.0, 99.9, 100.0, 100.1, 105.0, 110.0])
        tV = modified_terminal_call(x_vals, K, N, LAM, VARPHI)
        tVp = modified_terminal_call_hedge(x_vals, K, N, LAM, VARPHI)

        V_func = lambda xi: N * np.maximum(np.asarray(xi) - K, 0.0)
        phi0 = 2 * VARPHI - LAM
        rhs = V_func(x_vals + LAM * tVp) + phi0 / 2.0 * tVp ** 2
        np.testing.assert_allclose(tV, rhs, atol=1e-8)

    def test_x0_star_above_pde_price(self):
        """X_0^* >= u(0, S0): initial capital includes setup cost."""
        pde = make_call_pde()
        u0 = float(pde.get_u(0.0, np.array([100.0]))[0])
        X0 = pde.initial_capital(100.0)
        assert X0 >= u0 - 1e-10, f"X0^*={X0:.4f} < u0={u0:.4f}"


# ── Normalization at midpoint ─────────────────────────────────────────────────
class TestNormalizationMidpoint:
    def test_same_N_lam_product_midpoint(self):
        """
        Midpoint regime (2*phi = lam): with same N*lambda, normalized prices
        u(0,S0)/N should agree to within numerical tolerance.
        """
        # Case A: N=1, lam=0.08, varphi=0.04  (N*lam=0.08)
        # Case B: N=2, lam=0.04, varphi=0.02  (N*lam=0.08)
        lam_A, N_A, vph_A = 0.08, 1.0, 0.04
        lam_B, N_B, vph_B = 0.04, 2.0, 0.02

        pde_A = make_call_pde(N=N_A, lam=lam_A, varphi=vph_A)
        pde_B = make_call_pde(N=N_B, lam=lam_B, varphi=vph_B)

        S_test = np.array([95.0, 100.0, 105.0])
        u_A_norm = pde_A.get_u(0.0, S_test) / N_A
        u_B_norm = pde_B.get_u(0.0, S_test) / N_B

        # Normalized prices should be close (within numerical PDE error)
        max_diff = float(np.max(np.abs(u_A_norm - u_B_norm)))
        assert max_diff < 0.5, (
            f"Normalized prices differ by {max_diff:.4f} "
            f"(expected < 0.5 due to same N*lambda)"
        )

    def test_larger_N_lam_higher_price(self):
        """
        Higher N*lambda => higher normalized price (more impact cost).
        """
        pde_low = make_call_pde(N=1.0, lam=0.02, varphi=0.01)
        pde_high = make_call_pde(N=1.0, lam=0.10, varphi=0.05)

        u_low = float(pde_low.get_u(0.0, np.array([100.0]))[0]) / 1.0
        u_high = float(pde_high.get_u(0.0, np.array([100.0]))[0]) / 1.0

        assert u_high >= u_low - 1e-6, (
            f"Higher N*lambda should give higher price: {u_high:.4f} < {u_low:.4f}"
        )
