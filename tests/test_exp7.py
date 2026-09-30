"""
Tests for Experiment 7 — ITM Event Analysis.

Under permanent price impact, the impacted terminal price P_T = S_T + λ*Δ_T
can exceed K even when S_T < K (ITM event).  These tests verify that:
  1. The event is analytically possible (direct terminal-condition check).
  2. The event frequency is non-trivial (between 0 and some upper bound).
  3. The impacted price always exceeds the fundamental when delta >= 0.
"""
import numpy as np
import pytest
from scipy.stats import norm

from src.pde import NonlinearPDESolver
from src.terminal import modified_terminal_call, modified_terminal_call_hedge
from src.simulation import PathwiseReplicator

# ── Shared parameters ─────────────────────────────────────────────────────────
K = 100.0
T = 1.0
r = 0.05
sigma = 0.20
LAM = 0.10
VARPHI = 0.06
N = 1.0

# Wide-gap parameters for tests that require observable events
N_WIDE = 5.0
LAM_WIDE = 0.20
VARPHI_WIDE = 0.15    # 2*0.15 = 0.30 >= 0.20 ✓
K_EFF_WIDE = K - LAM_WIDE * N_WIDE   # 100 - 1.0 = 99.0


@pytest.fixture(scope='module')
def pde_replicator_wide():
    """PDE + replicator for wide-gap parameters."""
    terminal = lambda x: modified_terminal_call(
        np.asarray(x), K, N_WIDE, LAM_WIDE, VARPHI_WIDE)
    pde = NonlinearPDESolver(T=T, r=r, sigma=sigma,
                              lam=LAM_WIDE, varphi=VARPHI_WIDE,
                              x_min=50.0, x_max=200.0, Nx=100, Nt=100,
                              terminal_func=terminal)
    pde.solve()
    V_func = lambda p: N_WIDE * np.maximum(np.asarray(p, dtype=float) - K, 0.0)
    rep = PathwiseReplicator(
        S0=100.0, T=T, r=r, sigma=sigma,
        lam=LAM_WIDE, varphi=VARPHI_WIDE,
        K=K, N=N_WIDE,
        pde_u_func=pde.get_u,
        pde_ux_func=pde.get_ux,
        V_func=V_func,
    )
    return pde, rep


# ── ITM event is analytically possible ────────────────────────────────────────
class TestITMEventPossible:
    def test_terminal_hedge_N_in_gap_region(self):
        """
        For x in (K_eff, K), the terminal hedge tilde_V'(x) = N (upper region),
        so P_T = x + lambda*N > K while x < K — this IS the ITM event.
        """
        x_gap = np.array([K_EFF_WIDE + 0.1, K_EFF_WIDE + 0.5, K - 0.1])
        delta_gap = modified_terminal_call_hedge(x_gap, K, N_WIDE, LAM_WIDE, VARPHI_WIDE)

        # All deltas should equal N in the ITM region
        np.testing.assert_allclose(delta_gap, N_WIDE, atol=1e-10)

        # Fundamental prices are below K
        assert np.all(x_gap < K), "Test points should be below K"

        # Impacted prices are above K
        P_T_gap = x_gap + LAM_WIDE * delta_gap
        assert np.all(P_T_gap > K), (
            f"Impacted prices should exceed K: P_T = {P_T_gap}"
        )

    def test_gap_width_proportional_to_lambda_N(self):
        """Gap width = K - K_eff = lambda * N."""
        gap_width = K - K_EFF_WIDE
        assert abs(gap_width - LAM_WIDE * N_WIDE) < 1e-12

    def test_itm_event_in_simulation(self, pde_replicator_wide):
        """At least one simulated path satisfies S_T < K < P_T."""
        _, rep = pde_replicator_wide
        res = rep.run(n_steps=200, n_paths=500, seed=42)
        itm_mask = (res['S_T'] < K) & (res['P_T'] > K)
        n_events = int(np.sum(itm_mask))
        # With gap width 1.0 and ATM, we expect ~2% of paths
        assert n_events >= 1, (
            f"Expected at least 1 ITM event in 500 paths, got {n_events}"
        )


# ── Event frequency bounds ─────────────────────────────────────────────────────
class TestEventFrequency:
    def test_theoretical_frequency_nonzero(self):
        """Theoretical GBM probability of gap event is strictly positive."""
        # P(K_eff < S_T < K) under risk-neutral GBM
        mu = (r - 0.5 * sigma ** 2) * T
        vol = sigma * np.sqrt(T)
        d_K = (np.log(K / 100.0) - mu) / vol
        d_K_eff = (np.log(K_EFF_WIDE / 100.0) - mu) / vol
        freq = norm.cdf(d_K) - norm.cdf(d_K_eff)
        assert freq > 0.0, f"Theoretical event frequency should be > 0: {freq}"

    def test_theoretical_frequency_below_0_1(self):
        """Theoretical frequency is less than 10% for standard parameters."""
        mu = (r - 0.5 * sigma ** 2) * T
        vol = sigma * np.sqrt(T)
        d_K = (np.log(K / 100.0) - mu) / vol
        d_K_eff = (np.log(K_EFF_WIDE / 100.0) - mu) / vol
        freq = norm.cdf(d_K) - norm.cdf(d_K_eff)
        assert freq < 0.10, f"Theoretical frequency too high: {freq:.4f}"

    def test_simulated_frequency_in_plausible_range(self, pde_replicator_wide):
        """Simulated event frequency is between 0 and 0.15."""
        _, rep = pde_replicator_wide
        res = rep.run(n_steps=200, n_paths=500, seed=0)
        itm_mask = (res['S_T'] < K) & (res['P_T'] > K)
        freq = float(np.mean(itm_mask))
        assert 0.0 <= freq <= 0.15, (
            f"Simulated event frequency {freq:.4f} outside [0, 0.15]"
        )

    def test_no_false_positives_deep_otm(self, pde_replicator_wide):
        """Paths with S_T << K_eff should not be ITM events (delta ≈ 0)."""
        _, rep = pde_replicator_wide
        res = rep.run(n_steps=200, n_paths=500, seed=3)
        # Deep OTM paths: S_T < K_eff - 5
        deep_otm = res['S_T'] < K_EFF_WIDE - 5
        if deep_otm.any():
            itm_events_deep_otm = (res['P_T'][deep_otm] > K)
            # Should be very few or none
            assert float(np.mean(itm_events_deep_otm)) < 0.05, (
                "Too many ITM events for deep OTM paths"
            )


# ── Fundamental vs impacted price ─────────────────────────────────────────────
class TestFundamentalVsImpactedPrice:
    def test_impacted_price_ge_fundamental(self, pde_replicator_wide):
        """
        P_T = S_T + lambda * delta_T >= S_T for all paths
        since delta_T >= 0 (call option delta) and lambda > 0.
        """
        _, rep = pde_replicator_wide
        res = rep.run(n_steps=100, n_paths=300, seed=7)
        diff = res['P_T'] - res['S_T']
        # Lambda * delta_T >= 0
        assert np.all(diff >= -1e-8), (
            f"P_T < S_T found: min(P_T - S_T) = {diff.min():.6f}"
        )

    def test_price_divergence_proportional_to_delta(self, pde_replicator_wide):
        """P_T - S_T = lambda * delta_T (verified from formula)."""
        _, rep = pde_replicator_wide
        res = rep.run(n_steps=100, n_paths=100, seed=5)
        expected_diff = LAM_WIDE * res['delta_T']
        actual_diff = res['P_T'] - res['S_T']
        np.testing.assert_allclose(actual_diff, expected_diff, atol=1e-8)

    def test_delta_T_bounded_in_0_N(self, pde_replicator_wide):
        """Terminal hedge position should be in [0, N]."""
        _, rep = pde_replicator_wide
        res = rep.run(n_steps=100, n_paths=300, seed=12)
        delta = res['delta_T']
        assert np.all(delta >= -0.01), f"Negative delta found: min={delta.min():.5f}"
        assert np.all(delta <= N_WIDE + 0.01), f"Delta > N: max={delta.max():.5f}"

    def test_payoff_at_impacted_price(self, pde_replicator_wide):
        """Payoff = N * max(P_T - K, 0) matches V_func(P_T)."""
        _, rep = pde_replicator_wide
        res = rep.run(n_steps=100, n_paths=100, seed=8)
        payoff_expected = N_WIDE * np.maximum(res['P_T'] - K, 0.0)
        np.testing.assert_allclose(res['payoff'], payoff_expected, atol=1e-10)
