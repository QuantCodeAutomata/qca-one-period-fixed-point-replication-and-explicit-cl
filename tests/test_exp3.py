"""
Pytest tests for exp 3 — modified terminal condition (src/terminal.py).

All test functions are self-contained; parameters are fixed so results can be
verified analytically.
"""
import numpy as np
import pytest

from src.terminal import (
    modified_terminal_call,
    modified_terminal_call_hedge,
    modified_terminal_put,
    validate_implicit_relation,
)

# ── Shared baseline parameters ────────────────────────────────────────────────
K = 100.0
LAM = 0.1
VARPHI = 0.06        # 2*VARPHI=0.12 > LAM=0.1, so smoothing region is non-degenerate
N = 1.0

THR_LOW = K - 2 * VARPHI * N    # 99.88
THR_HIGH = K - LAM * N          # 99.90

ATOL = 1e-10


# ── Helper ───────────────────────────────────────────────────────────────────
def call_payoff(x, strike=K, n=N):
    return n * np.maximum(np.asarray(x, dtype=float) - strike, 0.0)


# ── tilde_V value tests ───────────────────────────────────────────────────────
class TestCallTerminalValue:
    def test_call_terminal_zero_below_lower_threshold(self):
        """tilde_V(x) = 0 for all x < K - 2*varphi*N."""
        x_below = np.array([80.0, 95.0, THR_LOW - 1e-6])
        result = modified_terminal_call(x_below, K, N, LAM, VARPHI)
        np.testing.assert_allclose(result, 0.0, atol=ATOL)

    def test_call_terminal_quadratic_in_middle_region(self):
        """
        tilde_V(x) = (x-K+2φN)^2 / (2*(2φ-λ))
        for K-2φN <= x < K-λN.
        """
        two_phi = 2 * VARPHI
        denom = 2 * (two_phi - LAM)
        x_mid = np.array([THR_LOW, THR_LOW + 0.01, (THR_LOW + THR_HIGH) / 2])
        expected = (x_mid - K + two_phi * N) ** 2 / denom
        result = modified_terminal_call(x_mid, K, N, LAM, VARPHI)
        np.testing.assert_allclose(result, expected, atol=ATOL)

    def test_call_terminal_linear_above_upper_threshold(self):
        """
        tilde_V(x) = N*(x-K) + (2φ+λ)*N^2/2
        for x >= K-λN.
        """
        const = (2 * VARPHI + LAM) * N ** 2 / 2
        x_high = np.array([THR_HIGH, 100.0, 110.0, 120.0])
        expected = N * (x_high - K) + const
        result = modified_terminal_call(x_high, K, N, LAM, VARPHI)
        np.testing.assert_allclose(result, expected, atol=ATOL)


# ── tilde_V' (hedge) tests ────────────────────────────────────────────────────
class TestCallTerminalHedge:
    def test_call_hedge_zero_below_lower_threshold(self):
        """tilde_V'(x) = 0 for x < K - 2*varphi*N."""
        x_below = np.array([80.0, 95.0, THR_LOW - 1e-6])
        result = modified_terminal_call_hedge(x_below, K, N, LAM, VARPHI)
        np.testing.assert_allclose(result, 0.0, atol=ATOL)

    def test_call_hedge_linear_in_middle_region(self):
        """
        tilde_V'(x) = (x-K+2φN)/(2φ-λ)
        for K-2φN <= x < K-λN.
        """
        two_phi = 2 * VARPHI
        denom = two_phi - LAM
        x_mid = np.array([THR_LOW, THR_LOW + 0.005, (THR_LOW + THR_HIGH) / 2])
        expected = (x_mid - K + two_phi * N) / denom
        result = modified_terminal_call_hedge(x_mid, K, N, LAM, VARPHI)
        np.testing.assert_allclose(result, expected, atol=ATOL)

    def test_call_hedge_equals_N_above_upper_threshold(self):
        """tilde_V'(x) = N for all x >= K-λN."""
        x_high = np.array([THR_HIGH, 100.0, 105.0, 120.0])
        result = modified_terminal_call_hedge(x_high, K, N, LAM, VARPHI)
        np.testing.assert_allclose(result, N, atol=ATOL)


# ── Implicit relation ─────────────────────────────────────────────────────────
class TestImplicitRelation:
    def test_implicit_relation_satisfied(self):
        """
        tilde_V(x) = V(x + λ*tilde_V'(x)) + (2φ-λ)/2 * (tilde_V'(x))^2
        must hold pointwise with residual < 1e-8.
        """
        x_grid = np.linspace(80, 120, 5000)
        tilde_V = modified_terminal_call(x_grid, K, N, LAM, VARPHI)
        tilde_Vp = modified_terminal_call_hedge(x_grid, K, N, LAM, VARPHI)
        result = validate_implicit_relation(
            x_grid, tilde_V, tilde_Vp, call_payoff, LAM, VARPHI
        )
        assert result["satisfied"], (
            f"Implicit relation not satisfied: max_residual={result['max_residual']:.3e}"
        )

    def test_implicit_relation_satisfied_N2(self):
        """Implicit relation holds also for N=2."""
        N2 = 2.0
        x_grid = np.linspace(80, 120, 5000)
        tilde_V = modified_terminal_call(x_grid, K, N2, LAM, VARPHI)
        tilde_Vp = modified_terminal_call_hedge(x_grid, K, N2, LAM, VARPHI)
        result = validate_implicit_relation(
            x_grid,
            tilde_V,
            tilde_Vp,
            lambda xi: call_payoff(xi, n=N2),
            LAM,
            VARPHI,
        )
        assert result["satisfied"], (
            f"Implicit relation not satisfied for N=2: max_residual={result['max_residual']:.3e}"
        )


# ── Midpoint / degeneracy tests ───────────────────────────────────────────────
class TestMidpointCollapse:
    def test_midpoint_collapse(self):
        """
        When 2*varphi = lambda, the quadratic smoothing region has zero width:
        thr_low = thr_high = K - lambda*N.
        The function must still be callable and return correct values in the
        zero-width limit.
        """
        varphi_mid = LAM / 2.0          # 2*varphi = lambda
        thr_both = K - LAM * N          # both thresholds coincide

        x_above = np.array([thr_both, thr_both + 0.5, 105.0])
        x_below = np.array([thr_both - 1e-6, 95.0])

        result_above = modified_terminal_call(x_above, K, N, LAM, varphi_mid)
        result_below = modified_terminal_call(x_below, K, N, LAM, varphi_mid)

        # Below the single threshold → 0
        np.testing.assert_allclose(result_below, 0.0, atol=ATOL)

        # At/above the threshold: linear region formula applies
        const = (2 * varphi_mid + LAM) * N ** 2 / 2
        expected_above = N * (x_above - K) + const
        np.testing.assert_allclose(result_above, expected_above, atol=ATOL)

    def test_midpoint_smoothing_region_width_is_zero(self):
        """
        Width = (2*varphi - lambda)*N = 0 when 2*varphi = lambda.
        Numerically the two thresholds coincide.
        """
        varphi_mid = LAM / 2.0
        thr_low_mid = K - 2 * varphi_mid * N
        thr_high_mid = K - LAM * N
        assert abs(thr_low_mid - thr_high_mid) < 1e-15


# ── Strike-shift test ─────────────────────────────────────────────────────────
class TestStrikeShift:
    def test_strike_shift(self):
        """
        The upper threshold (where linear region begins, i.e. the effective
        strike in the modified payoff) is at K - lambda*N, not K.
        For x slightly above K-lambda*N the hedge is already saturated at N,
        whereas for x slightly above K the original payoff has slope N.
        """
        effective_strike = K - LAM * N

        # Hedge is already N at the effective strike, not at K
        x_at_eff_strike = np.array([effective_strike + 1e-6])
        hedge_at_eff = modified_terminal_call_hedge(x_at_eff_strike, K, N, LAM, VARPHI)
        np.testing.assert_allclose(hedge_at_eff, N, atol=ATOL)

        # Hedge is strictly less than N for x in (thr_low, effective_strike)
        x_in_smooth = np.array([(THR_LOW + effective_strike) / 2])
        hedge_in_smooth = modified_terminal_call_hedge(x_in_smooth, K, N, LAM, VARPHI)
        assert hedge_in_smooth[0] < N - 1e-8

        # Original payoff has slope 0 at the effective strike (it starts at K)
        eps = 1e-4
        orig_slope = (call_payoff(effective_strike + eps) - call_payoff(effective_strike)) / eps
        assert orig_slope < N - 1e-3  # original payoff not at full slope here


# ── Continuity tests ──────────────────────────────────────────────────────────
class TestContinuityAtThresholds:
    def test_continuity_at_lower_threshold(self):
        """
        tilde_V is continuous at x = K - 2*varphi*N (lower threshold).
        Left limit (below region) = 0; value at threshold (mid formula) = 0.
        """
        eps = 1e-9
        x_left = np.array([THR_LOW - eps])
        x_right = np.array([THR_LOW])

        v_left = modified_terminal_call(x_left, K, N, LAM, VARPHI)
        v_right = modified_terminal_call(x_right, K, N, LAM, VARPHI)

        assert abs(v_left[0] - v_right[0]) < 1e-8

    def test_continuity_at_upper_threshold(self):
        """
        tilde_V is continuous at x = K - lambda*N (upper threshold).
        Both sides evaluate to N^2*(2*varphi - lambda)/2.
        """
        two_phi_minus_lam = 2 * VARPHI - LAM
        expected = N ** 2 * two_phi_minus_lam / 2

        eps = 1e-9
        x_mid_side = np.array([THR_HIGH - eps])
        x_lin_side = np.array([THR_HIGH])

        v_mid = modified_terminal_call(x_mid_side, K, N, LAM, VARPHI)
        v_lin = modified_terminal_call(x_lin_side, K, N, LAM, VARPHI)

        np.testing.assert_allclose(v_mid[0], expected, atol=1e-7)
        np.testing.assert_allclose(v_lin[0], expected, atol=ATOL)
        assert abs(v_mid[0] - v_lin[0]) < 1e-7

    def test_hedge_continuity_at_upper_threshold(self):
        """
        tilde_V' is continuous at the upper threshold:
        mid-region value = (thr_high - K + 2φN)/(2φ-λ) = N*(2φ-λ)/(2φ-λ) = N.
        """
        eps = 1e-9
        x_mid_side = np.array([THR_HIGH - eps])
        x_lin_side = np.array([THR_HIGH])

        hp_mid = modified_terminal_call_hedge(x_mid_side, K, N, LAM, VARPHI)
        hp_lin = modified_terminal_call_hedge(x_lin_side, K, N, LAM, VARPHI)

        np.testing.assert_allclose(hp_mid[0], N, atol=1e-6)
        np.testing.assert_allclose(hp_lin[0], N, atol=ATOL)


# ── Wider smoothing for larger N ──────────────────────────────────────────────
class TestN2WiderSmoothing:
    def test_N2_wider_smoothing(self):
        """
        Smoothing region width = (2*varphi - lambda)*N.
        For N=2 the width is exactly twice that for N=1.
        """
        N1, N2 = 1.0, 2.0
        width1 = (2 * VARPHI - LAM) * N1
        width2 = (2 * VARPHI - LAM) * N2
        assert abs(width2 - 2 * width1) < 1e-15

    def test_N2_thresholds_further_from_K(self):
        """
        For N=2, both thresholds are farther from K than for N=1,
        reflecting a wider price-impact adjustment region.
        """
        N1, N2 = 1.0, 2.0
        thr_low1 = K - 2 * VARPHI * N1
        thr_low2 = K - 2 * VARPHI * N2
        thr_high1 = K - LAM * N1
        thr_high2 = K - LAM * N2

        assert thr_low2 < thr_low1   # lower threshold is further left for N=2
        assert thr_high2 < thr_high1  # upper threshold also shifts left

    def test_N2_smoothing_region_twice_as_wide(self):
        """
        Numerically verify the smoothing region width doubles when N doubles.
        """
        N1, N2 = 1.0, 2.0
        width1 = (K - LAM * N1) - (K - 2 * VARPHI * N1)
        width2 = (K - LAM * N2) - (K - 2 * VARPHI * N2)
        np.testing.assert_allclose(width2, 2.0 * width1, rtol=1e-12)

    def test_N2_tilde_V_positive_on_wider_range(self):
        """
        For N=2, tilde_V is positive at a point where N=1 gives zero.
        Specifically at x = K - 1.5*VARPHI (below N=1 lower threshold but
        above N=2 lower threshold when 2*VARPHI*N2 = 4*VARPHI > 1.5*VARPHI).
        """
        # For N=2: thr_low = K - 2*VARPHI*2 = K - 4*VARPHI
        # For N=1: thr_low = K - 2*VARPHI*1 = K - 2*VARPHI
        # Choose x between the two lower thresholds
        x_test = np.array([K - 3 * VARPHI])  # inside N=2 region, outside N=1 region

        v1 = modified_terminal_call(x_test, K, 1.0, LAM, VARPHI)
        v2 = modified_terminal_call(x_test, K, 2.0, LAM, VARPHI)

        assert v1[0] == pytest.approx(0.0, abs=ATOL)
        assert v2[0] > ATOL
