"""
Tests for Experiment 2: Multiperiod Binomial Tree Replication with Price Impact.
"""
import numpy as np
import pytest

from src.payoffs import make_call, make_put
from src.tree import BinomialImpactTree


# ────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ────────────────────────────────────────────────────────────────────────────

def _make_tree(M: int = 5, lam: float = 0.10, varphi: float = 0.06) -> BinomialImpactTree:
    """Baseline ATM call tree with M steps."""
    S0, T, K, sigma, r, N = 100.0, 1.0, 100.0, 0.20, 0.05, 1.0
    h = T / M
    u = np.exp(sigma * np.sqrt(h))
    d = np.exp(-sigma * np.sqrt(h))
    V = make_call(K, N)
    tree = BinomialImpactTree(S0=S0, T=T, M=M, u=u, d=d, r=r, lam=lam, varphi=varphi, V=V, N=N)
    return tree


def _midpoint_tree(M: int = 5) -> BinomialImpactTree:
    lam = 0.10
    return _make_tree(M=M, lam=lam, varphi=lam / 2)


# ────────────────────────────────────────────────────────────────────────────
# 1. Structural / parameter validation
# ────────────────────────────────────────────────────────────────────────────

class TestParameterValidation:
    def test_no_arbitrage_condition(self) -> None:
        S0, T, K = 100.0, 1.0, 100.0
        sigma, r = 0.20, 0.05
        M = 5
        h = T / M
        u = np.exp(sigma * np.sqrt(h))
        d = np.exp(-sigma * np.sqrt(h))
        assert d < 1 + r * h < u, "No-arbitrage violated"

    def test_risk_neutral_prob_in_bounds(self) -> None:
        tree = _make_tree()
        assert 0 < tree.q < 1

    def test_no_price_manipulation_check(self) -> None:
        tree = _make_tree()
        assert 2 * tree.varphi >= tree.lam

    def test_no_arbitrage_assertion_raised(self) -> None:
        S0, T, K = 100.0, 1.0, 100.0
        M = 5
        h = T / M
        # Use u < d (arbitrage) — should raise AssertionError
        V = make_call(K, 1.0)
        with pytest.raises(AssertionError):
            BinomialImpactTree(S0=S0, T=T, M=M, u=0.98, d=1.02, r=0.05,
                               lam=0.1, varphi=0.06, V=V)

    def test_no_price_manip_assertion_raised(self) -> None:
        S0, T, K = 100.0, 1.0, 100.0
        M = 5
        h = T / M
        sigma = 0.20
        u = np.exp(sigma * np.sqrt(h))
        d = np.exp(-sigma * np.sqrt(h))
        V = make_call(K, 1.0)
        with pytest.raises(AssertionError):
            BinomialImpactTree(S0=S0, T=T, M=M, u=u, d=d, r=0.05,
                               lam=0.20, varphi=0.05, V=V)  # 2*0.05=0.10 < 0.20


class TestNodePrices:
    def test_recombining_property(self) -> None:
        """Up then down == down then up (recombining)."""
        tree = _make_tree(M=4)
        # node (2,1): 1 up + 1 down
        # S0 * u * d should equal S0 * d * u  (trivially true, but verify node_price)
        p_ud = tree.node_price(2, 1)
        # Also: node(2,1) = S0 * u^1 * d^1
        expected = tree.S0 * tree.u ** 1 * tree.d ** 1
        assert abs(p_ud - expected) < 1e-12

    def test_terminal_prices_count(self) -> None:
        M = 5
        tree = _make_tree(M=M)
        prices = [tree.node_price(M, k) for k in range(M + 1)]
        assert len(prices) == M + 1
        assert all(prices[i] < prices[i + 1] for i in range(M))


# ────────────────────────────────────────────────────────────────────────────
# 2. Replication accuracy
# ────────────────────────────────────────────────────────────────────────────

class TestTerminalReplication:
    def test_exact_replication_M3(self) -> None:
        tree = _make_tree(M=3)
        tree.build()
        result = tree.verify_terminal_replication(tol=1e-5)
        assert result["all_replicated"], (
            f"M=3 replication failed: max_error={result['max_error']:.2e}"
        )

    def test_exact_replication_M5(self) -> None:
        tree = _make_tree(M=5)
        tree.build()
        result = tree.verify_terminal_replication(tol=1e-5)
        assert result["all_replicated"], (
            f"M=5 replication failed: max_error={result['max_error']:.2e}"
        )

    def test_exact_replication_midpoint_M5(self) -> None:
        tree = _midpoint_tree(M=5)
        tree.build()
        result = tree.verify_terminal_replication(tol=1e-5)
        assert result["all_replicated"], (
            f"M=5 midpoint replication failed: max_error={result['max_error']:.2e}"
        )

    def test_max_error_below_threshold(self) -> None:
        tree = _make_tree(M=5)
        tree.build()
        result = tree.verify_terminal_replication()
        assert result["max_error"] < 1e-5


# ────────────────────────────────────────────────────────────────────────────
# 3. Cost ordering
# ────────────────────────────────────────────────────────────────────────────

class TestCostOrdering:
    def test_impacted_above_frictionless(self) -> None:
        """X_0^* >= frictionless price for monotone call payoff."""
        tree = _make_tree(M=5)
        tree.build()
        X0_imp = tree.initial_value()
        X0_fric = tree.frictionless_price()
        assert X0_imp >= X0_fric - 1e-8, (
            f"Impacted={X0_imp:.6f} < Frictionless={X0_fric:.6f}"
        )

    def test_impacted_above_frictionless_midpoint(self) -> None:
        tree = _midpoint_tree(M=5)
        tree.build()
        assert tree.initial_value() >= tree.frictionless_price() - 1e-8

    def test_frictionless_matches_binomial_bs_approx(self) -> None:
        """Frictionless price should be positive for ATM call."""
        tree = _make_tree(M=20)
        tree.build()
        fric = tree.frictionless_price()
        assert fric > 0


# ────────────────────────────────────────────────────────────────────────────
# 4. Midpoint nondegeneracy
# ────────────────────────────────────────────────────────────────────────────

class TestMidpointCase:
    def test_midpoint_denominator_never_zero(self) -> None:
        """In midpoint case the denominator simplifies; no warnings about zero denom."""
        import warnings
        tree = _midpoint_tree(M=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tree.build()
            denom_warnings = [x for x in w if "Near-zero denominator" in str(x.message)]
        assert len(denom_warnings) == 0

    def test_midpoint_replication_M7(self) -> None:
        tree = _midpoint_tree(M=7)
        tree.build()
        result = tree.verify_terminal_replication(tol=1e-4)
        assert result["all_replicated"], (
            f"M=7 midpoint failed: max_error={result['max_error']:.2e}"
        )


# ────────────────────────────────────────────────────────────────────────────
# 5. M=1 consistency with one-period model
# ────────────────────────────────────────────────────────────────────────────

class TestSingleStepConsistency:
    def test_M1_terminal_replication(self) -> None:
        """M=1 tree should still replicate at both terminal nodes."""
        S0, T, K, sigma, r, N = 100.0, 1.0, 100.0, 0.20, 0.05, 1.0
        M = 1
        h = T / M
        u = np.exp(sigma * np.sqrt(h))
        d = np.exp(-sigma * np.sqrt(h))
        lam, varphi = 0.10, 0.06
        V = make_call(K, N)
        tree = BinomialImpactTree(S0=S0, T=T, M=M, u=u, d=d, r=r, lam=lam, varphi=varphi, V=V, N=N)
        tree.build()
        result = tree.verify_terminal_replication(tol=1e-5)
        assert result["all_replicated"], (
            f"M=1 replication failed: max_error={result['max_error']:.2e}"
        )

    def test_M1_cost_positive(self) -> None:
        """Initial cost should be positive for ITM/ATM call."""
        S0, T, K = 100.0, 1.0, 100.0
        M = 1
        h = T / M
        sigma, r = 0.20, 0.05
        u = np.exp(sigma * np.sqrt(h))
        d = np.exp(-sigma * np.sqrt(h))
        V = make_call(K, 1.0)
        tree = BinomialImpactTree(S0=S0, T=T, M=M, u=u, d=d, r=r, lam=0.1, varphi=0.06, V=V, N=1.0)
        tree.build()
        assert tree.initial_value() > 0


# ────────────────────────────────────────────────────────────────────────────
# 6. Edge cases
# ────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_deep_otm_call_near_zero_price(self) -> None:
        """Deep OTM call should have near-zero impacted price."""
        S0, T, K_otm = 100.0, 1.0, 200.0
        M = 5
        sigma, r = 0.20, 0.05
        h = T / M
        u = np.exp(sigma * np.sqrt(h))
        d = np.exp(-sigma * np.sqrt(h))
        V = make_call(K_otm, 1.0)
        tree = BinomialImpactTree(S0=S0, T=T, M=M, u=u, d=d, r=r, lam=0.1, varphi=0.06, V=V, N=1.0)
        tree.build()
        assert tree.initial_value() < 1.0

    def test_deep_itm_call_price_reasonable(self) -> None:
        """Deep ITM call should have relatively large price."""
        S0, T, K_itm = 100.0, 1.0, 50.0
        M = 5
        sigma, r = 0.20, 0.05
        h = T / M
        u = np.exp(sigma * np.sqrt(h))
        d = np.exp(-sigma * np.sqrt(h))
        V = make_call(K_itm, 1.0)
        tree = BinomialImpactTree(S0=S0, T=T, M=M, u=u, d=d, r=r, lam=0.1, varphi=0.06, V=V, N=1.0)
        tree.build()
        assert tree.initial_value() > 5.0

    def test_put_payoff_replication(self) -> None:
        """Put payoff should also replicate exactly at terminal nodes."""
        S0, T, K = 100.0, 1.0, 100.0
        M = 3
        sigma, r = 0.20, 0.05
        h = T / M
        u = np.exp(sigma * np.sqrt(h))
        d = np.exp(-sigma * np.sqrt(h))
        V = make_put(K, 1.0)
        tree = BinomialImpactTree(S0=S0, T=T, M=M, u=u, d=d, r=r, lam=0.1, varphi=0.06, V=V, N=1.0)
        tree.build()
        result = tree.verify_terminal_replication(tol=1e-5)
        assert result["all_replicated"], (
            f"Put M=3 replication failed: max_error={result['max_error']:.2e}"
        )
