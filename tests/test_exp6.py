"""
Tests for Experiment 6 — Pathwise Replication Convergence.
"""
import numpy as np
import pytest
from src.pde import NonlinearPDESolver
from src.terminal import modified_terminal_call
from src.simulation import PathwiseReplicator

# ── Shared parameters ─────────────────────────────────────────────────────────
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.20
LAM = 0.10
VARPHI = 0.06
N = 1.0

# Small grid for fast tests
X_MIN, X_MAX, NX, NT = 50.0, 200.0, 100, 100


@pytest.fixture(scope='module')
def pde_replicator():
    """Build PDE solution and replicator once per module."""
    terminal = lambda x: modified_terminal_call(np.asarray(x), K, N, LAM, VARPHI)
    pde = NonlinearPDESolver(T=T, r=r, sigma=sigma, lam=LAM, varphi=VARPHI,
                              x_min=X_MIN, x_max=X_MAX, Nx=NX, Nt=NT,
                              terminal_func=terminal)
    pde.solve()
    V_func = lambda p: N * np.maximum(np.asarray(p, dtype=float) - K, 0.0)
    rep = PathwiseReplicator(
        S0=S0, T=T, r=r, sigma=sigma,
        lam=LAM, varphi=VARPHI,
        K=K, N=N,
        pde_u_func=pde.get_u,
        pde_ux_func=pde.get_ux,
        V_func=V_func,
    )
    return pde, rep


# ── GBM sampling ──────────────────────────────────────────────────────────────
class TestGBMSampling:
    def test_mean_close_to_risk_neutral(self, pde_replicator):
        """E[S_T] ≈ S0 * exp(r * T) under risk-neutral measure."""
        _, rep = pde_replicator
        S = rep.simulate_gbm_exact(n_steps=252, n_paths=2000, seed=0)
        S_T = S[:, -1]
        expected = S0 * np.exp(r * T)
        # Within 3% relative error for 2000 paths
        assert abs(np.mean(S_T) / expected - 1.0) < 0.03, (
            f"Mean S_T = {np.mean(S_T):.2f}, expected ≈ {expected:.2f}"
        )

    def test_initial_value_correct(self, pde_replicator):
        """S[:, 0] == S0 for all paths."""
        _, rep = pde_replicator
        S = rep.simulate_gbm_exact(n_steps=10, n_paths=50, seed=1)
        np.testing.assert_allclose(S[:, 0], S0)

    def test_paths_positive(self, pde_replicator):
        """GBM paths are strictly positive."""
        _, rep = pde_replicator
        S = rep.simulate_gbm_exact(n_steps=100, n_paths=500, seed=7)
        assert np.all(S > 0), "GBM paths contain non-positive values"

    def test_seed_reproducibility(self, pde_replicator):
        """Same seed gives identical paths."""
        _, rep = pde_replicator
        S1 = rep.simulate_gbm_exact(n_steps=20, n_paths=10, seed=99)
        S2 = rep.simulate_gbm_exact(n_steps=20, n_paths=10, seed=99)
        np.testing.assert_array_equal(S1, S2)


# ── Simulation output structure ────────────────────────────────────────────────
class TestSimulationOutputs:
    def test_output_keys(self, pde_replicator):
        """run() returns required keys."""
        _, rep = pde_replicator
        res = rep.run(n_steps=20, n_paths=30, seed=0)
        for key in ('S_T', 'P_T', 'X_T', 'payoff', 'residuals', 'paths', 'delta_0'):
            assert key in res, f"Missing key: {key}"

    def test_shapes(self, pde_replicator):
        """Arrays have correct shapes."""
        _, rep = pde_replicator
        n_paths, n_steps = 50, 30
        res = rep.run(n_steps=n_steps, n_paths=n_paths, seed=5)
        assert res['S_T'].shape == (n_paths,)
        assert res['P_T'].shape == (n_paths,)
        assert res['residuals'].shape == (n_paths,)
        assert res['paths'].shape == (n_paths, n_steps + 1)

    def test_residuals_equal_XT_minus_payoff(self, pde_replicator):
        """residuals = X_T - payoff."""
        _, rep = pde_replicator
        res = rep.run(n_steps=20, n_paths=30, seed=3)
        np.testing.assert_allclose(res['residuals'],
                                   res['X_T'] - res['payoff'], atol=1e-10)

    def test_impacted_price_formula(self, pde_replicator):
        """P_T = S_T + lambda * delta_T."""
        _, rep = pde_replicator
        res = rep.run(n_steps=20, n_paths=30, seed=3)
        P_T_expected = res['S_T'] + LAM * res['delta_T']
        np.testing.assert_allclose(res['P_T'], P_T_expected, atol=1e-10)


# ── Convergence ────────────────────────────────────────────────────────────────
class TestSimulationConvergence:
    def test_rmse_decreases_100_to_1000(self, pde_replicator):
        """RMSE at n=100 steps should exceed RMSE at n=1000 steps."""
        _, rep = pde_replicator
        n_paths = 300

        res_coarse = rep.run(n_steps=100, n_paths=n_paths, seed=42)
        res_fine = rep.run(n_steps=1000, n_paths=n_paths, seed=42)

        rmse_coarse = float(np.sqrt(np.mean(res_coarse['residuals'] ** 2)))
        rmse_fine = float(np.sqrt(np.mean(res_fine['residuals'] ** 2)))

        assert rmse_coarse > rmse_fine, (
            f"RMSE did not decrease: coarse={rmse_coarse:.5f}, fine={rmse_fine:.5f}"
        )

    def test_residuals_bounded(self, pde_replicator):
        """Residuals should be bounded (no blow-up)."""
        _, rep = pde_replicator
        res = rep.run(n_steps=100, n_paths=200, seed=42)
        max_abs = float(np.max(np.abs(res['residuals'])))
        # Residual should not exceed a few option prices
        assert max_abs < 50.0, f"Residuals too large: max|e|={max_abs:.2f}"

    def test_payoff_nonnegative(self, pde_replicator):
        """Call payoff V(P_T) is always non-negative."""
        _, rep = pde_replicator
        res = rep.run(n_steps=100, n_paths=200, seed=42)
        assert np.all(res['payoff'] >= -1e-10), "Negative payoffs found"

    def test_bs_price_function(self):
        """BS price formula gives known value for standard parameters."""
        # BS call: S=K=100, T=1, r=0.05, sigma=0.20 => approx 10.45
        price = PathwiseReplicator.bs_price(100.0, 100.0, 1.0, 0.05, 0.20)
        assert 9.0 < price < 12.0, f"BS price = {price:.4f} outside expected range"
