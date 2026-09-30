# Custom — Context7 found no library equivalent (paper self-impact mechanics)
import numpy as np
from scipy.stats import norm
from typing import Callable


class PathwiseReplicator:
    """
    Discrete-time hedging under permanent price impact (linear model).

    Model mechanics at each rebalancing time t_i:
      - Fundamental price S_{t_i} follows GBM (exact sampling).
      - Target hedge: Delta_i^* = u_x(t_i, S_{t_i})  [from PDE solution].
      - Trade size: delta_i = Delta_i^* - Delta_{i-1}.
      - Impacted execution price: P_i = S_{t_i} + lambda * Delta_i^*.
      - Cash change: -delta_i * (P_i + varphi * delta_i)
          (buy delta_i shares at price P_i, pay execution cost varphi*delta_i^2).
      - At maturity: liquidate all shares; proceeds = Delta_T*(P_T - varphi*Delta_T)
          where P_T = S_T + lambda*Delta_T.

    Payoff is evaluated at the impacted terminal price P_T.
    """

    def __init__(self, S0: float, T: float, r: float, sigma: float,
                 lam: float, varphi: float, K: float, N: float,
                 pde_u_func: Callable, pde_ux_func: Callable,
                 V_func: Callable):
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.lam = lam
        self.varphi = varphi
        self.K = K
        self.N = N
        self.pde_u = pde_u_func
        self.pde_ux = pde_ux_func
        self.V = V_func

    def simulate_gbm_exact(self, n_steps: int, n_paths: int,
                           seed: int = 42) -> np.ndarray:
        """Exact GBM sampling S_{t_{i+1}} = S_{t_i} * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)."""
        rng = np.random.default_rng(seed)
        dt = self.T / n_steps
        Z = rng.standard_normal((n_paths, n_steps))
        S = np.empty((n_paths, n_steps + 1))
        S[:, 0] = self.S0
        log_drift = (self.r - 0.5 * self.sigma ** 2) * dt
        log_vol = self.sigma * np.sqrt(dt)
        for i in range(n_steps):
            S[:, i + 1] = S[:, i] * np.exp(log_drift + log_vol * Z[:, i])
        return S

    def run(self, n_steps: int, n_paths: int, seed: int = 42) -> dict:
        """
        Run pathwise replication over all paths simultaneously.

        Returns
        -------
        dict with keys:
            S_T       : terminal fundamental prices (n_paths,)
            P_T       : terminal impacted prices (n_paths,)
            X_T       : terminal cash value after liquidation (n_paths,)
            payoff    : option payoff V(P_T) (n_paths,)
            residuals : X_T - payoff (n_paths,)
            paths     : full S matrix (n_paths, n_steps+1)
            delta_0   : initial hedge (n_paths,)
            delta_T   : terminal hedge position (n_paths,)
        """
        S = self.simulate_gbm_exact(n_steps, n_paths, seed)
        dt = self.T / n_steps
        t_grid = np.linspace(0.0, self.T, n_steps + 1)

        # ── t=0: establish initial hedge ──────────────────────────────────────
        # Start with the required initial capital X_0^* then buy the initial
        # hedge position.  Without this, the portfolio starts underfunded by X_0^*.
        S0_vec = S[:, 0]
        Delta = self.pde_ux(0.0, S0_vec)          # initial target hedge
        # X_0^* = u(0,S0) + (2*phi - lam)/2 * u_x(0,S0)^2
        X0_star = (self.pde_u(0.0, S0_vec)
                   + (2 * self.varphi - self.lam) / 2 * Delta ** 2)
        P_exec = S0_vec + self.lam * Delta        # execution price
        Cash = X0_star - Delta * (P_exec + self.varphi * Delta)  # fund less initial buy

        # ── Rebalancing: t_1 ... t_{n-1} ─────────────────────────────────────
        for i in range(1, n_steps):
            t = t_grid[i]

            # Accrue risk-free interest on cash
            Cash *= np.exp(self.r * dt)

            S_t = S[:, i]
            Delta_new = self.pde_ux(t, S_t)       # new target hedge
            trade = Delta_new - Delta

            P_obs = S_t + self.lam * Delta_new    # impacted price at new position
            Cash -= trade * (P_obs + self.varphi * trade)
            Delta = Delta_new

        # ── Final period interest ─────────────────────────────────────────────
        Cash *= np.exp(self.r * dt)

        # ── Maturity: liquidate position ──────────────────────────────────────
        S_T = S[:, -1]
        P_T = S_T + self.lam * Delta              # terminal impacted price
        Cash += Delta * (P_T - self.varphi * Delta)  # liquidation proceeds

        X_T = Cash
        payoff = self.V(P_T)
        residuals = X_T - payoff

        return {
            'S_T': S_T,
            'P_T': P_T,
            'X_T': X_T,
            'payoff': payoff,
            'residuals': residuals,
            'paths': S,
            'delta_0': self.pde_ux(0.0, S[:, 0]),
            'delta_T': Delta,
        }

    def bs_delta(self, t: float, S: np.ndarray) -> np.ndarray:
        """Black-Scholes delta for a call with strike K."""
        S = np.asarray(S, dtype=float)
        tau = max(self.T - t, 1e-10)
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2) * tau) / (
            self.sigma * np.sqrt(tau))
        return norm.cdf(d1)

    @staticmethod
    def bs_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes call price."""
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
