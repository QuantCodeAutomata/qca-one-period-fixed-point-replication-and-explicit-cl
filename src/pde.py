# Custom — Context7 found no library equivalent (paper-specific nonlinear PDE)
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import fsolve
from typing import Callable, Tuple


class QuadraticBenchmark:
    """
    Analytical solution for quadratic payoff V(x) = N*(alpha*x^2 + beta*x + gamma).

    Ansatz: u(t,x) = a(t)*x^2 + b(t)*x + c(t).  Substituting into the PDE
        0 = -r*u + u_t + r*x*u_x + (sigma^2*x^2/2)*u_xx
            + (r*lambda/2)*u_x^2 + ((2*varphi-lambda)/2)*sigma^2*x^2*u_xx^2
    yields the decoupled ODE system (in forward calendar time t):
        a'(t) = -(r + sigma^2)*a - 2*a^2*(r*lambda + (2*varphi-lambda)*sigma^2)
        b'(t) = -2*r*lambda*a*b
        c'(t) =  r*c - (r*lambda/2)*b^2
    Terminal conditions a(T), b(T), c(T) come from the self-consistency equation
        u(T,x) = V(x + lambda*u_x(T,x)) + (2*varphi-lambda)/2*(u_x(T,x))^2.
    """

    def __init__(self, S0: float, T: float, r: float, sigma: float,
                 lam: float, varphi: float, alpha: float, beta: float,
                 gamma: float, N: float = 1.0):
        assert 2 * varphi >= lam, "Need 2*varphi >= lambda for admissibility"
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.lam = lam
        self.varphi = varphi
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.N = N
        self._a_func = None
        self._b_func = None
        self._c_func = None

    def _terminal_coefficients(self) -> Tuple[float, float, float]:
        """
        Solve for a(T), b(T), c(T) from the self-consistency equation.

        x^2 coefficient: a_T = N*alpha*(1 + 2*lam*a_T)^2 + 2*phi0*a_T^2
          => (4*N*alpha*lam^2 + 2*phi0)*a_T^2 + (4*N*alpha*lam - 1)*a_T + N*alpha = 0
          Discriminant: D = 1 - 16*N*alpha*varphi  (simplified)
          Physical root: the small one near N*alpha as lam,phi->0.

        x^1 coefficient (linear in b_T given a_T):
          b_T*(1 - 2*N*alpha*lam*(1 + 2*lam*a_T) - 2*phi0*a_T) = N*beta*(1 + 2*lam*a_T)

        x^0 coefficient (explicit given a_T, b_T):
          c_T = N*alpha*(lam*b_T)^2 + N*beta*lam*b_T + N*gamma + phi0/2*b_T^2
        """
        lam = self.lam
        phi0 = 2 * self.varphi - lam  # 2*varphi - lambda >= 0
        N = self.N
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        # --- a_T: solve quadratic ---
        # A2*a_T^2 + A1*a_T + A0 = 0
        A2 = 4 * N * alpha * lam ** 2 + 2 * phi0
        A1 = 4 * N * alpha * lam - 1
        A0 = N * alpha

        if A2 < 1e-14:
            # phi0 and lam both essentially zero: a_T = N*alpha
            a_T = N * alpha
        else:
            disc = A1 ** 2 - 4 * A2 * A0
            if disc < 0:
                disc = 0.0
            sqrt_disc = np.sqrt(disc)
            roots = [(-A1 - sqrt_disc) / (2 * A2),
                     (-A1 + sqrt_disc) / (2 * A2)]
            # Physical root: closest to N*alpha (the frictionless limit)
            a_T = min(roots, key=lambda v: abs(v - N * alpha))

        # --- b_T: linear equation ---
        xi_factor = 1.0 + 2 * lam * a_T
        coeff_b = 1.0 - 2 * N * alpha * lam * xi_factor - 2 * phi0 * a_T
        rhs_b = N * beta * xi_factor
        if abs(coeff_b) < 1e-14:
            b_T = rhs_b  # fallback (degenerate)
        else:
            b_T = rhs_b / coeff_b

        # --- c_T: explicit ---
        c_T = (N * alpha * (lam * b_T) ** 2
               + N * beta * lam * b_T
               + N * gamma
               + phi0 / 2.0 * b_T ** 2)

        return float(a_T), float(b_T), float(c_T)

    def solve_ode(self) -> None:
        """
        Integrate the ODE system backward from t=T to t=0.
        Uses tau = T - t as the independent variable (forward integration).
        """
        a_T, b_T, c_T = self._terminal_coefficients()
        r = self.r
        sigma = self.sigma
        lam = self.lam
        phi0 = 2 * self.varphi - lam  # >= 0

        def ode(tau, y):
            a, b, c = y
            # In tau = T-t convention:  dy/dtau = -dy/dt
            da = (r + sigma ** 2) * a + 2 * a ** 2 * (r * lam + phi0 * sigma ** 2)
            db = 2 * r * lam * a * b
            dc = -r * c + (r * lam / 2) * b ** 2
            return [da, db, dc]

        tau_eval = np.linspace(0.0, self.T, 1000)
        sol = solve_ivp(ode, [0.0, self.T], [a_T, b_T, c_T],
                        t_eval=tau_eval, method='RK45',
                        rtol=1e-10, atol=1e-12)

        # Convert back to calendar time t = T - tau
        t_vals = self.T - sol.t        # decreasing from T to 0
        idx = np.argsort(t_vals)       # sort ascending in t
        self.t_grid = t_vals[idx]
        self.a_vals = sol.y[0][idx]
        self.b_vals = sol.y[1][idx]
        self.c_vals = sol.y[2][idx]

        self._a_func = interp1d(self.t_grid, self.a_vals, kind='cubic',
                                fill_value='extrapolate')
        self._b_func = interp1d(self.t_grid, self.b_vals, kind='cubic',
                                fill_value='extrapolate')
        self._c_func = interp1d(self.t_grid, self.c_vals, kind='cubic',
                                fill_value='extrapolate')

    def u(self, t: float, x: np.ndarray) -> np.ndarray:
        """Analytical price u(t,x) = a(t)*x^2 + b(t)*x + c(t)."""
        a = float(self._a_func(t))
        b = float(self._b_func(t))
        c = float(self._c_func(t))
        x = np.asarray(x, dtype=float)
        return a * x ** 2 + b * x + c

    def u_x(self, t: float, x: np.ndarray) -> np.ndarray:
        """Analytical delta u_x(t,x) = 2*a(t)*x + b(t)."""
        a = float(self._a_func(t))
        b = float(self._b_func(t))
        x = np.asarray(x, dtype=float)
        return 2 * a * x + b

    def initial_capital(self) -> float:
        """X_0^* = u(0,S0) + (2*varphi - lambda)/2 * u_x(0,S0)^2."""
        u0 = float(self.u(0.0, np.array([self.S0]))[0])
        ux0 = float(self.u_x(0.0, np.array([self.S0]))[0])
        return u0 + (2 * self.varphi - self.lam) / 2.0 * ux0 ** 2


class NonlinearPDESolver:
    """
    Finite-difference solver for the nonlinear pricing PDE (backward in time):
        0 = -r*u + u_t + r*x*u_x + (sigma^2*x^2/2)*u_xx
            + (r*lambda/2)*u_x^2 + ((2*varphi-lambda)/2)*sigma^2*x^2*u_xx^2

    Solved in forward tau = T-t direction with backward-Euler + Picard iteration.
    Grid: uniform in x. BCs: Dirichlet u=0 on left, linear extrapolation on right.
    """

    def __init__(self, T: float, r: float, sigma: float, lam: float, varphi: float,
                 x_min: float, x_max: float, Nx: int, Nt: int,
                 terminal_func: Callable, left_bc: str = 'extrapolate'):
        """
        Parameters
        ----------
        left_bc : 'extrapolate' (linear extrapolation, default) or
                  'dirichlet_zero' (pin left boundary to 0, good for call OTM side).
        """
        self.T = T
        self.r = r
        self.sigma = sigma
        self.lam = lam
        self.varphi = varphi
        self.x_min = x_min
        self.x_max = x_max
        self.Nx = Nx
        self.Nt = Nt
        self.terminal_func = terminal_func
        self.left_bc = left_bc

        self.x_grid = np.linspace(x_min, x_max, Nx)
        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dtau = T / Nt                          # forward tau step
        self.t_grid = np.linspace(0.0, T, Nt + 1)  # calendar time grid

        # Storage: u_grid[t_idx, x_idx], t_idx=0 => t=0, t_idx=Nt => t=T
        self.u_grid = np.zeros((Nt + 1, Nx))
        self._built = False

    def _nonlinear_terms(self, u: np.ndarray) -> np.ndarray:
        """Compute NL(u) = (r*lam/2)*u_x^2 + ((2phi-lam)/2)*sigma^2*x^2*u_xx^2."""
        dx = self.dx
        x = self.x_grid
        Nx = self.Nx
        phi0 = 2 * self.varphi - self.lam

        ux = np.empty(Nx)
        uxx = np.empty(Nx)

        # Central differences interior
        ux[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        uxx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
        # One-sided at boundaries
        ux[0] = (u[1] - u[0]) / dx
        ux[-1] = (u[-1] - u[-2]) / dx
        uxx[0] = uxx[1]
        uxx[-1] = uxx[-2]

        nl_ux2 = (self.r * self.lam / 2.0) * ux ** 2
        nl_uxx2 = (phi0 / 2.0) * self.sigma ** 2 * x ** 2 * uxx ** 2
        return nl_ux2 + nl_uxx2

    def _build_matrix_rhs(self, u_old: np.ndarray,
                          nl: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build tridiagonal coefficients for the interior nodes (indices 1 .. Nx-2).

        Left BC options:
          'dirichlet_zero' : u[0] = 0 (call OTM side).
          'extrapolate'    : u[0] = 2*u[1] - u[2] (zero curvature, default).
        Right BC: linear extrapolation u[Nx-1] = 2*u[Nx-2] - u[Nx-3].

        The system A * u_int = rhs is (Nx-2) x (Nx-2).
        """
        dx = self.dx
        dtau = self.dtau
        r = self.r
        sigma = self.sigma
        x = self.x_grid
        Ni = self.Nx - 2  # number of interior nodes

        xi = x[1:-1]
        alpha_i = sigma ** 2 * xi ** 2 / (2.0 * dx ** 2) * dtau
        mu_i = r * xi / (2.0 * dx) * dtau

        sub = np.empty(Ni)
        diag = np.empty(Ni)
        sup = np.empty(Ni)
        rhs = np.empty(Ni)

        sub[:] = -(alpha_i - mu_i)
        diag[:] = 1.0 + r * dtau + 2 * alpha_i
        sup[:] = -(alpha_i + mu_i)
        # RHS of (I - dtau*L)[u_new] = u_old + dtau*NL
        rhs[:] = u_old[1:-1] + dtau * nl[1:-1]

        if self.left_bc == 'extrapolate':
            # u[0] = 2*u[1] - u[2]  =>  sub[0]*u[0] = sub[0]*(2*u[1]-u[2])
            diag[0] = diag[0] + 2 * sub[0]
            sup[0] = sup[0] - sub[0]
            sub[0] = 0.0
        else:
            # Dirichlet: u[0] = 0, contribution to rhs[0] is zero (already 0)
            pass

        # Right BC: u[Nx-1] = 2*u[Nx-2] - u[Nx-3]
        sub[-1] = sub[-1] - sup[-1]
        diag[-1] = diag[-1] + 2 * sup[-1]
        sup[-1] = 0.0

        return sub, diag, sup, rhs

    @staticmethod
    def _solve_tridiagonal(sub: np.ndarray, diag: np.ndarray,
                           sup: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Thomas algorithm for tridiagonal system."""
        n = len(diag)
        c = sup.copy()
        d = rhs.copy()
        b = diag.copy()
        a = sub.copy()

        c[0] /= b[0]
        d[0] /= b[0]
        for i in range(1, n):
            m = a[i] / (b[i] - a[i] * c[i - 1]) if i > 0 else 0.0
            # Safer:
            denom = b[i] - a[i] * c[i - 1]
            if abs(denom) < 1e-300:
                denom = 1e-300
            c[i] = c[i] / denom
            d[i] = (d[i] - a[i] * d[i - 1]) / denom

        x = np.empty(n)
        x[-1] = d[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]
        return x

    def solve(self, picard_max_iter: int = 20, picard_tol: float = 1e-8) -> None:
        """Solve PDE backward using forward tau integration (backward Euler + Picard)."""
        Nx = self.Nx

        # tau=0 corresponds to t=T (terminal condition)
        self.u_grid[self.Nt, :] = self.terminal_func(self.x_grid)

        # March from tau=0 to tau=T (i.e., from t=T to t=0)
        for n in range(self.Nt - 1, -1, -1):
            # u_old is at tau_{Nt-n}, u_new will be at tau_{Nt-n} + dtau
            # In calendar time: u_old is at t = n+1, u_new is at t = n
            u_old = self.u_grid[n + 1, :].copy()
            u_curr = u_old.copy()

            for _ in range(picard_max_iter):
                u_prev = u_curr.copy()
                nl = self._nonlinear_terms(u_curr)
                sub, diag, sup, rhs = self._build_matrix_rhs(u_old, nl)
                u_int = self._solve_tridiagonal(sub, diag, sup, rhs)

                # Full solution: reconstruct boundary nodes
                u_new = np.empty(Nx)
                u_new[1:-1] = u_int
                u_new[-1] = 2 * u_new[-2] - u_new[-3]   # linear extrapolation right
                if self.left_bc == 'extrapolate':
                    u_new[0] = 2 * u_new[1] - u_new[2]  # linear extrapolation left
                else:
                    u_new[0] = 0.0                       # Dirichlet zero

                u_curr = u_new
                if np.isnan(u_curr).any():
                    u_curr = u_prev  # reject diverged iterate
                    break
                if np.max(np.abs(u_curr - u_prev)) < picard_tol:
                    break

            self.u_grid[n, :] = u_curr

        self._built = True
        self._build_interpolants()

    def _build_interpolants(self) -> None:
        """Build 2-D interpolants for u(t,x) and u_x(t,x)."""
        self._u_interp = RegularGridInterpolator(
            (self.t_grid, self.x_grid), self.u_grid,
            method='linear', bounds_error=False, fill_value=None)

        # u_x by numpy.gradient along x-axis
        ux_grid = np.gradient(self.u_grid, self.x_grid, axis=1)
        self._ux_interp = RegularGridInterpolator(
            (self.t_grid, self.x_grid), ux_grid,
            method='linear', bounds_error=False, fill_value=None)

    def _query(self, interp: RegularGridInterpolator,
               t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        pts = np.column_stack([np.full(x.shape, t), x])
        return interp(pts)

    def get_u(self, t: float, x: np.ndarray) -> np.ndarray:
        """Interpolate u(t, x)."""
        return self._query(self._u_interp, t, x)

    def get_ux(self, t: float, x: np.ndarray) -> np.ndarray:
        """Interpolate u_x(t, x) = Delta*(t, x)."""
        return self._query(self._ux_interp, t, x)

    def initial_capital(self, S0: float) -> float:
        """X_0^* = u(0, S0) + (2*varphi - lambda)/2 * u_x(0, S0)^2."""
        u0 = float(self.get_u(0.0, np.array([S0]))[0])
        ux0 = float(self.get_ux(0.0, np.array([S0]))[0])
        return u0 + (2 * self.varphi - self.lam) / 2.0 * ux0 ** 2
