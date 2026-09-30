# Project: Price-Impact Option Replication

## Purpose
Research codebase implementing continuous-time option hedging/replication under
price-impact constraints.  Each experiment explores a different aspect of the
modified terminal conditions derived in the paper.

## Layout
```
src/          — reusable library modules
exp/          — standalone experiment scripts (exp_N_<name>.py)
tests/        — pytest test suite (test_expN.py mirrors exp_N)
results/      — auto-generated PNGs + JSON summaries (not committed)
data/         — input data (empty for now)
```

## Key modules
| File | What it contains |
|------|-----------------|
| `src/terminal.py` | Explicit piecewise modified-terminal formulas for calls and puts; implicit-relation validator |
| `src/payoffs.py` | `call_payoff`, `put_payoff`, `digital_call_payoff` + factory functions `make_call`, `make_put`, `make_digital_call` |
| `src/impact.py` | `linear_impact`, `sqrt_impact`, `linear_exec_cost`, `fig2_exec_cost` |
| `src/fixed_point.py` | `no_price_manipulation_criterion`, `fixed_point_residual`, `find_all_roots`, `select_min_abs_root`, `replication_cost`, `frictionless_price` |
| `src/pde.py` | `QuadraticBenchmark` (ODE-based analytical solution for quadratic payoffs), `NonlinearPDESolver` (backward FD solver for the nonlinear pricing PDE) |
| `src/simulation.py` | `PathwiseReplicator` (discrete-time replication with permanent price impact), `bs_delta`, `bs_price` |

## Running experiments
```bash
python exp/exp_1_one_period.py  # results/exp1_*.png + exp1_results.csv + exp1_summary.json
python exp/exp_3_terminal.py    # results/exp3_*.png + exp3_summary.json
python exp/exp_4_pde_quadratic.py  # results/exp4_*.png + exp4_summary.json  (~1 s)
python exp/exp_5_call_pde.py    # results/exp5_*.png + exp5_summary.json     (~1 s)
python exp/exp_6_simulation.py  # results/exp6_*.png + exp6_summary.json     (~5 s)
python exp/exp_7_itm_events.py  # results/exp7_*.png + exp7_event_frequency.json
```

## Running tests
```bash
pytest tests/test_exp1.py -v    # 57 tests, ~1.4 s
pytest tests/test_exp3.py -v    # 18 tests, ~0.3 s
pytest tests/test_exp4.py -v    # 12 tests, ~0.3 s
pytest tests/test_exp5.py -v    # 13 tests, ~0.5 s
pytest tests/test_exp6.py -v    # 12 tests, ~0.5 s
pytest tests/test_exp7.py -v    # 11 tests, ~0.2 s
pytest tests/ -v                # 123 tests total, ~2.8 s
```

## Exp 1 — One-Period Price-Impact Replication (src/payoffs + impact + fixed_point)

### Baseline parameters
`S0=100, S_u=105, S_d=95, r=0.01, K=100, N=1, λ=0.1, φ=0.05`
`q = (S0*(1+r) - S_d)/(S_u - S_d) = 0.6`

### Fixed-point equation
`F(Δ) = Δ - [V(S_u+I(Δ)) - V(S_d+I(Δ))] / (S_u - S_d) = 0`
Root-finding via sign-change scan + `scipy.optimize.brentq`.

### No-price-manipulation criterion (NPC)
`C(x) = φ(x)*(1+r) - φ(-x) - I(x)`;  valid iff `x·C(x) >= 0`.
For linear φ and I: `C(x) = x*(φ*(2+r) - λ)` → satisfied iff `φ*(2+r) >= λ`.

### Analytical Δ* — linear impact I(x) = λx
| K range | Δ* |
|---------|-----|
| K >= S_u | 0 |
| S_d+λN ≤ K < S_u | N*(S_u-K)/(S_u-S_d-λN) |
| K < S_d+λN | N |

### Analytical Δ* — sqrt impact I(x) = λ·sign(x)·√|x|
`c_* = [λN + √(λ²N²+4|N|(S_u-S_d)(S_u-K))] / [2(S_u-S_d)]`
| K range | Δ* |
|---------|-----|
| K >= S_u | 0 |
| S_d+λ√N ≤ K < S_u | sign(N)·c_*² |
| K < S_d+λ√N | N |

### Replication cost
`X_0* = (E^Q[V(P_T^Δ)] + Δ·C(Δ)) / (1+r)  ≥  frictionless price`

### Digital call non-replicable interval (linear impact)
`J = (S_d, S_d + λN/(S_u-S_d))` — e.g. (95, 95.01) for baseline params.
Inside J the fixed-point equation has no solution; `find_all_roots` correctly
returns `[]` because the residual check `|F(root)| < 1e-6` filters out the
spurious sign-change from the digital payoff discontinuity.

### Key implementation gotcha — tangent zeros at K = S_u
For an OTM call (K = S_u) under sqrt impact, `F(0) = 0` but `F` has the same
sign on both sides of 0, so the sign-change scan misses it.  Fixed by seeding
the grid with `0` explicitly (`np.unique(np.sort(np.append(linspace(...), [0])))`);
the `abs(F[i]) < tol` branch then catches the exact zero.

---

## Exp 3 — Modified Terminal Condition (src/terminal.py)

### Core formulas (call, V(x)=N*(x-K)^+)
Thresholds: `thr_low = K - 2*varphi*N`, `thr_high = K - lambda*N`

| Region | tilde_V(x) | tilde_V'(x) |
|--------|-----------|-------------|
| x < thr_low | 0 | 0 |
| thr_low <= x < thr_high | `(x-K+2φN)^2 / (2*(2φ-λ))` | `(x-K+2φN)/(2φ-λ)` |
| x >= thr_high | `N*(x-K) + (2φ+λ)*N^2/2` | `N` |

### Implicit terminal relation (paper identity)
`tilde_V(x) = V(x + λ*tilde_V'(x)) + (2φ-λ)/2 * (tilde_V'(x))^2`
Validated to machine precision (~7e-15) on a 2000-point grid.

### Degenerate / midpoint case
When `2*varphi = lambda`, the quadratic smoothing region collapses to zero
width; both thresholds coincide at `K - lambda*N`.

### Put construction
`_general_modified_terminal(..., payoff_type='put')` uses put-call symmetry:
`tilde_V_put(x) = tilde_V_call(2K - x)` evaluated with mirrored thresholds
at `K + lambda*N` and `K + 2*varphi*N`.

---

## Exp 4 — PDE Quadratic Benchmark (src/pde.QuadraticBenchmark + NonlinearPDESolver)

### Parameters
`S0=100, T=1, r=0.05, sigma=0.20, lam=0.1, varphi=0.06, alpha=1e-4, beta=0, gamma=0, N=1`
Constraint: `2*varphi >= lambda` → `0.12 >= 0.10` ✓

### QuadraticBenchmark
Ansatz `u(t,x) = a(t)*x^2 + b(t)*x + c(t)` reduces the nonlinear PDE to an ODE system:
```
da/dtau = (r+sigma^2)*a + 2*a^2*(r*lam + sigma^2*(2phi-lam))
db/dtau = -(r - 2*r*lam*a)*b
dc/dtau = -(r*c - (r*lam/2)*b^2)
```
Integrated via `scipy.integrate.solve_ivp` with `tau=T-t` convention (tau=0 is terminal).
Terminal coefficients `a_T, b_T, c_T` solved self-consistently via `scipy.optimize.fsolve`.

### NonlinearPDESolver
- Backward Euler time-stepping on uniform `(t,x)` grid
- Picard iteration for nonlinear terms (max 20 iters, tol=1e-8)
- Left BC: linear extrapolation `u[0] = 2*u[1] - u[2]`
- Right BC: linear extrapolation `u[-1] = 2*u[-2] - u[-3]`

### Critical domain-width issue
The linear extrapolation BC introduces errors that do NOT decay with grid
refinement — they saturate at ~4% for domain `[60,160]` regardless of Nx/Nt.
Fix: use a **wide** domain `[20, 200]` (or wider) so BC errors stay far from
the evaluation region `[80,140]`. With domain `[20,200]`, Nx=300, Nt=200:
- sup-norm error < 0.006 (threshold 0.01) ✓
- RMSE < 0.002 (threshold 0.005) ✓
All PDE accuracy tests use `x_min=20, x_max=200, Nx=300, Nt=200`.

### `initial_capital()` methods
Both `QuadraticBenchmark` and `NonlinearPDESolver` expose:
`X_0^* = u(0,S0) + (2*varphi - lam)/2 * u_x(0,S0)^2`
Always call `[0]` to index into the returned array before wrapping in `float()`.

---

## Exp 5 — Call PDE Pricing (src/terminal.modified_terminal_call + NonlinearPDESolver)

### Parameters
`K=100, T=1, r=0.05, sigma=0.20, varphi=0.06`; multiple `(N, lam)` combos

### Key findings
- PDE price at S0=K exceeds BS price for all `lam > 0` (impact premium)
- Delta `u_x` is monotone increasing in S, bounded in `[0, N]`
- Midpoint normalisation: same `N*lam` product → same normalised price curve
- Call option delta approaches 0 (deep OTM) and N (deep ITM)
- `X_0^* = u(0,S0) + (2*varphi-lam)/2*u_x(0,S0)^2` always exceeds `u(0,S0)`

---

## Exp 6 — Simulation Convergence (src/simulation.PathwiseReplicator)

### Parameters
`S0=100, K=100, T=1, r=0.05, sigma=0.20, lam=0.1, varphi=0.06, N=1`

### run() cash dynamics
Cash starts at 0 (no initial capital deposited). Residuals measure
`X_T - V(P_T)` where V is the call payoff at impacted price `P_T = S_T + lam*Delta_T`.
Mean residual ≈ -X_0^* ≈ -10.5 (the option premium not deposited).
Convergence test checks RMSE decreases from n=100 to n=1000 steps (variance shrinks).

### Impacted price formula
`P_T = S_T + lam * Delta_T`  (permanent impact, linear, at maturity)
Payoff is `N * max(P_T - K, 0)` — settled at the *impacted* price.

---

## Exp 7 — ITM Event Analysis

### ITM gap event
Occurs when `S_T < K <= P_T^Delta`:
- Fundamental price is OTM but impacted price is ATM or ITM
- Width of risk gap: `lam * N` (e.g. 0.1 for baseline)
- Occurs when `S_T ∈ (K - lam*N, K)` = `(99.9, 100)` for baseline

### Event frequency
Roughly `P(S_T ∈ (K-lam*N, K)) ≈ N(d2) * lam*N / (S_T * sigma * sqrt(T))`
For baseline: theoretical frequency ≈ 1.5%, simulated ≈ 1-3%

---

## Notes / gotchas
- `file_editor` view command fails on files containing Unicode box-drawing
  characters (U+2500 range); use `cat -n` via terminal instead.
- Experiment scripts use `matplotlib.use("Agg")` and `sys.path.insert` so
  they can be run from any directory without package installation.
- The `results/` directory is created automatically by each experiment script.
- The `NonlinearPDESolver` domain must be significantly wider than the
  evaluation region to prevent BC contamination (rule of thumb: extend by
  at least ±40% of S0 beyond the target range).
- pytest class-scoped fixtures should use `@classmethod` + `@pytest.fixture`
  (both decorators) to avoid `PytestRemovedIn10Warning` from pytest >= 9.
- `get_u()` and `get_ux()` return arrays; always index with `[0]` before
  casting to `float()` to avoid deprecation errors.
