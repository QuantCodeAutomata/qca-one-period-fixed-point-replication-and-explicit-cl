# Option Replication with Price Impact

Python implementation of the price-impact option replication framework. Covers one-period
fixed-point theory, multiperiod backward induction on a binomial tree, continuous-time
terminal-condition modification, nonlinear PDE solving, and pathwise simulation validation.

## Repository Structure

```
.
├── data/                   # No empirical data required (synthetic inputs only)
├── src/
│   ├── payoffs.py          # Call, put, digital payoff callables
│   ├── impact.py           # Linear and square-root impact/execution-cost functions
│   ├── fixed_point.py      # Multi-start fixed-point solver (brentq + fsolve)
│   ├── tree.py             # Recombining binomial tree with backward induction
│   ├── terminal.py         # Modified terminal condition ~V for continuous-time PDE
│   ├── pde.py              # Nonlinear pricing PDE solver (finite differences)
│   └── simulation.py       # Pathwise GBM replication simulator
├── exp/
│   ├── exp_1_one_period.py     # One-period fixed-point validation
│   ├── exp_2_multiperiod.py    # Multiperiod backward-induction replication
│   ├── exp_3_terminal.py       # Modified terminal condition analysis
│   ├── exp_4_pde_quadratic.py  # PDE validation via quadratic closed form
│   ├── exp_5_call_pde.py       # Call option PDE price and delta surfaces
│   ├── exp_6_simulation.py     # Pathwise convergence experiment
│   └── exp_7_itm_events.py     # Hedger-induced in-the-money events
├── tests/                  # pytest tests (144 tests, all passing)
├── results/                # Output plots, CSVs, JSONs, and RESULTS.md
└── README.md
```

## Setup

```bash
pip install numpy scipy matplotlib pandas
```

All required libraries are standard scientific Python; no additional packages are needed.

## Running Experiments

```bash
python exp/exp_1_one_period.py
python exp/exp_2_multiperiod.py
python exp/exp_3_terminal.py
python exp/exp_4_pde_quadratic.py
python exp/exp_5_call_pde.py
python exp/exp_6_simulation.py
python exp/exp_7_itm_events.py
```

## Running Tests

```bash
pytest tests/ -q
```

All 144 tests pass in approximately 3 seconds.

## Key Results

| Experiment | Core Finding |
|------------|-------------|
| Exp 1 | Numerical fixed-point roots match analytical piecewise formulas exactly; NPC satisfied |
| Exp 2 | Exact replication at all tree leaves (error ≤ 3×10⁻¹⁴); impacted cost > frictionless |
| Exp 3 | Strike shifts from K to K−λN; interpolation region [K−2φN, K−λN] collapses at midpoint |
| Exp 4 | PDE matches quadratic closed form (sup-norm < 0.03); initial capital > frictionless |
| Exp 5 | Impacted call price > Black–Scholes; midpoint normalization N·λ confirmed numerically |
| Exp 6 | RMSE: 0.687 (n=100) → 0.239 (n=1000) → 0.106 (n=10000): clear convergence |
| Exp 7 | 3/1000 paths have S_T < K < P_T^Δ*: hedger's buying induces self-driven ITM events |

See `results/RESULTS.md` for full metrics and figure descriptions.
