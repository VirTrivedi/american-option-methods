# american-option-methods

Comparison of binomial-tree, Crank-Nicolson PDE and Longstaff-Schwartz Monte Carlo pricers
for American options (continuous dividend yield `q`). Rigor and reproducibility over speed.

**Status: Step 1 complete** (CRR tree, Black-Scholes, exercise boundary, reference price).
PDE and LSM are not implemented; `price(..., method="pde"|"lsm")` raises `NotImplementedError`.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/python scripts/step1_convergence.py   # writes results/figures/step1_*.png
```

## Layout

| File | Purpose |
|---|---|
| `src/aopt/params.py` | frozen, validated `OptionParams(S, K, T, r, q, sigma, option_type)` |
| `src/aopt/bs.py` | vectorized Black-Scholes price and delta with yield `q` |
| `src/aopt/tree.py` | CRR tree (O(N) memory), `exercise_boundary` |
| `src/aopt/reference.py` | averaged + Richardson-extrapolated reference price with error estimate |
| `src/aopt/interface.py` | `price(params, method, **kwargs)` dispatcher |

## Methodology notes

- **Tree:** CRR, `u = exp(sigma*sqrt(dt))`, `p = (exp((r-q)dt) - d)/(u - d)`; raises `ValueError` if `p` is outside [0, 1].
- **Boundary:** at each step, S*(t) is the extreme node where exercise strictly beats continuation
  (puts: largest S; calls: smallest S); NaN if no node qualifies. It is a step function with
  resolution ~ `S*sigma*sqrt(dt)`, so tests use node-spacing tolerances.
- **Reference price:** `A(N) = (V_N + V_{N+1})/2` damps CRR oscillation; `R(N) = 2A(2N) - A(N)`
  removes the leading 1/N term. Reported price is `R` from `A(4N), A(2N)`; the error estimate is
  the gap between the two successive Richardson values (empirical, not a bound).
- **Perpetual put check:** the closed-form boundary for q = 0 is `S* = K*g/(g+1)`, `g = 2r/sigma^2`.
  This equals `K*beta/(beta-1)` with the *negative* root `beta = -2r/sigma^2`; using `beta = +2r/sigma^2`
  in that formula gives a value above K, which is wrong.

## Step 1 results (S=K=100, T=1, r=5%, q=0, sigma=20%, put)

- Reference American put: **6.090371 +/- 1.9e-6** (literature: ~6.0904).
- Measured log-log convergence slope vs N (N = 20..1600): raw CRR **-1.01**, averaged **-0.92**,
  Richardson **-1.36**. Raw CRR is first order and oscillates; averaging removes the oscillation
  (about 2-3x smaller error at equal N, in the plot); Richardson is 1-2 orders of magnitude more accurate
  but irregular, not a clean second-order rate, because of the early-exercise kink.
- European tree error vs Black-Scholes: ~1.94/N.

![convergence](results/figures/step1_convergence.png)
![boundary](results/figures/step1_boundary.png)
