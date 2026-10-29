# american-option-methods

Comparison of binomial-tree, Crank-Nicolson PDE and Longstaff-Schwartz Monte Carlo pricers
for American options (continuous dividend yield `q`). Rigor and reproducibility over speed.

**Status: Steps 1-2 complete** (CRR tree, Black-Scholes, reference price; Crank-Nicolson PDE with PSOR/penalty).
LSM is not implemented; `price(..., method="lsm")` raises `NotImplementedError`.

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
| `src/aopt/pde.py` | Crank-Nicolson solver (log-price grid), PSOR and penalty American handling, Rannacher startup, event hooks |
| `src/aopt/timing.py` | `median_runtime` helper |
| `src/aopt/interface.py` | `price(params, method, **kwargs)` dispatcher |

## Methodology notes

- **Tree:** CRR, `u = exp(sigma*sqrt(dt))`, `p = (exp((r-q)dt) - d)/(u - d)`; raises `ValueError` if `p` is outside [0, 1].
- **Boundary:** at each step, S*(t) is the extreme node where exercise strictly beats continuation
  (puts: largest S; calls: smallest S); NaN if no node qualifies. It is a step function with
  resolution ~ `S*sigma*sqrt(dt)`, so tests use node-spacing tolerances.
- **Reference price:** `A(N) = (V_N + V_{N+1})/2` damps CRR oscillation; `R(N) = 2A(2N) - A(N)`
  removes the leading 1/N term. Reported price is `R` from `A(4N), A(2N)`; the error estimate is
  the gap between the two successive Richardson values (empirical, not a bound).
- **Perpetual put check (q = 0):** try `V = A*S^(-b)` in `0.5*sigma^2*S^2*V'' + r*S*V' - r*V = 0`;
  this requires `0.5*sigma^2*b*(b+1) - r*b - r = 0`, so `b = 2r/sigma^2` is the decaying root.
  Value matching and smooth pasting (`V = K - S*`, `V' = -1` at `S*`) give `S* = K*b/(b+1)`.
  Example: r = 8%, sigma = 40% gives b = 1 and S* = 50, which the tree test checks; r = 5%, sigma = 20% gives b = 2.5, S* = 71.4.

## Step 1 results (S=K=100, T=1, r=5%, q=0, sigma=20%, put)

- Reference American put: **6.090371 +/- 1.9e-6** (literature: ~6.0904).
- Measured log-log convergence slope vs N (N = 20..1600): raw CRR **-1.01**, averaged **-0.92**,
  Richardson **-1.36**. Raw CRR is first order and oscillates; averaging removes the oscillation
  (about 2-3x smaller error at equal N, in the plot). Richardson is 1-2 orders of magnitude more accurate
  but irregular, and -1.36 is not a clean second-order rate: the early-exercise kink and the position of the
  strike relative to the final-layer nodes degrade the extrapolation, so no second-order claim is made.
- The reference is built from the tree, so tree-vs-reference errors are partly self-referential; Step 2's
  PDE is the first independent check (see below).
- European tree error vs Black-Scholes: ~1.94/N.

![convergence](results/figures/step1_convergence.png)
![boundary](results/figures/step1_boundary.png)

## Step 2: Crank-Nicolson PDE (same contract, 400 x 400 unless stated)

Design: log-price grid on `[ln K - L, ln K + L]` (`L = max(3, 5*sigma*sqrt(T) + |r-q|*T)`), shifted by at most half
a cell so S0 is exactly a node (no interpolation); Dirichlet boundaries `max(European asymptote, intrinsic)`.
American exercise per time step by red-black PSOR (vectorized, auto-optimal omega) or a penalty method
(semi-smooth Newton, `rho = 1e8`). Rannacher: first two CN steps replaced by four implicit half-steps.
`solve_pde(..., events=[Event(t, hook)])` forces event times to be step boundaries (unused so far).
The boundary S*(t) is sub-node accurate: the contact edge node is refined by extrapolating `sqrt(V - payoff)`,
which is locally linear: smooth pasting (V = payoff and V' = payoff' at S*) makes V - payoff vanish quadratically there.

Measured (`python scripts/step2_pde.py`, S=K=100, T=1, r=5%, q=0, sigma=20%, put):

- PDE price **6.084234** vs tree reference 6.090371: gap **-6.1e-3**; M = N = 1600 gives -4.1e-4.
  The error is spatial and second order: slope **-1.99** (Rannacher on) and **-2.00** (off), unchanged when N
  is raised at fixed M. It is large at M = 400 because the default domain is wide (h = 2L/M = 0.015); L = 2 with
  M = N = 800 reaches 7e-4, and L = 1 gives 8e-4 already at M = 400. The initial condition samples the kink on a
  node, which biases the price low (the European PDE shows the same sign against Black-Scholes).
- **Independent check of the tree reference.** The PDE shares no code or discretization with the tree. From the
  unrounded PDE prices at M = N = 400, 800, 1600 (6.0842341, 6.0887987, 6.0899630) the observed order is 1.97, and
  Richardson extrapolation gives 6.090362 (using the observed order; 6.090345 assuming exactly order 2 from the 400 and
  1600 prices alone). Both are within 9e-6 and 2.6e-5 of the tree reference 6.090371, so the two methods agree
  independently. This is an agreement check, not a proof: the extrapolation error is larger than the tree's own
  estimated error (1.9e-6) and is not quantified from three points.
- With S0 = 97 (kink between nodes) the slope is **-1.76** (on) / **-1.77** (off).
- **Rannacher has no measurable effect on price convergence here**; its effect is on gamma: with M = 400, N = 20 the
  unsmoothed gamma zigzags around the strike (second differences ~16x larger than with Rannacher). Delta is unaffected.
- Why Rannacher leaves the price order alone: the price at one point barely sees the high-frequency error the kink
  injects into the Crank-Nicolson solution; that error shows up in second derivatives, hence gamma.
- PSOR and penalty agree to ~1e-8 in price. Penalty is about 3.5x faster at every size (M = N = 1600:
  0.065 s vs 0.23 s) and needs ~1.03 Newton iterations per step; PSOR needs 5-9 sweeps per step, growing with M.
  Timings are machine dependent (table: `results/step2_table.csv`).
- Boundary: matches the tree within 0.8 on 0.1 < t < 0.9 (the tree's node spacing is ~0.8-1.2), has no plateaus,
  and is monotone only up to O(h) jumps where the contact edge advances one node (largest measured drop 0.4).
  Against a M = N = 3200 PDE boundary its error is ~30-40% below the raw contact-edge node and halves per
  doubling of M, i.e. first order.

![pde convergence](results/figures/step2_convergence.png)
![greeks](results/figures/step2_greeks.png)
![boundary](results/figures/step2_boundary.png)

### Notes on domain width and tolerances

- **The default domain is wider than needed.** `L = max(3, ...)` applies the minimum of 3 even though the
  `5*sigma*sqrt(T) + |r-q|*T` rule gives 1.05 for these parameters, so the grid is about 3x too wide and the PDE
  error is about 7x larger than necessary at a given M (h = 2L/M). Measured with L = 1.05 (not yet the default):
  gap to the reference -9.0e-4 at M = N = 400 and -7.7e-5 at 1600 (the observed order drops slightly below 2 at small L,
  where boundary truncation starts to matter). The numbers and plots above use the L = 3 default.
- **Test tolerances were set from measurements**, not derived a priori: each is a measured value with margin, with the
  reason in the test comment (for example 8e-3 for the M = N = 400 tree agreement against a measured 6.1e-3, and a
  separate L = 2, M = N = 800 test that does reach 1e-3).
