"""Step 2 figures and table: Crank-Nicolson PDE accuracy, Rannacher smoothing, boundary, runtime."""
from __future__ import annotations
import csv
from functools import partial
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace
from aopt.params import OptionParams
from aopt.pde import default_half_width, grid_greeks, solve_pde
from aopt.reference import reference_price
from aopt.timing import median_runtime
from aopt.tree import exercise_boundary


# configuration
PARAMS = OptionParams(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="put")
OFFNODE_S0 = 97.0  # second convergence panel
REFERENCE_N_BASE = 4000
CONVERGENCE_M = [50, 100, 200, 400, 800, 1600]  # M = N, doubling together
GREEKS_M, GREEKS_N = 400, 20  # few time steps relative to space steps, where CN oscillates
GREEKS_WINDOW = (90.0, 110.0)
ROUGHNESS_N = [10, 20, 40, 80, 160, 320, 640]  # time steps swept at fixed M for the roughness plot
BOUNDARY_M = BOUNDARY_N = 800
TREE_BOUNDARY_STEPS = 2000
TABLE_SIZES = [(100, 100), (200, 200), (400, 400), (800, 800), (1600, 1600)]
TIMING_REPEATS = 5
ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR, TABLE_PATH = ROOT / "results" / "figures", TABLE_DIR / "step2_table.csv"
SENSITIVITY_PATH = TABLE_DIR / "pde_domain_sensitivity.csv"
ROUGHNESS_PATH = TABLE_DIR / "rannacher_gamma_roughness.csv"
OLD_L = 3.0  # previous default domain half-width, kept for the sensitivity table
C_ON, C_OFF, C_REF = "#4C78A8", "#F58518", "#444444"


def slope(ms: list[int], errs: list[float]) -> float:
    return float(np.polyfit(np.log(ms), np.log(errs), 1)[0])


def convergence_figure(ref_on_node: float, ref_off_node: float) -> tuple[plt.Figure, dict[str, float]]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    slopes: dict[str, float] = {}
    for ax, p, ref, title in ((axes[0], PARAMS, ref_on_node, f"S0 = K = {PARAMS.K:g} (kink on a node)"),
        (axes[1], replace(PARAMS, S=OFFNODE_S0), ref_off_node, f"S0 = {OFFNODE_S0:g} (kink between nodes)")):
        for rannacher, color, marker, name in ((True, C_ON, "o", "Rannacher on"), (False, C_OFF, "s", "Rannacher off")):
            errs = [abs(solve_pde(p, m, m, rannacher=rannacher).price - ref) for m in CONVERGENCE_M]
            slopes[f"{title.split(' (')[0]} {name}"] = s = slope(CONVERGENCE_M, errs)
            ax.loglog(CONVERGENCE_M, errs, marker + "-", color=color, label=f"{name} (slope {s:.2f})")

        ax.loglog(CONVERGENCE_M, 2.0 * (np.array(CONVERGENCE_M) / CONVERGENCE_M[0]) ** -2.0, ":", color=C_REF, lw=1, label="slope -2 guide")
        ax.set_xlabel("grid size M = N")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("|PDE price - tree reference|")
    fig.suptitle("American put, Crank-Nicolson + PSOR: price error vs grid size")
    fig.tight_layout()

    return fig, slopes


def gamma_roughness(res, window: tuple[float, float] = GREEKS_WINDOW) -> float:
    """Largest second difference of gamma across the nodes near the strike (zero for a smooth curve)."""
    S, _, gamma = grid_greeks(np.log(res.S), res.V)
    w = (S >= window[0]) & (S <= window[1])
    return float(np.max(np.abs(np.diff(gamma[w], 2))))


def greeks_figure() -> plt.Figure:
    """Greeks near the strike at an extreme (N = 20) and a realistic (N = M) time grid, plus roughness vs N."""
    fig = plt.figure(figsize=(14, 7.5))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15])
    fine = solve_pde(PARAMS, 1600, 1600)
    axes = {(r, c): fig.add_subplot(grid[r, c]) for r in range(2) for c in range(2)}
    for row, n_steps in enumerate((GREEKS_N, GREEKS_M)):
        runs = [("Rannacher on", C_ON, True), ("Rannacher off", C_OFF, False)]
        results = [(name, color, solve_pde(PARAMS, GREEKS_M, n_steps, rannacher=flag)) for name, color, flag in runs]
        results.append(("fine grid (M=N=1600)", C_REF, fine))
        for name, color, res in results:
            S, delta, gamma = grid_greeks(np.log(res.S), res.V)
            w = (S >= GREEKS_WINDOW[0]) & (S <= GREEKS_WINDOW[1])
            style = dict(color=color, lw=1.0 if color != C_REF else 1.5, ls="--" if color == C_REF else "-")
            marker = dict(marker="o", ms=2.5) if color != C_REF else {}
            axes[row, 0].plot(S[w], delta[w], label=name, **style, **marker)
            axes[row, 1].plot(S[w], gamma[w], label=name, **style, **marker)

        for col, ylabel in enumerate(("delta", "gamma")):
            ax = axes[row, col]
            ax.axvline(PARAMS.K, color="k", lw=0.6, ls=":")
            ax.set_xlabel("S")
            ax.set_ylabel(ylabel)
            ax.set_title(f"M={GREEKS_M}, N={n_steps}" + (" (deliberately extreme)" if row == 0 else " (realistic)"), fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

    ax = fig.add_subplot(grid[:, 2])
    ns = ROUGHNESS_N
    rows = []
    for flag, color, name in ((True, C_ON, "Rannacher on"), (False, C_OFF, "Rannacher off")):
        rough = [gamma_roughness(solve_pde(PARAMS, GREEKS_M, n, rannacher=flag)) for n in ns]
        ax.loglog(ns, rough, "o-", color=color, label=name)
        rows += [dict(M=GREEKS_M, N=n, rannacher=flag, gamma_roughness=f"{r:.4e}") for n, r in zip(ns, rough)]

    ax.set_xlabel(f"time steps N (M = {GREEKS_M})")
    ax.set_ylabel("gamma roughness (max 2nd difference, S in 90-110)")
    ax.set_title("When Rannacher matters", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    with ROUGHNESS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig.suptitle("American put Greeks near the strike: Rannacher on vs off")
    fig.tight_layout()

    return fig


def boundary_figure() -> tuple[plt.Figure, float]:
    pde = solve_pde(PARAMS, BOUNDARY_M, BOUNDARY_N, return_boundary=True)
    t_tree, s_tree = exercise_boundary(PARAMS, TREE_BOUNDARY_STEPS)
    gap = float(np.nanmax(np.abs(np.interp(t_tree, pde.boundary_t, pde.boundary_S) - s_tree)[(t_tree > 0.1) & (t_tree < 0.9)]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t_tree, s_tree, color=C_OFF, lw=1.0, label=f"tree (N={TREE_BOUNDARY_STEPS}, staircase)")
    ax.plot(pde.boundary_t, pde.boundary_S, color=C_ON, lw=1.4, label=f"PDE (M={BOUNDARY_M}, N={BOUNDARY_N})")
    ax.axhline(PARAMS.K, color="k", ls="--", lw=1, label="strike K")
    ax.set_xlabel("time t (years)")
    ax.set_ylabel("critical stock price S*")
    ax.set_title("American put early-exercise boundary: PDE vs tree")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    return fig, gap


def write_domain_sensitivity(ref: float) -> dict[str, float]:
    """Price gap vs the tree reference for the old fixed L = 3 and the current default domain."""
    rows, slopes = [], {}
    for label, L in (("old_L3", OLD_L), ("current_default", None)):
        gaps = []
        for m in CONVERGENCE_M:
            res = solve_pde(PARAMS, m, m, L=L)
            used = L if L is not None else default_half_width(PARAMS)
            gaps.append(res.price - ref)
            rows.append(dict(domain=label, L=f"{used:.4g}", M=m, N=m, price=f"{res.price:.8f}", gap_vs_reference=f"{gaps[-1]:.3e}"))

        slopes[label] = slope(CONVERGENCE_M, [abs(g) for g in gaps])

    with SENSITIVITY_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        
    return slopes


def write_table(ref: float) -> list[dict]:
    rows = []
    for m, n in TABLE_SIZES:
        for method in ("psor", "penalty"):
            res, seconds = median_runtime(partial(solve_pde, PARAMS, m, n, american_method=method), repeats=TIMING_REPEATS)
            rows.append(dict(method=method, M=m, N=n, price=f"{res.price:.8f}", error_vs_reference=f"{res.price - ref:.3e}", mean_iterations_per_step=f"{res.mean_iterations:.2f}", runtime_seconds=f"{seconds:.4f}"))

    with TABLE_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ref = reference_price(PARAMS, REFERENCE_N_BASE)
    ref_off = reference_price(replace(PARAMS, S=OFFNODE_S0), REFERENCE_N_BASE)
    print(f"Tree reference: {ref.price:.6f} +/- {ref.error_estimate:.1e}")
    pde400 = solve_pde(PARAMS, 400, 400)
    print(f"PDE M=N=400: {pde400.price:.6f}, gap to reference {pde400.price - ref.price:+.2e}, delta {pde400.delta:.5f}, gamma {pde400.gamma:.5f}, PSOR sweeps/step {pde400.mean_iterations:.1f}")

    fig, slopes = convergence_figure(ref.price, ref_off.price)
    fig.savefig(FIG_DIR / "step2_convergence.png", dpi=150)
    print("Convergence slopes:", {k: round(v, 2) for k, v in slopes.items()})
    print("Domain sensitivity slopes (price error vs M=N):", {k: round(v, 2) for k, v in write_domain_sensitivity(ref.price).items()})
    greeks_figure().savefig(FIG_DIR / "step2_greeks.png", dpi=150)
    fig, gap = boundary_figure()
    fig.savefig(FIG_DIR / "step2_boundary.png", dpi=150)
    print(f"Max |PDE - tree| boundary difference for 0.1 < t < 0.9: {gap:.3f}")
    for row in write_table(ref.price):
        print(row)

    print(f"Saved figures to {FIG_DIR} and table to {TABLE_PATH}")


if __name__ == "__main__":
    main()