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
from aopt.pde import grid_greeks, solve_pde
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
BOUNDARY_M = BOUNDARY_N = 800
TREE_BOUNDARY_STEPS = 2000
TABLE_SIZES = [(100, 100), (200, 200), (400, 400), (800, 800), (1600, 1600)]
TIMING_REPEATS = 5
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR, TABLE_PATH = ROOT / "results" / "figures", ROOT / "results" / "step2_table.csv"
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


def greeks_figure() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fine = solve_pde(PARAMS, 1600, 1600)
    runs = [("Rannacher on", C_ON, True), ("Rannacher off", C_OFF, False)]
    results = [(name, color, solve_pde(PARAMS, GREEKS_M, GREEKS_N, rannacher=flag)) for name, color, flag in runs]
    results.append(("fine grid (M=N=1600)", C_REF, fine))
    for name, color, res in results:
        x = np.log(res.S)
        S, delta, gamma = grid_greeks(x, res.V)
        w = (S >= GREEKS_WINDOW[0]) & (S <= GREEKS_WINDOW[1])
        style = dict(color=color, lw=1.0 if color != C_REF else 1.5, ls="--" if color == C_REF else "-")
        marker = dict(marker="o", ms=2.5) if color != C_REF else {}
        axes[0].plot(S[w], delta[w], label=name, **style, **marker)
        axes[1].plot(S[w], gamma[w], label=name, **style, **marker)

    for ax, ylabel in zip(axes, ("delta", "gamma")):
        ax.axvline(PARAMS.K, color="k", lw=0.6, ls=":")
        ax.set_xlabel("S")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"American put Greeks near the strike, M={GREEKS_M}, N={GREEKS_N}")
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
    ref = reference_price(PARAMS, REFERENCE_N_BASE)
    ref_off = reference_price(replace(PARAMS, S=OFFNODE_S0), REFERENCE_N_BASE)
    print(f"Tree reference: {ref.price:.6f} +/- {ref.error_estimate:.1e}")
    pde400 = solve_pde(PARAMS, 400, 400)
    print(f"PDE M=N=400: {pde400.price:.6f}, gap to reference {pde400.price - ref.price:+.2e}, delta {pde400.delta:.5f}, gamma {pde400.gamma:.5f}, PSOR sweeps/step {pde400.mean_iterations:.1f}")

    fig, slopes = convergence_figure(ref.price, ref_off.price)
    fig.savefig(FIG_DIR / "step2_convergence.png", dpi=150)
    print("Convergence slopes:", {k: round(v, 2) for k, v in slopes.items()})
    greeks_figure().savefig(FIG_DIR / "step2_greeks.png", dpi=150)
    fig, gap = boundary_figure()
    fig.savefig(FIG_DIR / "step2_boundary.png", dpi=150)
    print(f"Max |PDE - tree| boundary difference for 0.1 < t < 0.9: {gap:.3f}")
    for row in write_table(ref.price):
        print(row)

    print(f"Saved figures to {FIG_DIR} and table to {TABLE_PATH}")


if __name__ == "__main__":
    main()