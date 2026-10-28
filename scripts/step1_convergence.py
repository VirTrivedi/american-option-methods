"""Step 1 figures: tree convergence (raw vs averaged vs extrapolated) and exercise boundary."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from aopt.params import OptionParams
from aopt.reference import averaged_price, reference_price, richardson
from aopt.tree import exercise_boundary, tree_price


# configuration
PARAMS = OptionParams(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="put")
BOUNDARY_STEPS = 2000
N_GRID = np.unique(np.round(np.geomspace(20, 1600, 24)).astype(int))
REFERENCE_N_BASE = 4000  # must be finer than any size in N_GRID, so the reference is not self-referential
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
COLORS = {"raw": "#4C78A8", "avg": "#F58518", "rich": "#54A24B"}


def convergence_figure(ref_price: float) -> tuple[plt.Figure, dict[str, float]]:
    raw = np.array([abs(tree_price(PARAMS, int(n)) - ref_price) for n in N_GRID])
    avg = np.array([abs(averaged_price(PARAMS, int(n)) - ref_price) for n in N_GRID])
    # Richardson at x-value N uses trees of size N and 2N
    rich = np.array([abs(richardson(PARAMS, int(n)) - ref_price) for n in N_GRID])
    floor = 1e-9  # avoid log(0)
    slopes = {k: float(np.polyfit(np.log(N_GRID), np.log(np.maximum(v, floor)), 1)[0]) for k, v in (("raw", raw), ("avg", avg), ("rich", rich))}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(N_GRID, raw, "o-", color=COLORS["raw"], label=f"raw CRR (slope {slopes['raw']:.2f})")
    ax.loglog(N_GRID, avg, "s-", color=COLORS["avg"], label=f"avg of N, N+1 (slope {slopes['avg']:.2f})")
    ax.loglog(N_GRID, rich, "^-", color=COLORS["rich"], label=f"Richardson 2A(2N)-A(N) (slope {slopes['rich']:.2f})")
    ax.loglog(N_GRID, 1.0 / N_GRID, "k:", lw=1, label="reference slope -1 (1/N)")
    ax.set_xlabel("tree steps N")
    ax.set_ylabel("|price - reference|")
    ax.set_title("American put: CRR error vs N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, slopes


def boundary_figure() -> plt.Figure:
    t, s_star = exercise_boundary(PARAMS, BOUNDARY_STEPS)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, s_star, color=COLORS["raw"], lw=1.2, label=f"tree boundary S*(t), N={BOUNDARY_STEPS}")
    ax.axhline(PARAMS.K, color="k", ls="--", lw=1, label="strike K")
    ax.set_xlabel("time t (years)")
    ax.set_ylabel("critical stock price S*")
    ax.set_title("American put: early-exercise boundary (exercise for S <= S*)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = reference_price(PARAMS, n_base=REFERENCE_N_BASE)
    print(f"Reference price: {ref.price:.6f} +/- {ref.error_estimate:.1e} (n_base={ref.n_base})")

    fig, slopes = convergence_figure(ref.price)
    fig.savefig(OUT_DIR / "step1_convergence.png", dpi=150)
    print("Log-log convergence slopes:", {k: round(v, 2) for k, v in slopes.items()})

    boundary_figure().savefig(OUT_DIR / "step1_boundary.png", dpi=150)
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()