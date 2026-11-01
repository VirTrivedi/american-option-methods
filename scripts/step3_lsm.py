"""Step 3 experiments: LSM vs the Bermudan tree benchmark, basis choices, bias decomposition, boundary, Pareto data."""
from __future__ import annotations
import csv
import sys
from functools import partial
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from aopt.lsm import evaluate_fit, fit_lsm, lsm_boundary, lsm_price
from aopt.params import OptionParams
from aopt.pde import solve_pde
from aopt.reference import bermudan_reference, reference_price
from aopt.timing import median_runtime
from aopt.tree import bermudan_steps, exercise_boundary, tree_price


# configuration
PARAMS = OptionParams(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="put")
N_EX = 50 # exercise dates for the main experiments
SEED_BASE = 1000
N_SEEDS = 30
PATH_COUNTS = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
N_EVAL_COMMON = 2_000_000 # ONE evaluation set shared by every run in the path-count experiment
EVAL_SEED = 777
DEFAULT_DEGREE, DEFAULT_BASIS = 3, "laguerre"
DECOMP_PATHS, DECOMP_N_EVAL = 1_000_000, 2_000_000 # decomposition: fresh evaluation set per seed, so its noise averages out
SWEEP_PATHS, SWEEP_SEEDS = 50_000, 20
SWEEP_N_EVAL_INDEPENDENT, SWEEP_N_EVAL_COMMON = 100_000, 500_000
DEGREES = [1, 2, 3, 4, 5, 6]
# (label, family, european_basis); chebyshev is omitted: it spans the same space as monomial
VARIANTS = [("monomial", "monomial", False), ("exp-weighted Laguerre", "laguerre", False),
            ("monomial + European value", "monomial", True)]
SHIFT_N_EX = [10, 25, 50, 100]
TREE_BERMUDAN_STEPS, SHIFT_TREE_STEPS, PDE_SHIFT_M = 8000, 12000, 1600
BGK_BETA = 0.5826
BERMUDAN_N_EX = [2, 5, 10, 25, 50, 100]
REFERENCE_N_BASE = 4000
BOUNDARY_PATHS, TREE_BOUNDARY_STEPS, PDE_BOUNDARY_M = 200_000, 2000, 800
PARETO_SEEDS = 5
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR, TABLE_DIR = ROOT / "results" / "figures", ROOT / "results" / "tables"
C = {"in": "#F58518", "out": "#4C78A8", "ref": "#444444", "am": "#54A24B"}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_seeds(n_paths: int, degree: int, family: str, n_eval: int, n_seeds: int, euro: bool = False, 
              eval_seed: int | None = None, n_ex: int = N_EX) -> np.ndarray:
    """Array (n_seeds, 4): in-sample price, in-sample SE, out-of-sample price, out-of-sample SE."""
    out = np.empty((n_seeds, 4))
    for i in range(n_seeds):
        fit = fit_lsm(PARAMS, n_paths, n_ex, degree, family, seed=SEED_BASE + i, european_basis=euro)
        oos = evaluate_fit(PARAMS, fit, n_eval, eval_seed=eval_seed)
        out[i] = fit.price, fit.stderr, oos.price, oos.stderr

    return out


def bermudan_table() -> tuple[list[dict], float]:
    american = reference_price(PARAMS, REFERENCE_N_BASE).price
    rows = []
    for n_ex in BERMUDAN_N_EX:
        ref = bermudan_reference(PARAMS, n_ex, 2000)
        rows.append(dict(n_ex=n_ex, bermudan_price=f"{ref.price:.6f}", error_estimate=f"{ref.error_estimate:.1e}",
                         american_reference=f"{american:.6f}", american_minus_bermudan=f"{american - ref.price:.6f}"))
        
    write_csv(TABLE_DIR / "bermudan_vs_nex.csv", rows)

    return rows, american


def paths_experiment(benchmark: float, american: float) -> list[dict]:
    rows, stats = [], {}
    for n in PATH_COUNTS:
        res = run_seeds(n, DEFAULT_DEGREE, DEFAULT_BASIS, N_EVAL_COMMON, N_SEEDS, eval_seed=EVAL_SEED)
        stats[n] = res
        for label, col in (("in_sample", 0), ("out_of_sample_common_set", 2)):
            rows.append(dict(n_paths=n, mode=label, mean_price=f"{res[:, col].mean():.6f}", std_across_seeds=f"{res[:, col].std(ddof=1):.6f}",
                             mean_reported_se=f"{res[:, col + 1].mean():.6f}", bias_vs_bermudan=f"{res[:, col].mean() - benchmark:+.6f}"))
            
        print(f"  paths {n:>9}: in {res[:, 0].mean():.4f} (sd {res[:, 0].std(ddof=1):.4f})  oos {res[:, 2].mean():.4f} (sd {res[:, 2].std(ddof=1):.4f})")

    write_csv(TABLE_DIR / "lsm_paths.csv", rows)
    common_se = float(np.mean(stats[PATH_COUNTS[-1]][:, 3]))
    print(f"  common evaluation set: {N_EVAL_COMMON:,} paths, its own standard error ~{common_se:.4f} (shared offset of every out-of-sample point)")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ns = np.array(PATH_COUNTS)
    for label, col, color in (("in-sample", 0, C["in"]), ("out-of-sample", 2, C["out"])):
        mean = np.array([stats[n][:, col].mean() for n in PATH_COUNTS])
        std = np.array([stats[n][:, col].std(ddof=1) for n in PATH_COUNTS])
        axes[0].plot(ns, mean, "o-", color=color, label=f"{label} mean over {N_SEEDS} seeds")
        axes[0].fill_between(ns, mean - std, mean + std, color=color, alpha=0.2, label=f"{label} +/- 1 sd")
        axes[1].loglog(ns, std, "o-", color=color, label=f"{label} sd across seeds")

    axes[0].axhline(benchmark, color=C["ref"], ls="--", label=f"Bermudan tree, n_ex={N_EX} ({benchmark:.4f})")
    axes[0].axhline(american, color=C["am"], ls=":", label=f"American reference ({american:.4f})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training paths")
    axes[0].set_ylabel("price")
    axes[0].set_title(f"LSM price vs path count\n(out-of-sample on one common {N_EVAL_COMMON // 10**6}M-path set, shared offset +/-{common_se:.4f})", fontsize=9)
    guide = std[0] * (ns[0] / ns) ** 0.5
    axes[1].loglog(ns, guide, ":", color=C["ref"], label="1/sqrt(n) guide")
    axes[1].set_xlabel("training paths")
    axes[1].set_ylabel("sd across seeds")
    axes[1].set_title("Training-noise spread vs path count", fontsize=9)
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "step3_paths.png", dpi=150)

    return rows


def basis_experiment(benchmark: float) -> None:
    rows, grid = [], {}
    for label, family, euro in VARIANTS:
        for degree in DEGREES:
            ind = run_seeds(SWEEP_PATHS, degree, family, SWEEP_N_EVAL_INDEPENDENT, SWEEP_SEEDS, euro)
            com = run_seeds(SWEEP_PATHS, degree, family, SWEEP_N_EVAL_COMMON, SWEEP_SEEDS, euro, eval_seed=EVAL_SEED)
            grid[label, degree] = (ind, com)

    ref_oos = grid["exp-weighted Laguerre", 3][1][:, 2]
    for (label, degree), (ind, com) in grid.items():
        diff = com[:, 2] - ref_oos  # paired: same training seeds, same evaluation set, so offsets cancel
        rows.append(dict(
            variant=label, degree=degree,
            in_sample_bias=f"{ind[:, 0].mean() - benchmark:+.5f}", in_sample_sd=f"{ind[:, 0].std(ddof=1):.5f}",
            out_of_sample_bias_independent_eval=f"{ind[:, 2].mean() - benchmark:+.5f}", out_of_sample_sd=f"{ind[:, 2].std(ddof=1):.5f}",
            paired_policy_gain_vs_laguerre3=f"{diff.mean():+.5f}", paired_gain_se=f"{diff.std(ddof=1) / np.sqrt(SWEEP_SEEDS):.5f}"))
        
    write_csv(TABLE_DIR / "lsm_basis_sweep.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    colors = {"monomial": "#4C78A8", "exp-weighted Laguerre": "#F58518", "monomial + European value": "#54A24B"}
    for i, (label, _, _) in enumerate(VARIANTS):
        x = np.array(DEGREES) + 0.08 * (i - 1)
        ins = np.array([grid[label, d][0][:, 0].mean() for d in DEGREES]) - benchmark
        ins_se = np.array([grid[label, d][0][:, 0].std(ddof=1) for d in DEGREES]) / np.sqrt(SWEEP_SEEDS)
        axes[0].errorbar(x, ins, yerr=ins_se, fmt="o-", color=colors[label], capsize=3, label=label)
        gain = np.array([(grid[label, d][1][:, 2] - ref_oos).mean() for d in DEGREES])
        gain_se = np.array([(grid[label, d][1][:, 2] - ref_oos).std(ddof=1) for d in DEGREES]) / np.sqrt(SWEEP_SEEDS)
        axes[1].errorbar(x, gain, yerr=gain_se, fmt="o-", color=colors[label], capsize=3, label=label)

    axes[0].axhline(0.0, color=C["ref"], ls="--", lw=1)
    axes[0].set_title(f"in-sample bias vs Bermudan (error bar = SE of mean, {SWEEP_SEEDS} seeds):\noverfitting grows with degree", fontsize=9)
    axes[0].set_ylabel(f"mean in-sample price - benchmark ({SWEEP_PATHS:,} paths)")
    axes[1].axhline(0.0, color=C["ref"], ls="--", lw=1)
    axes[1].set_title("policy quality: out-of-sample price minus laguerre-degree-3\n(paired, common evaluation set; higher = better rule)", fontsize=9)
    axes[1].set_ylabel("paired out-of-sample gain")
    for ax in axes:
        ax.set_xlabel("basis degree")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "step3_basis.png", dpi=150)
    for r in rows:
        if r["degree"] in (1, 2, 3, 6):
            print("  ", r["variant"], r["degree"], "in-sample bias", r["in_sample_bias"], "paired gain", r["paired_policy_gain_vs_laguerre3"], "+/-", r["paired_gain_se"])


def decomposition(benchmark: float, american: float) -> None:
    am_err = reference_price(PARAMS, REFERENCE_N_BASE).error_estimate
    bm_err = bermudan_reference(PARAMS, N_EX, 2000).error_estimate
    res = run_seeds(DECOMP_PATHS, DEFAULT_DEGREE, DEFAULT_BASIS, DECOMP_N_EVAL, N_SEEDS)  # fresh evaluation set per seed
    oos = res[:, 2]
    oos_mean, oos_se_mean = oos.mean(), oos.std(ddof=1) / np.sqrt(N_SEEDS)
    discretization, disc_err = american - benchmark, float(np.hypot(am_err, bm_err))
    policy, policy_err = benchmark - oos_mean, float(oos_se_mean)
    total_err = float(np.hypot(disc_err, policy_err))
    one_run_se = res[:, 3].mean()
    rows = [dict(component="discretization (American - Bermudan benchmark)", value=f"{discretization:.6f}", error=f"{disc_err:.1e}"),
            dict(component="regression/policy (Bermudan benchmark - mean out-of-sample LSM)", value=f"{policy:.6f}", error=f"{policy_err:.6f}"),
            dict(component="total (American - mean out-of-sample LSM)", value=f"{discretization + policy:.6f}", error=f"{total_err:.6f}"),
            dict(component="Monte Carlo standard error of ONE run (out-of-sample)", value=f"{one_run_se:.6f}", error=""),
            dict(component="policy term in units of its standard error (z)", value=f"{policy / policy_err:.2f}", error=""),
            dict(component="mean in-sample LSM minus Bermudan benchmark", value=f"{res[:, 0].mean() - benchmark:+.6f}",
                 error=f"{res[:, 0].std(ddof=1) / np.sqrt(N_SEEDS):.6f}")]
    write_csv(TABLE_DIR / "bias_decomposition.csv", rows)
    for r in rows:
        print("  ", r["component"], r["value"], r["error"])

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.bar(0, discretization, width=0.5, color=C["am"], label=f"exercise-date discretization {discretization:.4f} +/- {disc_err:.0e}")
    ax.bar(0, policy, width=0.5, bottom=discretization, color=C["out"], label=f"regression / policy {policy:.4f} +/- {policy_err:.4f} ({policy / policy_err:.1f} sigma)")
    ax.errorbar(0.0, discretization, yerr=disc_err, fmt="none", ecolor="k", capsize=6)
    ax.errorbar(0.0, discretization + policy, yerr=policy_err, fmt="none", ecolor="k", capsize=6)
    ax.set_xticks([0], [f"{DEFAULT_BASIS} degree {DEFAULT_DEGREE}, {DECOMP_PATHS:,} training paths, n_ex={N_EX}\nmean of {N_SEEDS} seeds, fresh {DECOMP_N_EVAL // 10**6}M-path evaluation each"])
    ax.set_ylabel("American reference - out-of-sample LSM")
    ax.set_title(f"Where the LSM shortfall comes from (one-run MC SE would be {one_run_se:.4f})", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step3_bias.png", dpi=150)


def boundary_plot() -> None:
    fit = fit_lsm(PARAMS, BOUNDARY_PATHS, N_EX, DEFAULT_DEGREE, DEFAULT_BASIS, seed=SEED_BASE)
    t_lsm, s_lsm = lsm_boundary(PARAMS, fit)
    t_tree, s_tree = exercise_boundary(PARAMS, TREE_BOUNDARY_STEPS)
    pde = solve_pde(PARAMS, PDE_BOUNDARY_M, PDE_BOUNDARY_M, return_boundary=True)
    steps = bermudan_steps(TREE_BERMUDAN_STEPS, N_EX)
    t_b, s_b = exercise_boundary(PARAMS, TREE_BERMUDAN_STEPS, steps)
    idx = np.array(sorted(steps))[:-1]
    t_b, s_b = t_b[idx], s_b[idx]

    window = (t_lsm > 0.3) & (t_lsm < 0.9) & ~np.isnan(s_lsm)
    pde_at = np.interp(t_lsm[window], pde.boundary_t, pde.boundary_S)
    berm_at = np.interp(t_lsm[window], t_b, s_b)
    gap_pde, gap_berm = float(np.mean(s_lsm[window] - pde_at)), float(np.mean(s_lsm[window] - berm_at))
    write_csv(TABLE_DIR / "boundary_gap.csv", [
        dict(comparison="LSM - American PDE boundary", mean_gap=f"{gap_pde:+.3f}", mean_abs_gap=f"{np.mean(np.abs(s_lsm[window] - pde_at)):.3f}"),
        dict(comparison="LSM - Bermudan tree boundary (same dates)", mean_gap=f"{gap_berm:+.3f}", mean_abs_gap=f"{np.mean(np.abs(s_lsm[window] - berm_at)):.3f}")])
    print(f"  mean LSM-minus-boundary over 0.3<t<0.9: vs American PDE {gap_pde:+.3f}, vs Bermudan tree {gap_berm:+.3f}")

    shift_rows = []
    pde_fine = solve_pde(PARAMS, PDE_SHIFT_M, PDE_SHIFT_M, return_boundary=True)
    for n_ex in SHIFT_N_EX:
        st = bermudan_steps(SHIFT_TREE_STEPS, n_ex)
        tt, ss = exercise_boundary(PARAMS, SHIFT_TREE_STEPS, st)
        ii = np.array(sorted(st))[:-1]
        tt, ss = tt[ii], ss[ii]
        keep = ~np.isnan(ss) & (tt > 0.2) & (tt < 0.8)
        am_at = np.interp(tt[keep], pde_fine.boundary_t, pde_fine.boundary_S)
        ratio = float(np.median(ss[keep] / am_at))
        mean_gap = float(np.mean(ss[keep] - am_at))
        pred_gap = float(np.mean(am_at * (np.exp(BGK_BETA * PARAMS.sigma * np.sqrt(PARAMS.T / n_ex)) - 1.0)))
        x = PARAMS.sigma * np.sqrt(PARAMS.T / n_ex)
        shift_rows.append(dict(n_ex=n_ex, sigma_sqrt_dt=f"{x:.5f}", measured_ratio=f"{ratio:.5f}", mean_gap_bermudan_minus_american=f"{mean_gap:.4f}", predicted_mean_gap_bgk=f"{pred_gap:.4f}", predicted_ratio_bgk=f"{np.exp(BGK_BETA * x):.5f}",
                               implied_beta=f"{np.log(ratio) / x:.3f}", share_of_predicted_shift=f"{(ratio - 1) / (np.exp(BGK_BETA * x) - 1):.2f}"))
        
    write_csv(TABLE_DIR / "boundary_shift.csv", shift_rows)
    for r in shift_rows:
        print("  ", r)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.plot(t_tree, s_tree, color=C["in"], lw=1.0, label=f"American tree (N={TREE_BOUNDARY_STEPS})")
    ax.plot(pde.boundary_t, pde.boundary_S, color=C["out"], lw=1.4, label=f"American PDE (M=N={PDE_BOUNDARY_M})")
    ax.plot(t_b, s_b, "s", color="#B279A2", ms=3.5, label=f"Bermudan tree, same {N_EX} dates (N={TREE_BERMUDAN_STEPS})")
    ax.plot(t_lsm, s_lsm, "o-", color=C["am"], ms=3.5, lw=0.8, label=f"LSM ({BOUNDARY_PATHS:,} paths, {N_EX} dates, unsmoothed)")
    ax.axhline(PARAMS.K, color="k", ls="--", lw=1, label="strike K")
    ax.set_ylim(75, 101)
    ax.set_xlabel("time t (years)")
    ax.set_ylabel("critical stock price S*")
    ax.set_title(f"Boundaries: LSM minus Bermudan tree {gap_berm:+.2f}, minus American PDE {gap_pde:+.2f} (mean, 0.3<t<0.9)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    xs = np.array([float(r["sigma_sqrt_dt"]) for r in shift_rows])
    ys = np.array([float(r["measured_ratio"]) - 1.0 for r in shift_rows])
    grid_x = np.linspace(0, xs.max() * 1.05, 50)
    ax.plot(xs, ys, "o", color=C["out"], label="measured (Bermudan / American boundary - 1)")
    ax.plot(grid_x, np.exp(BGK_BETA * grid_x) - 1, "--", color=C["ref"], label=f"exp({BGK_BETA} sigma sqrt(dt)) - 1")
    beta_fit = float(np.sum(xs * np.log1p(ys)) / np.sum(xs**2))
    ax.plot(grid_x, np.exp(beta_fit * grid_x) - 1, "-", color=C["am"], label=f"fitted constant {beta_fit:.2f}")
    ax.set_xlabel("sigma * sqrt(dt)  (dt = T / n_ex)")
    ax.set_ylabel("relative upward shift of the boundary")
    ax.set_title("Discrete-exercise continuity correction", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step3_boundary.png", dpi=150)
    print(f"  fitted beta through the origin: {beta_fit:.3f}")


def pareto_table(american: float, benchmark: float) -> None:
    rows = []

    def add(method: str, config: str, price: float, seconds: float, seed: str = "", bermudan: bool = False) -> None:
        rows.append(dict(method=method, config=config, price=f"{price:.8f}", error_vs_american=f"{price - american:+.3e}",
                         error_vs_bermudan=f"{price - benchmark:+.3e}" if bermudan else "", runtime_seconds=f"{seconds:.5f}", seed=seed))

    for k in (2, 4, 8, 16, 32, 64):  # Bermudan tree on the LSM's N_EX dates; N must be a multiple of N_EX
        n = N_EX * k
        value, sec = median_runtime(partial(tree_price, PARAMS, n, False, bermudan_steps(n, N_EX)), repeats=3)
        add("tree_bermudan", f"N={n}, n_ex={N_EX}", value, sec, bermudan=True)

    for n in (50, 100, 200, 400, 800, 1600, 3200):
        value, sec = median_runtime(partial(tree_price, PARAMS, n), repeats=3)
        add("tree", f"N={n}", value, sec)

    for nb in (250, 500, 1000, 2000):
        ref, sec = median_runtime(partial(reference_price, PARAMS, nb), repeats=3)
        add("tree_reference", f"n_base={nb}", ref.price, sec)

    for m in (100, 200, 400, 800, 1600):
        for method in ("psor", "penalty"):
            res, sec = median_runtime(partial(solve_pde, PARAMS, m, m, american_method=method), repeats=3)
            add(f"pde_{method}", f"M=N={m}", res.price, sec)

    for n in PATH_COUNTS:
        for i in range(PARETO_SEEDS):
            seed = SEED_BASE + i
            for mode in ("in_sample", "out_of_sample"):
                fn = partial(lsm_price, PARAMS, n, N_EX, DEFAULT_DEGREE, DEFAULT_BASIS, seed, True, mode, n)
                res, sec = median_runtime(fn, repeats=1, warmup=0)
                add(f"lsm_{mode}", f"paths={n}, n_ex={N_EX}, {DEFAULT_BASIS} deg {DEFAULT_DEGREE}", res.price, sec, str(seed), bermudan=True)

    write_csv(TABLE_DIR / "pareto_raw.csv", rows)
    print(f"  wrote {len(rows)} rows to pareto_raw.csv")


def main(stages: set[str]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    print("Bermudan tree benchmark vs n_ex")
    rows, american = bermudan_table()
    for r in rows:
        print("  ", r)

    benchmark = bermudan_reference(PARAMS, N_EX, 2000).price
    print(f"American reference {american:.6f}; Bermudan benchmark (n_ex={N_EX}) {benchmark:.6f}")
    for name, fn in (("paths", lambda: paths_experiment(benchmark, american)), ("basis", lambda: basis_experiment(benchmark)),
                     ("decomposition", lambda: decomposition(benchmark, american)), ("boundary", boundary_plot),
                     ("pareto", lambda: pareto_table(american, benchmark))):
        if not stages or name in stages:
            print(name)
            fn()


if __name__ == "__main__":
    main(set(sys.argv[1:]))