"""Step 4 experiments: discrete cash dividends in the tree, PDE and LSM."""
from __future__ import annotations
import csv
import math
from dataclasses import replace
from functools import partial
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from aopt.lsm import fit_lsm, evaluate_fit, lsm_boundary
from aopt.params import OptionParams
from aopt.pde import solve_pde
from aopt.reference import bermudan_reference, dividend_reference, dividend_tree_reference
from aopt.timing import median_runtime
from aopt.tree import aligned_steps, bermudan_steps, escrowed_price, exercise_boundary, tree_price


# configuration
BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2)
T_DIV = 0.5
D_GRID = np.round(np.linspace(0.0, 5.0, 21), 4)
PDE_M = 800 # PDE grid for the sweeps (M = N)
TREE_N = 4000
BOUNDARY_D = {"put": 1.0, "call": 6.0}
BOUNDARY_PATHS, BOUNDARY_N_EX, SEED = 200_000, 50, 4242
YIELD_D = 3.0 # discrete vs continuous comparison
SCENARIOS = { # name -> (params, n_ex for the LSM / Bermudan tree)
    "one_dividend_put": (OptionParams(**BASE, option_type="put", dividends=((T_DIV, 3.0),)), 50),
    "quarterly_put": (OptionParams(**BASE, option_type="put", dividends=((0.25, 1.0), (0.5, 1.0), (0.75, 1.0))), 48),
    "large_dividend_call": (OptionParams(**BASE, option_type="call", dividends=((T_DIV, 6.0),)), 50),
}
PARETO_TREE_N = (250, 500, 1000, 2000, 4000)
PARETO_PDE_M = (100, 200, 400, 800, 1600)
LSM_PATHS, LSM_SEEDS, LSM_DEGREE, LSM_BASIS = (10_000, 100_000, 500_000), 3, 3, "laguerre"
LSM_N_EVAL = 500_000
EVAL_SEED = 777 # one common evaluation set, so policy comparisons are paired
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR, TABLE_DIR = ROOT / "results" / "figures", ROOT / "results" / "tables"
C = {"tree": "#F58518", "pde": "#4C78A8", "lsm": "#54A24B", "ref": "#444444", "berm": "#B279A2"}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make(option_type: str, d: float, t_d: float = T_DIV, q: float = 0.0) -> OptionParams:
    return OptionParams(**{**BASE, "q": q}, option_type=option_type, dividends=((t_d, d),) if d else ())


def pde_price(p: OptionParams, european: bool = False) -> float:
    return solve_pde(p, PDE_M, PDE_M, european=european, american_method="penalty").price


def merton_threshold() -> float:
    return BASE["K"] * (1.0 - math.exp(-BASE["r"] * (BASE["T"] - T_DIV)))


def premium_vs_dividend() -> None:
    rows = []
    for typ in ("put", "call"):
        for d in D_GRID:
            p = make(typ, float(d))
            am, eu = pde_price(p), pde_price(p, True)
            tr = tree_price(p, TREE_N)
            rows.append(dict(option_type=typ, D=f"{d:.2f}", pde_american=f"{am:.6f}", pde_european=f"{eu:.6f}", early_exercise_premium=f"{am - eu:.6f}",
                             tree_american=f"{tr:.6f}", tree_minus_pde=f"{tr - am:+.2e}"))
    write_csv(TABLE_DIR / "dividend_premium_vs_D.csv", rows)
    thr = merton_threshold()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for typ, ax in (("put", axes[0]), ("call", axes[1])):
        sub = [r for r in rows if r["option_type"] == typ]
        ds = [float(r["D"]) for r in sub]
        ax.plot(ds, [float(r["pde_american"]) for r in sub], "o-", color=C["pde"], label="American (PDE)")
        ax.plot(ds, [float(r["pde_european"]) for r in sub], "s--", color=C["ref"], ms=4, label="European (PDE)")
        ax.plot(ds, [float(r["tree_american"]) for r in sub], "x", color=C["tree"], label=f"American tree N={TREE_N}")
        if typ == "call":
            ax.axvline(thr, color="r", ls=":", label=f"Merton threshold K(1-e^(-r(T-t_d))) = {thr:.3f}")

        ax.set_title(f"{typ}: price vs cash dividend D at t={T_DIV}", fontsize=9)
        ax.set_xlabel("D")
        ax.set_ylabel("price")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for typ, color in (("put", C["pde"]), ("call", C["lsm"])):
        sub = [r for r in rows if r["option_type"] == typ]
        axes[2].plot([float(r["D"]) for r in sub], [float(r["early_exercise_premium"]) for r in sub], "o-", color=color, label=typ)

    axes[2].axvline(thr, color="r", ls=":", label="call Merton threshold")
    axes[2].set_title("early-exercise premium (American - European, same dividend)", fontsize=9)
    axes[2].set_xlabel("D")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step4_premium_vs_D.png", dpi=150)
    call_below = [float(r["early_exercise_premium"]) for r in rows if r["option_type"] == "call" and float(r["D"]) < thr]
    call_above = [(float(r["D"]), float(r["early_exercise_premium"])) for r in rows if r["option_type"] == "call" and float(r["D"]) > thr]
    print(f"  Merton threshold {thr:.4f}; call premium for D below it: max {max(call_below):.2e}; above: {call_above[:3]} ...")


def premium_decomposition() -> None:
    """Put premium split into exercise only up to the ex-date and only after it (tree, N=4000)."""
    n = TREE_N
    k_d = round(T_DIV / BASE["T"] * n)
    before, after = set(range(1, k_d + 1)), set(range(k_d + 1, n + 1))
    rows = []
    for d in np.round(np.linspace(0.0, 5.0, 11), 2):
        p = make("put", float(d))
        eu = tree_price(p, n, european=True)
        both, pre, post = (tree_price(p, n, exercise_steps=steps) for steps in (None, before, after))
        rows.append(dict(D=f"{d:.2f}", premium_total=f"{both - eu:.5f}", premium_exercise_only_up_to_ex_date=f"{pre - eu:.5f}",
                         premium_exercise_only_after_ex_date=f"{post - eu:.5f}", interaction=f"{both - eu - (pre - eu) - (post - eu):+.5f}"))
        print("  ", rows[-1])

    write_csv(TABLE_DIR / "dividend_put_premium_split.csv", rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ds = [float(r["D"]) for r in rows]
    for key, label, color in (("premium_total", "total", C["pde"]), ("premium_exercise_only_up_to_ex_date", "exercise only up to the ex-date", C["tree"]),
                              ("premium_exercise_only_after_ex_date", "exercise only after the ex-date", C["lsm"])):
        ax.plot(ds, [float(r[key]) for r in rows], "o-", color=color, label=label)
    ax.set_xlabel("D")
    ax.set_ylabel("premium over the European put")
    ax.set_title("American put early-exercise premium by exercise period", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step4_put_premium_split.png", dpi=150)


def boundaries() -> list[dict]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    out = []
    for ax, typ in zip(axes, ("put", "call")):
        d = BOUNDARY_D[typ]
        p = make(typ, d)
        n = aligned_steps(p, TREE_N)
        t_t, s_t = exercise_boundary(p, n)
        pde = solve_pde(p, 1600, 1600, american_method="penalty", return_boundary=True)
        fit = fit_lsm(p, BOUNDARY_PATHS, BOUNDARY_N_EX, LSM_DEGREE, LSM_BASIS, seed=SEED)
        t_l, s_l = lsm_boundary(p, fit)
        if typ == "put":
            ax.plot(t_t, s_t, "-", color=C["tree"], lw=1.2, label=f"American tree (N={n})")
            m = ~np.isnan(pde.boundary_S) & (pde.boundary_t < BASE["T"] - 1e-9)
            order = np.argsort(pde.boundary_t)  # keep NaNs so the line breaks where the exercise region is empty
            ax.plot(pde.boundary_t[order], pde.boundary_S[order], "-", color=C["pde"], lw=1.2, label="American PDE (M=N=1600)")
            ax.plot(t_l, s_l, "o", color=C["lsm"], ms=3, label=f"LSM ({BOUNDARY_PATHS:,} paths, {BOUNDARY_N_EX} dates; inside the empty window these are polynomial crossings, not a region)")
            ax.axvline(T_DIV, color="r", ls=":", label="ex-date")
            ax.set_ylim(55, 101)
            before = np.flatnonzero(~np.isnan(s_t[: n // 2])).max()
            jump_tree = s_t[n // 2 + 1] - s_t[before]
            pm = m & (pde.boundary_t < T_DIV - 1e-9)
            qm = m & (pde.boundary_t > T_DIV + 1e-9)
            pre_p, post_p = pde.boundary_S[pm][np.argmax(pde.boundary_t[pm])], pde.boundary_S[qm][np.argmin(pde.boundary_t[qm])]
            lsm_pre = s_l[~np.isnan(s_l) & (t_l < T_DIV - 1e-9)]
            out.append(dict(case=f"put D={d}", tree_vanishes_at_t=f"{before / n:.4f}", tree_S_before=f"{s_t[before]:.3f}", tree_S_after=f"{s_t[n // 2 + 1]:.3f}", tree_jump=f"{jump_tree:+.3f}",
                            pde_vanishes_at_t=f"{pde.boundary_t[pm].max():.4f}", pde_S_before=f"{pre_p:.3f}", pde_S_after=f"{post_p:.3f}", pde_jump=f"{post_p - pre_p:+.3f}",
                            lsm_last_pre_ex_date_S=f"{lsm_pre[-1]:.3f}" if lsm_pre.size else "", d_over_rK=f"{d / (BASE['r'] * BASE['K']):.3f}"))
            ax.set_title(f"put, D={d} at t={T_DIV}: no exercise in the last ~D/(rK)={d / (BASE['r'] * BASE['K']):.2f} yr before the ex-date;\nboundary re-emerges {jump_tree:+.1f} higher", fontsize=9)
        else:
            k = n // 2
            ex_lsm = np.flatnonzero(~np.isnan(s_l))
            ax.plot([T_DIV], [s_t[k]], "D", color=C["tree"], ms=8, label=f"tree: {s_t[k]:.2f}")
            at = np.abs(pde.boundary_t - T_DIV) < 1e-9
            ax.plot([T_DIV], [pde.boundary_S[at][0]], "s", color=C["pde"], ms=6, label=f"PDE: {pde.boundary_S[at][0]:.2f}")
            lsm_at = s_l[BOUNDARY_N_EX // 2 - 1]
            ax.plot([T_DIV], [lsm_at], "o", color=C["lsm"], ms=4, label=f"LSM: {lsm_at:.2f}")
            others = [i for i in ex_lsm if i != BOUNDARY_N_EX // 2 - 1]
            ax.plot(t_l[others], s_l[others], ".", color=C["lsm"], alpha=0.4, label="LSM at other dates (fitted-polynomial crossing; no true region)")
            ax.axhline(BASE["K"], color="k", ls="--", lw=1, label="strike")
            ax.set_xlim(0, 1)
            ax.set_ylim(90, 140)
            out.append(dict(case=f"call D={d}", tree_S_at_ex_date=f"{s_t[k]:.3f}", pde_S_at_ex_date=f"{pde.boundary_S[at][0]:.3f}", lsm_S_at_ex_date=f"{lsm_at:.3f}",
                            tree_exercise_steps=str(list(np.flatnonzero(~np.isnan(s_t[:-1])))), lsm_other_dates_with_boundary=str(len(others))))
            ax.set_title(f"call, D={d} at t={T_DIV}: exercise region exists only at the ex-date (cum-dividend)", fontsize=9)

        ax.set_xlabel("time t")
        ax.set_ylabel("critical stock price")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "step4_boundaries.png", dpi=150)
    write_csv(TABLE_DIR / "dividend_boundaries.csv", [{k: r.get(k, "") for k in dict.fromkeys(k for r in out for k in r)} for r in out])
    for r in out:
        print("  ", r)

    return out


def discrete_vs_continuous() -> None:
    d = YIELD_D
    q = -math.log(1.0 - d * math.exp(-BASE["r"] * T_DIV) / BASE["S"]) / BASE["T"]
    disc, cont, none = make("put", d), make("put", 0.0, q=q), make("put", 0.0)
    res = {}
    for name, p in (("discrete", disc), ("continuous", cont), ("none", none)):
        am = solve_pde(p, 1600, 1600, american_method="penalty", return_boundary=True)
        eu = solve_pde(p, 1600, 1600, european=True)
        res[name] = (am, eu)

    rows = [dict(model=name, american=f"{am.price:.6f}", european=f"{eu.price:.6f}", premium=f"{am.price - eu.price:.6f}", yield_q=f"{q if name == 'continuous' else 0.0:.5f}")
            for name, (am, eu) in res.items()]
    write_csv(TABLE_DIR / "dividend_discrete_vs_continuous.csv", rows)
    for r in rows:
        print("  ", r)

    am_d, am_c = res["discrete"][0], res["continuous"][0]
    s_grid = np.linspace(60, 140, 161)
    diff_s = np.interp(s_grid, am_d.S, am_d.V) - np.interp(s_grid, am_c.S, am_c.V)
    # re-solve on shifted spots for the S0 dependence (the grid at t=0 is the full value function, so interpolation is enough)
    print(f"  price difference discrete - continuous at S0=100: {am_d.price - am_c.price:+.4f}; over S0 in [60,140]: max |diff| {np.abs(diff_s).max():.4f} at S0={s_grid[np.abs(diff_s).argmax()]:.1f}")
    prem_diff = (am_d.price - res['discrete'][1].price) - (am_c.price - res['continuous'][1].price)
    print(f"  early-exercise premium: discrete {am_d.price - res['discrete'][1].price:.4f}, continuous {am_c.price - res['continuous'][1].price:.4f} (diff {prem_diff:+.4f})")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(s_grid, np.interp(s_grid, am_d.S, am_d.V), color=C["pde"], label="discrete D=3 at t=0.5")
    axes[0].plot(s_grid, np.interp(s_grid, am_c.S, am_c.V), "--", color=C["tree"], label=f"continuous yield q={q:.4f}")
    axes[0].set_title("American put value at t=0", fontsize=9)
    axes[1].plot(s_grid, diff_s, color=C["ref"])
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("discrete - continuous (same forward at T)", fontsize=9)
    for name, color, ls in (("discrete", C["pde"], "-"), ("continuous", C["tree"], "--"), ("none", C["ref"], ":")):
        am = res[name][0]
        m = ~np.isnan(am.boundary_S) & (am.boundary_t < BASE["T"] - 1e-9)
        axes[2].plot(am.boundary_t[m], am.boundary_S[m], ls, color=color, label=name)

    axes[2].axvline(T_DIV, color="r", ls=":", lw=1)
    axes[2].set_ylim(60, 101)
    axes[2].set_title("early-exercise boundary", fontsize=9)
    for ax in axes:
        ax.set_xlabel("S0" if ax is not axes[2] else "time t")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step4_discrete_vs_continuous.png", dpi=150)
    for name in ("discrete", "continuous"):
        am = res[name][0]
        m = ~np.isnan(am.boundary_S) & (am.boundary_t < BASE["T"] - 1e-9)
        pre = am.boundary_t[m & (am.boundary_t < T_DIV)]
        print(f"    {name}: exercise boundary before the ex-date " + (f"exists for t < {pre.max():.3f}" if pre.size else "is empty for all t < 0.5"))
        for t0 in (0.1, 0.25, 0.4, 0.6, 0.8):
            if t0 > T_DIV or pre.size:
                print(f"    {name} boundary at t={t0}: {np.interp(t0, am.boundary_t[m], am.boundary_S[m]):.2f}")


def model_risk() -> None:
    rows = []
    for name, (p, _) in SCENARIOS.items():
        ref = dividend_reference(p)
        esc = escrowed_price(p, 4000)
        eu_spot, eu_esc = tree_price(p, aligned_steps(p, 4000), european=True), escrowed_price(p, 4000, european=True)
        rows.append(dict(scenario=name, spot_jump_reference=f"{ref.price:.6f}", method_uncertainty_pde_vs_tree=f"{ref.uncertainty:.1e}", escrowed_tree_N4000=f"{esc:.6f}",
                         model_difference_american=f"{esc - ref.price:+.5f}", european_spot_jump=f"{eu_spot:.5f}", european_escrowed=f"{eu_esc:.5f}",
                         model_difference_european=f"{eu_esc - eu_spot:+.5f}"))
        print("  ", rows[-1])

    write_csv(TABLE_DIR / "dividend_model_risk.csv", rows)


def call_basis_check() -> None:
    """Large-dividend call: the LSM policy is only right at one date; how much does the basis matter?"""
    p, n_ex = SCENARIOS["large_dividend_call"]
    berm = bermudan_reference(p, n_ex, 2000).price
    k_d = n_ex // 2
    rows = []
    for label, family, degree, euro in (("laguerre", "laguerre", 3, False), ("monomial", "monomial", 3, False), ("monomial", "monomial", 5, False),
                                        ("monomial + European value", "monomial", 3, True)):
        gaps, off_dates = [], []
        for i in range(LSM_SEEDS):
            fit = fit_lsm(p, 200_000, n_ex, degree, family, seed=SEED + i, european_basis=euro, return_policy=True)
            oos = evaluate_fit(p, fit, LSM_N_EVAL, eval_seed=EVAL_SEED)
            counts = np.bincount(fit.exercise_index, minlength=n_ex + 1)
            gaps.append(oos.price - berm)
            off_dates.append(1.0 - counts[k_d] / counts[1:n_ex].sum())

        rows.append(dict(basis=label, degree=degree, mean_oos_minus_bermudan=f"{np.mean(gaps):+.4f}", sd_across_seeds=f"{np.std(gaps, ddof=1):.4f}",
                         share_of_early_exercises_off_ex_date=f"{np.mean(off_dates):.2f}"))
        print("  ", rows[-1])

    write_csv(TABLE_DIR / "dividend_lsm_call_basis.csv", rows)


def scenario_table() -> None:
    rows, pareto = [], []

    def add(scn: str, method: str, config: str, price: float, ref: float, berm: float | None, sec: float, ci: tuple[float, float] | None = None, seed: str = "") -> None:
        rows.append(dict(scenario=scn, method=method, config=config, seed=seed, price=f"{price:.6f}", error_vs_reference=f"{price - ref:+.2e}",
                         error_vs_bermudan=f"{price - berm:+.2e}" if berm is not None else "", runtime_seconds=f"{sec:.4f}",
                         ci95_low=f"{ci[0]:.5f}" if ci else "", ci95_high=f"{ci[1]:.5f}" if ci else ""))

    for scn, (p, n_ex) in SCENARIOS.items():
        ref = dividend_reference(p)
        berm = bermudan_reference(p, n_ex, 2000)
        print(f"  {scn}: reference {ref.price:.6f} (PDE {ref.price:.6f}, tree {ref.tree_price:.6f}, uncertainty {ref.uncertainty:.1e}, PDE order {ref.pde_order:.2f}); Bermudan({n_ex}) {berm.price:.6f} +/- {berm.error_estimate:.1e}")
        for n in PARETO_TREE_N:
            nn = aligned_steps(p, n)
            v, sec = median_runtime(partial(tree_price, p, nn), repeats=3)
            add(scn, "tree", f"N={nn}", v, ref.price, None, sec)

        for n in (250, 500, 1000, 2000):
            nn = aligned_steps(p, n, n_ex)
            v, sec = median_runtime(partial(tree_price, p, nn, False, bermudan_steps(nn, n_ex)), repeats=3)
            add(scn, "tree_bermudan", f"N={nn}, n_ex={n_ex}", v, ref.price, berm.price, sec)

        v, sec = median_runtime(partial(dividend_tree_reference, p, 1000), repeats=1, warmup=0)
        add(scn, "tree_reference", "n_base=1000", v.price, ref.price, None, sec)
        for m in PARETO_PDE_M:
            res, sec = median_runtime(partial(solve_pde, p, m, m, american_method="penalty"), repeats=3)
            add(scn, "pde_penalty", f"M=N={m}", res.price, ref.price, None, sec)

        for n in LSM_PATHS:
            for i in range(LSM_SEEDS):
                seed = SEED + i
                fit, sec_fit = median_runtime(partial(fit_lsm, p, n, n_ex, LSM_DEGREE, LSM_BASIS, seed), repeats=1, warmup=0)
                oos, sec_oos = median_runtime(partial(evaluate_fit, p, fit, LSM_N_EVAL), repeats=1, warmup=0)
                add(scn, "lsm_in_sample", f"paths={n}, n_ex={n_ex}", fit.price, ref.price, berm.price, sec_fit, fit.ci95, str(seed))
                add(scn, "lsm_out_of_sample", f"paths={n}, n_ex={n_ex}, n_eval={LSM_N_EVAL}", oos.price, ref.price, berm.price, sec_fit + sec_oos, oos.ci95, str(seed))

    write_csv(TABLE_DIR / "dividend_scenarios.csv", rows)

    # append to pareto_raw.csv with a scenario column (existing no-dividend rows are labelled "no_dividend"; reruns replace ours)
    path = TABLE_DIR / "pareto_raw.csv"
    with path.open() as f:
        old = [r for r in csv.DictReader(f) if r.get("scenario", "no_dividend") == "no_dividend"]
    fields = list(old[0]) if "scenario" in old[0] else [*old[0], "scenario"]
    out = [{**r, "scenario": "no_dividend"} for r in old]
    for r in rows:
        out.append(dict(method=r["method"], config=r["config"], price=r["price"], error_vs_american=r["error_vs_reference"], error_vs_bermudan=r["error_vs_bermudan"],
                        runtime_seconds=r["runtime_seconds"], seed=r["seed"], scenario=r["scenario"]))
        
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in out])
    print(f"  pareto_raw.csv now has {len(out)} rows ({len(rows)} dividend rows; error_vs_american is vs the dividend reference for those)")

    # compact summary at the largest settings
    for scn in SCENARIOS:
        for method in ("tree", "pde_penalty", "lsm_out_of_sample"):
            sub = [r for r in rows if r["scenario"] == scn and r["method"] == method]
            last = sub[-LSM_SEEDS:] if method.startswith("lsm") else sub[-1:]
            print(f"    {scn:22s} {method:18s} {last[0]['config']:38s} err_vs_ref {[r['error_vs_reference'] for r in last]} runtime {last[0]['runtime_seconds']}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in (("Premium vs dividend size", premium_vs_dividend), ("Put premium split", premium_decomposition), ("Boundaries", boundaries), ("Discrete vs continuous yield", discrete_vs_continuous),
                     ("Model risk (escrowed vs spot-jump)", model_risk), ("LSM basis check, large-dividend call", call_basis_check), ("Three-method scenario table", scenario_table)):
        print(name)
        fn()


if __name__ == "__main__":
    main()