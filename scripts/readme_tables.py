"""Every results table in README.md is generated from a CSV in results/tables/ by the functions below (no hand-typed numbers in tables)."""
from __future__ import annotations
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
DOCS = [ROOT / "README.md", ROOT / "docs" / "steps_1_4.md", ROOT / "docs" / "market_check.md", ROOT / "docs" / "diagnostics.md"]  # files holding generated-table markers
Table = tuple[list[str], list[list[str]]]


def fx(x: float, nd: int = 3) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{nd}f}"


def sci(x: float, nd: int = 2) -> str:
    if pd.isna(x):
        return "n/a"

    mant, exp = f"{x:.{nd}e}".split("e")

    return f"{mant}e{int(exp)}"


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TAB / name)


def pde_domain() -> Table:
    d = csv("pde_domain_sensitivity.csv")
    rows = []
    for M in (400, 800, 1600):
        old = d[(d.domain == "old_L3") & (d.M == M)].gap_vs_reference.iloc[0]
        new = d[(d.domain == "current_default") & (d.M == M)].gap_vs_reference.iloc[0]
        rows.append([str(M), sci(old), sci(new)])

    return ["M = N", "gap to tree reference, old L = 3", "gap, current default (L = 1.5)"], rows


def bermudan_nex() -> Table:
    d = csv("bermudan_vs_nex.csv")

    return (["n_ex"] + [str(n) for n in d.n_ex], [["Bermudan price"] + [fx(x, 4) for x in d.bermudan_price], ["American - Bermudan"] + [fx(x, 4) for x in d.american_minus_bermudan]])


def lsm_paths() -> Table:
    d = csv("lsm_paths.csv")
    rows = []
    for n in (1000, 10000, 100000, 1000000):
        i, o = d[(d.n_paths == n) & (d["mode"] == "in_sample")].iloc[0], d[(d.n_paths == n) & (d["mode"] == "out_of_sample_common_set")].iloc[0]
        rows.append([f"{n:,}", f"{i.mean_price:.4f} ({i.std_across_seeds:.4f})", f"{o.mean_price:.4f} ({o.std_across_seeds:.4f})"])

    return ["training paths", "in-sample mean (sd across seeds)", "out-of-sample mean (sd across seeds)"], rows


def bias_decomposition() -> Table:
    d = csv("bias_decomposition.csv")

    return ["component", "value", "error"], [[r.component, fx(r.value, 5), fx(r.error, 5)] for r in d.itertuples()]


def validation() -> Table:
    d = csv("validation.csv")
    keep = d[[any(k in f"{r.check} {r.quantity}" for k in ("tree N=500", "PDE M=400, N=200", "D = 0.8 x threshold tree", "D = 3 x threshold", "window length", "perpetual put boundary")) for r in d.itertuples()]]
    keep = keep[~((keep.check.str.contains("0.8 x")) & keep.quantity.str.contains("PDE"))]

    return (["check", "measured", "compared with"],
            [[f"{r.check}: {r.quantity}", ("0" if r.measured == 0 else sci(r.measured, 2) if abs(r.measured) < 0.01 or abs(r.measured) >= 1000 else fx(r.measured, 4)), str(r.compared_with)] for r in keep.itertuples()])


def model_risk() -> Table:
    d = csv("dividend_model_risk.csv")

    return (["scenario", "spot-jump reference", "PDE vs tree gap", "escrowed tree", "model difference (American)"],
            [[r.scenario, fx(r.spot_jump_reference, 4), sci(r.method_uncertainty_pde_vs_tree, 1), fx(r.escrowed_tree_N4000, 4), fx(r.model_difference_american, 3)] for r in d.itertuples()])


def cleaning() -> Table:
    log, sens = csv("step5_reject_log.csv"), csv("step5_cleaning_threshold_sensitivity.csv")
    names = ["AMZN", "SPY"]
    def rej(u, reason):
        return int(log[(log.underlying == u) & (log.reason == reason)]["count"].iloc[0])

    def left(u):
        return int(log[log.underlying == u].remaining.dropna().iloc[-1])

    def surv(u, thr):
        return int(sens[(sens.underlying == u) & (sens.max_rel_spread == thr)].surviving.iloc[0])

    rows = [["rejected: bid <= 0"] + [str(rej(u, "bid_nonpositive")) for u in names], ["rejected: spread > 10%"] + [str(rej(u, "wide_spread")) for u in names],
            ["rejected: open interest < 10"] + [str(rej(u, "low_open_interest")) for u in names], ["rejected: below intrinsic"] + [str(rej(u, "below_intrinsic")) for u in names],
            ["rejected: K/S outside [0.8, 1.2]"] + [str(rej(u, "moneyness_out_of_range")) for u in names], ["rejected: no BS IV"] + [str(rej(u, "no_bs_iv")) for u in names],
            ["contracts after cleaning"] + [str(left(u)) for u in names]] + [[f"surviving at spread {int(100 * t)}%"] + [str(surv(u, t)) for u in names] for t in (0.05, 0.10, 0.20)]

    return ["", "AMZN", "SPY"], rows


def checks() -> Table:
    d = csv("step5_hypothesis_checks.csv")

    return (["hypothesis", "quantity", "measured", "registered surprise threshold", "triggered"],
            [[r.hypothesis, r.quantity, fx(r.value, 5), r.pre_registered_surprise_threshold, "yes" if r.surprise else "no"] for r in d.itertuples()])


def bootstrap() -> Table:
    d = csv("step5_f_bootstrap.csv")

    return ["statistic", "n", "estimate", "95% CI low", "95% CI high"], [[r.statistic, str(r.n), fx(r.estimate, 4), fx(r.ci_low, 4), fx(r.ci_high, 4)] for r in d.itertuples()]


def vendor() -> Table:
    d = csv("step5_e_vendor_summary.csv")

    return (["underlying", "n", "median abs diff (vol pts)", "p95 abs diff (vol pts)", "median signed vendor - mine (vol pts)", "vendor inside my interval"],
            [[r.underlying, str(r.n), fx(100 * r.median_abs_diff_american, 3), fx(100 * r.p95_abs_diff_american, 3), fx(100 * r.median_signed_vendor_minus_american, 3), fx(r.frac_vendor_inside_my_american_interval, 2)] for r in d.itertuples()])


def discretization() -> Table:
    d = csv("diag_discretization.csv").set_index("underlying")
    spec = [("contracts (headline set)", "n", 0), ("max abs IV(1600) - IV(800), vol pts", "max_abs_iv_1600_800_vp", 4), ("median abs IV(1600) - IV(800), vol pts", "median_abs_iv_1600_800_vp", 4),
            ("max abs IV(1600) - IV(400), vol pts", "max_abs_iv_1600_400_vp", 4), ("median abs IV(1600) - IV(400), vol pts", "median_abs_iv_1600_400_vp", 4),
            ("empirical order (from medians)", "empirical_order_from_medians", 2), ("median per-contract order", "median_per_contract_order", 2),
            ("median bid-ask IV half-width, vol pts", "median_half_width_vp", 4), ("max abs IV(1600) - IV(400) / median half-width", "ratio_max_1600_400_to_median_half_width", 2),
            ("share above own half-width: 1600 vs 800", "frac_1600_800_above_own_half_width", 3), ("share above own half-width: 800 vs 400", "frac_800_400_above_own_half_width", 3),
            ("share above own half-width: 1600 vs 400", "frac_1600_400_above_own_half_width", 3),
            ("share above own half-width: 400 vs Richardson limit", "frac_400_vs_richardson_above_own_half_width", 3)]

    return ["", "AMZN", "SPY"], [[lab] + [fx(d.loc[u, col], nd) for u in ("AMZN", "SPY")] for lab, col, nd in spec]


def split() -> Table:
    d = csv("diag_dividend_split.csv")

    return (["side of ex-date", "type", "K/S", "n", "dividend model (BS - EU spot-jump)", "early exercise (EU - American)", "total @1600", "total @400"],
            [[r.side, r.cp_flag, r.mny_bucket, str(r.n), fx(r.median_div_model_1600_vp, 3), fx(r.median_early_exercise_1600_vp, 3), fx(r.median_total_1600_vp, 3), fx(r.median_total_400_vp, 3)] for r in d.itertuples()])


def parity() -> Table:
    d = csv("diag_parity.csv")

    return (["variant", "underlying", "expiry", "n", "slope ± se", "implied rate ± se", "model rate", "intercept ± se", "model escrowed spot", "fixed-slope intercept ± se", "offset vs escrowed spot, $", "vendor forward (non-null)"],
            [[r.variant, r.underlying, r.expiry, str(r.n_strikes), f"{r.free_slope:.4f} ± {r.se_free_slope:.4f}", f"{r.implied_rate:.3f} ± {r.se_implied_rate:.3f}", fx(r.model_rate, 4),
              f"{r.free_intercept:.2f} ± {r.se_free_intercept:.2f}", fx(r.model_escrowed_spot, 2), f"{r.fixed_slope_intercept:.2f} ± {r.se_fixed_slope_intercept:.2f}", fx(r.offset_vs_model_escrowed_spot_dollars, 2),
              str(r.vendor_forward_nonnull)] for r in d.itertuples()])


def bound() -> Table:
    d = csv("diag_model_free_bound.csv")
    d = d[d.by.isin(["all", "expiry", "moneyness"])]

    return (["by", "group", "calls", "ask below bound", "bid below bound", "min (ask - bound), $", "median (ask - bound), $"],
            [[r.by, r.group, str(r.n_calls), str(r.n_ask_below_bound), str(r.n_bid_below_bound), fx(r.min_ask_minus_bound, 2), fx(r.median_ask_minus_bound, 2)] for r in d.itertuples()])


def dollar_gap() -> Table:
    d = csv("diag_dollar_gap.csv")
    d = d[d.expiry == "all"]

    return (["underlying", "side", "K/S wing", "pairs", "market - model (C - P), $", "se, $", "mean abs, $", "mean sum of two half-spreads, $", "share within the half-spreads", "median call vega, $ per 1.00 vol", "median abs implied call IV shift, vol pts"],
            [[r.underlying, r.side, r.wing, str(int(r.n)), fx(r.mean_market_minus_model, 3), fx(r.se_market_minus_model, 3), fx(r.mean_abs_market_minus_model, 3), fx(r.mean_quoted_half_spread_sum, 3),
              fx(r.frac_gap_within_half_spread_sum, 2), fx(r.median_call_vega, 1), fx(r.median_abs_implied_call_iv_shift_vp, 2)] for r in d.itertuples()])


def grid_checks() -> Table:
    d = csv("diag_grid_checks.csv")

    return (["hypothesis", "quantity", "registered (M=N=400)", "M=N=1600", "Richardson", "surprise: registered / 1600 / Richardson"],
            [[r.hypothesis, r.quantity, fx(r.value_400_recomputed, 5), fx(r.value_1600, 5), fx(r.value_richardson, 5), f"{r.surprise_400_recomputed} / {r.surprise_1600} / {r.surprise_richardson}"] for r in d.itertuples()])


def snapshot_level() -> Table:
    d = csv("diag_snapshot_level.csv")

    return ["underlying", "expiry", "days", "n", "median ATM American IV (vol pts)"], [[r.underlying, r.expiry, str(r.days), str(r.n), fx(r.median_atm_iv_vp, 2)] for r in d.itertuples()]


def wings() -> Table:
    d = csv("diag_h4_wings.csv")
    a, b = d[d.grid == 400], d[d.grid == 1600].set_index(["wing", "side"])

    return (["wing", "side of ex-date", "pairs", "mean abs gap, BS (vol pts)", "mean abs gap, American @400", "American smaller @400", "American smaller @1600"],
            [[r.wing, r.side, str(r.n_pairs), fx(r.mean_abs_gap_bs_vp, 2), fx(r.mean_abs_gap_american_vp, 2), fx(r.frac_american_smaller, 2), fx(b.loc[(r.wing, r.side), "frac_american_smaller"], 2)] for r in a.itertuples()])


def offset() -> Table:
    d = csv("diag_amzn_offset.csv")

    return (["strikes", "n", "mean market - model", "mean adjusted - model", "within band, registered", "within band, offset subtracted (circular)"],
            [[r.group, str(r.n), fx(r.mean_market_minus_model, 3), fx(r.mean_adjusted_minus_model, 3), fx(r.frac_within_band_original, 3), fx(r.frac_within_band_adjusted, 3)] for r in d.itertuples()])


def pde_order() -> Table:
    d = csv("step2_table.csv")
    d = d[d.method == "penalty"].set_index("M")
    p4, p8, p16 = (float(d.loc[M, "price"]) for M in (400, 800, 1600))
    ref = float(csv("bermudan_vs_nex.csv").american_reference.iloc[0])
    order = float(np.log2((p8 - p4) / (p16 - p8)))
    rich = p16 + (p16 - p8) / (2 ** order - 1)
    rich2 = p16 + (p16 - p4) / 15.0  # exactly second order from the 400 and 1600 prices alone
    slope = float(np.polyfit(np.log(d.index.to_numpy(float)), np.log(d.error_vs_reference.abs().to_numpy(float)), 1)[0])
    rows = [["penalty PDE price, M=N=400", fx(p4, 7)], ["penalty PDE price, M=N=800", fx(p8, 7)], ["penalty PDE price, M=N=1600", fx(p16, 7)], ["observed order from the three prices", fx(order, 2)],
            ["Richardson limit at the observed order", fx(rich, 6)], ["Richardson limit assuming order 2 from M=N=400 and 1600", fx(rich2, 6)], ["tree reference price", fx(ref, 6)],
            ["abs(Richardson at observed order - tree reference)", sci(abs(rich - ref))], ["abs(Richardson at order 2 - tree reference)", sci(abs(rich2 - ref))],
            [f"log-log slope of the price error, M=N={int(d.index.min())}..{int(d.index.max())}", fx(slope, 2)]]

    return ["quantity", "value"], rows


def rannacher() -> Table:
    d = csv("rannacher_gamma_roughness.csv")
    on, off = d[d.rannacher].set_index("N"), d[~d.rannacher].set_index("N")

    return ["N (M = 400)", "gamma roughness, Rannacher on", "gamma roughness, Rannacher off"], [[str(n), sci(on.loc[n, "gamma_roughness"]), sci(off.loc[n, "gamma_roughness"])] for n in on.index]


def lsm_basis() -> Table:
    d = csv("lsm_basis_sweep.csv")

    return (["basis", "degree", "in-sample bias", "out-of-sample bias (fresh evaluation)", "paired policy gain vs Laguerre degree 3 ± se"],
            [[r.variant, str(r.degree), fx(r.in_sample_bias, 4), fx(r.out_of_sample_bias_independent_eval, 4), f"{r.paired_policy_gain_vs_laguerre3:+.4f} ± {r.paired_gain_se:.4f}"] for r in d.itertuples()])


def boundary_gap() -> Table:
    d = csv("boundary_gap.csv")
    
    return ["comparison (0.3 < t < 0.9)", "mean gap", "mean abs gap"], [[r.comparison, fx(float(r.mean_gap), 3), fx(r.mean_abs_gap, 3)] for r in d.itertuples()]


def boundary_shift() -> Table:
    d = csv("boundary_shift.csv")
    
    return (["n_ex", "measured Bermudan/American ratio", "ratio predicted with 0.5826", "implied constant", "share of predicted shift"],
            [[str(r.n_ex), fx(r.measured_ratio, 4), fx(r.predicted_ratio_bgk, 4), fx(r.implied_beta, 3), fx(r.share_of_predicted_shift, 2)] for r in d.itertuples()])


def put_split() -> Table:
    d = csv("dividend_put_premium_split.csv")
    
    return (["D", "total put premium", "exercise only up to the ex-date", "exercise only after the ex-date"], [[fx(r.D, 2), fx(r.premium_total, 4), fx(r.premium_exercise_only_up_to_ex_date, 4), fx(r.premium_exercise_only_after_ex_date, 4)] for r in d.itertuples()])


def discrete_cont() -> Table:
    d = csv("dividend_discrete_vs_continuous.csv")
    
    return ["model", "American put", "European put", "premium", "yield q"], [[r.model, fx(r.american, 4), fx(r.european, 4), fx(r.premium, 4), fx(r.yield_q, 4)] for r in d.itertuples()]


def dividend_lsm() -> Table:
    d = csv("dividend_lsm_call_basis.csv")
    
    return (["basis", "degree", "mean out-of-sample minus Bermudan", "sd across seeds", "share of early exercises off the ex-date"],
            [[r.basis, str(r.degree), fx(r.mean_oos_minus_bermudan, 4), fx(r.sd_across_seeds, 4), fx(r.share_of_early_exercises_off_ex_date, 2)] for r in d.itertuples()])


def figure_provenance(name: str) -> str:
    """Provenance text stamped on a figure by figstyle.save (N, snapshot date, source), read back from figure_manifest.csv."""
    d = pd.read_csv(ROOT / "results" / "figures" / "figure_manifest.csv").set_index("figure")

    return f"*Provenance (generated from `results/figures/figure_manifest.csv`, the text printed on the figure):* {d.loc[name, 'provenance']}"


CAPTIONS = ["step5_a_smiles_AMZN", "step5_a_smiles_SPY", "step5_a_difference_AMZN", "step5_a_difference_SPY", "step5_c_call_put_gap_SPY", "step6_amzn_premium_two_row"]


TABLES: dict[str, Callable[[], Table]] = {"pde_domain": pde_domain, "bermudan_nex": bermudan_nex, "lsm_paths": lsm_paths, "bias_decomposition": bias_decomposition, "validation": validation, "model_risk": model_risk, "cleaning": cleaning, "checks": checks, "bootstrap": bootstrap, "vendor": vendor, "discretization": discretization, "split": split, "parity": parity,
          "bound": bound, "dollar_gap": dollar_gap, "wings": wings, "offset": offset, "grid_checks": grid_checks, "snapshot_level": snapshot_level,
          "pde_order": pde_order, "rannacher": rannacher, "lsm_basis": lsm_basis, "boundary_gap": boundary_gap, "boundary_shift": boundary_shift, "put_split": put_split, "discrete_cont": discrete_cont, "dividend_lsm": dividend_lsm}


def render(header: list[str], rows: list[list[str]]) -> str:
    esc = lambda cells: [x.replace("|", "\\|") for x in cells]  # a literal pipe inside a cell must be escaped in a markdown table
    lines = ["| " + " | ".join(esc(header)) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(esc(r)) + " |" for r in rows]

    return "\n".join(lines)