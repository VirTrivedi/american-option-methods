"""Numbers quoted in README prose and the CSV cells they come from (see audit_readme_numbers.py). Each regex must match the README exactly once;
capture groups are the numbers, in the units the README displays."""
from __future__ import annotations
import pandas as pd
from audit_readme_numbers import Claim
from readme_tables import TAB


def c(name: str) -> pd.DataFrame:
    return pd.read_csv(TAB / name)


def hyp(key: str) -> float:
    d = c("step5_hypothesis_checks.csv")

    return float(d[d.quantity.str.contains(key, regex=False)].value.iloc[0])


def boot(key: str) -> pd.Series:
    d = c("step5_f_bootstrap.csv")
    
    return d[d.statistic == key].iloc[0]


def vend(u: str, col: str) -> float:
    d = c("step5_e_vendor_summary.csv")
    
    return float(d[d.underlying == u][col].iloc[0])


def disc(u: str, col: str) -> float:
    return float(c("diag_discretization.csv").set_index("underlying").loc[u, col])


def sens(scen: str, stat: str) -> float:
    d = c("step5_d_sensitivity.csv")
    
    return float(d[(d.scenario == scen) & (d.statistic == stat)].change_relative_to_model_effect.iloc[0])


def both(stat: str, scens: tuple[str, ...]) -> list[float]:
    v = [sens(s, stat) for s in scens]
    
    return [sum(v) / len(v)] if len(v) else v


PREM = "AMZN: mean (market - model) put premium"
GAP = "SPY: mean |call-put IV gap| American"
R, SPOT, DIV = ("r -50bp", "r +50bp"), ("spot -0.05%", "spot +0.05%"), ("dividends x0.9", "dividends x1.1")


def wing_all(col: str, side: str = "all", grid: int = 400) -> float:
    d = c("diag_h4_wings.csv")
    
    return float(d[(d.wing == "all wings") & (d.side == side) & (d.grid == grid)][col].iloc[0])


def par(u: str, variant: str, col: str = "offset_vs_model_escrowed_spot_dollars") -> pd.Series:
    d = c("diag_parity.csv")
    
    return d[(d.underlying == u) & d.variant.str.startswith(variant)][col]


def par_all(variant: str, col: str) -> pd.Series:
    d = c("diag_parity.csv")
    
    return d[d.variant.str.startswith(variant)][col]


def pde(label: str) -> float:
    from readme_tables import pde_order
    rows = dict(pde_order()[1])
    return float(rows[label])


def grid(quantity: str, col: str) -> float:
    d = c("diag_grid_checks.csv")
    
    return float(d[d.quantity.str.contains(quantity, regex=False)][col].iloc[0])


def dol(u: str, side: str, wing: str, col: str) -> float:
    d = c("diag_dollar_gap.csv")
    
    return float(d[(d.underlying == u) & (d.side == side) & (d.wing == wing) & (d.expiry == "all")][col].iloc[0])


def wing_row(wing: str, col: str, side: str = "all", grid_: int = 400) -> float:
    d = c("diag_h4_wings.csv")
    
    return float(d[(d.wing == wing) & (d.side == side) & (d.grid == grid_)][col].iloc[0])


CLAIMS = [
    Claim("discretization headline", r"from M=N=400 to 1600 is ([\d.]+) vol points, ([\d.]+) times the median bid-ask half-width of ([\d.]+),",
          lambda: [disc("SPY", "max_abs_iv_1600_400_vp"), disc("SPY", "ratio_max_1600_400_to_median_half_width"), disc("SPY", "median_half_width_vp")]),
    Claim("discretization share", r"([\d.]+)% of SPY contracts change by more", lambda: [100 * disc("SPY", "frac_1600_400_above_own_half_width")]),
    Claim("H1 bullet", r"Median gap at the money ([\d.]+) vol points \(AMZN\) and ([\d.]+) \(SPY\); for K/S <= 0\.95, ([\d.]+) and ([\d.]+);",
          lambda: [100 * hyp("AMZN median put diff, ATM"), 100 * hyp("SPY median put diff, ATM"), 100 * hyp("AMZN median put diff, K/S <= 0.95"), 100 * hyp("SPY median put diff, K/S <= 0.95")]),
    Claim("H1 fractions", r"exceeds the American half-width is ([\d.]+) for AMZN and ([\d.]+) for SPY", lambda: [boot("AMZN: fraction |BS - American| > half-width (puts)").estimate, boot("SPY: fraction |BS - American| > half-width (puts)").estimate]),
    Claim("H2 bullet", r"Maximum \|BS - American\| ([\d.]+) vol points for 0\.9", lambda: [100 * hyp("AMZN max |BS - American|")]),
    Claim("H3 bullet", r"market premium averages ([\d.]+), the model premium ([\d.]+), and only ([\d.]+)% of strikes",
          lambda: [c("step5_b_premium_summary_AMZN.csv").query("by == 'all'").mean_market_premium.iloc[0], c("step5_b_premium_summary_AMZN.csv").query("by == 'all'").mean_model_premium.iloc[0],
                   100 * c("step5_b_premium_summary_AMZN.csv").query("by == 'all'").frac_within_band.iloc[0]]),
    Claim("H4 bullet", r"is ([\d.]+) vol points under the American model and ([\d.]+) under BS; the American gap is smaller for ([\d.]+)% of the (\d+) pairs",
          lambda: [wing_all("mean_abs_gap_american_vp"), wing_all("mean_abs_gap_bs_vp"), 100 * wing_all("frac_american_smaller"), wing_all("n_pairs")]),
    Claim("H5 bullet", r"Median \|my American IV - vendor\| is ([\d.]+) vol points \(AMZN\) and ([\d.]+) \(SPY\)", lambda: [100 * vend("AMZN", "median_abs_diff_american"), 100 * vend("SPY", "median_abs_diff_american")]),
    Claim("H5 interval", r"inside my American bid-ask IV interval for ([\d.]+)% of AMZN and ([\d.]+)% of SPY", lambda: [100 * vend("AMZN", "frac_vendor_inside_my_american_interval"), 100 * vend("SPY", "frac_vendor_inside_my_american_interval")]),
    Claim("sensitivity AMZN", r"offset by ([\d.]+) times the model premium, a 0\.05% spot shift by ([\d.]+) to ([\d.]+) times",
          lambda: both(PREM, R) + [min(sens(s, PREM) for s in SPOT), max(sens(s, PREM) for s in SPOT)]),
    Claim("sensitivity SPY", r"by ([\d.]+) \(r\) and ([\d.]+) to ([\d.]+) \(spot\)", lambda: both(GAP, R) + [min(sens(s, GAP) for s in SPOT), max(sens(s, GAP) for s in SPOT)]),
    Claim("sensitivity dividends", r"gap by ([\d.]+) to ([\d.]+) times\.", lambda: [min(sens(s, GAP) for s in DIV), max(sens(s, GAP) for s in DIV)]),
    Claim("Q4.1 registered grid", r"changes by at most ([\d.]+) vol points for SPY against a median half-width of ([\d.]+),", lambda: [disc("SPY", "max_abs_iv_1600_400_vp"), disc("SPY", "median_half_width_vp")]),
    Claim("Q4.1 fine grids", r"differ by at most ([\d.]+) \(([\d.]+) of the median half-width\)", lambda: [disc("SPY", "max_abs_iv_1600_800_vp"), disc("SPY", "max_abs_iv_1600_800_vp") / disc("SPY", "median_half_width_vp")]),
    Claim("Q4.2 call early exercise", r"K/S 0\.9-0\.95 after the ex-date \(about ([\d.]+) vol points\)",
          lambda: [c("diag_dividend_split.csv").query("side == 'after ex-date' and cp_flag == 'call' and mny_bucket == '(0.9, 0.95]'").median_early_exercise_1600_vp.iloc[0]]),
    Claim("Q4.4 bound", r"\(minimum slack \$([\d.]+)\)", lambda: [c("diag_model_free_bound.csv").query("by == 'all'").min_ask_minus_bound.iloc[0]]),
    Claim("Q4.4 bids", r"(\d+) bids are below it", lambda: [c("diag_model_free_bound.csv").query("by == 'all'").n_bid_below_bound.iloc[0]]),
    Claim("Q4.6 offset", r"13 strikes with K/S < 0\.9, where the model premium is about zero: ([\d.]+) \(interquartile ([\d.]+) to ([\d.]+)\)",
          lambda: [c("diag_amzn_offset.csv").offset_used.iloc[0], c("diag_amzn_offset.csv").offset_iqr_low.iloc[0], c("diag_amzn_offset.csv").offset_iqr_high.iloc[0]]),
    Claim("Q4.6 band", r"inside the propagated band from ([\d.]+) to ([\d.]+)\.", lambda: [c("diag_amzn_offset.csv").frac_within_band_original.iloc[0], c("diag_amzn_offset.csv").frac_within_band_adjusted.iloc[0]]),
    Claim("Q4.1 order and half-width", r"empirical order from the medians is ([\d.]+) \(AMZN\) and ([\d.]+) \(SPY\)\. The median half-width is ([\d.]+) vol points for AMZN and ([\d.]+) for SPY",
          lambda: [disc("AMZN", "empirical_order_from_medians"), disc("SPY", "empirical_order_from_medians"), disc("AMZN", "median_half_width_vp"), disc("SPY", "median_half_width_vp")]),
    Claim("Q4.1 grid method row", r"largest change ([\d.]+) at 400 vs 1600, ([\d.]+) at 1600 vs 800, against a median half-width of ([\d.]+)\)",
          lambda: [grid("max IV change", "value_400_recomputed"), grid("max IV change", "value_1600"), disc("SPY", "median_half_width_vp") / 100]),
    Claim("Q4.1 H2 grid", r"The H2 maximum falls from ([\d.]+) to ([\d.]+) vol points", lambda: [100 * grid("max |BS - American|", "value_400_recomputed"), 100 * grid("max |BS - American|", "value_1600")]),
    Claim("Q4.1 one-line caveat", r"grid error of up to about ([\d.]+) vol points", lambda: [disc("SPY", "max_abs_iv_1600_400_vp")]),
    Claim("Q4.3 raw slopes", r"every slope is below -1 \((-[\d.]+) to (-[\d.]+)\)", lambda: [par_all("registered", "free_slope").max(), par_all("registered", "free_slope").min()]),
    Claim("Q4.3 corrected slopes", r"the slopes are (-[\d.]+) to (-[\d.]+) \(premium subtracted; the negligible-premium fits are noisier, with only (\d+) to (\d+) strikes\)",
          lambda: [par_all("model premium subtracted", "free_slope").min(), par_all("model premium subtracted", "free_slope").max(), par_all("model premium negligible", "n_strikes").min(), par_all("model premium negligible", "n_strikes").max()]),
    Claim("Q4.3 corrected rates", r"implied rates are no longer systematically negative, \((-[\d.]+) to ([\d.]+)\) against model rates of ([\d.]+) to ([\d.]+)\.",
          lambda: [par_all("model premium subtracted", "implied_rate").min(), par_all("model premium subtracted", "implied_rate").max(), par_all("model premium subtracted", "model_rate").min(), par_all("model premium subtracted", "model_rate").max()]),
    Claim("Q4.3 SPY rates", r"implied SPY rates \(([\d.]+) to ([\d.]+)\)", lambda: [par("SPY", "model premium subtracted", "implied_rate").min(), par("SPY", "model premium subtracted", "implied_rate").max()]),
    Claim("Q4.3 offsets", r"is -\$([\d.]+) to -\$([\d.]+) for AMZN in the premium-subtracted fits \(-\$([\d.]+) to -\$([\d.]+) in the raw fits\) and -\$([\d.]+) to -\$([\d.]+) for SPY \(-\$([\d.]+) to -\$([\d.]+) raw\)",
          lambda: [x for u in ("AMZN", "SPY") for v in ("model premium subtracted", "registered") for x in (par(u, v).abs().min(), par(u, v).abs().max())]),
    Claim("Q4.3 SPY pre-ex-date", r"gives -\$([\d.]+) in the negligible-premium and subtracted fits \(-\$([\d.]+) in the raw fit\)",
          lambda: [par("SPY", "model premium subtracted").abs().iloc[0], par("SPY", "registered").abs().iloc[0]]),
    Claim("Q4.4 deep ITM", r"mean dollar gap is \$([\d.]+) against a mean sum of half-spreads of \$([\d.]+), and (\d+)% of those pairs lie inside their quoted half-spreads, yet the same dollars are a median ([\d.]+) vol points because the call vega is only about (\d+)\.",
          lambda: [abs(dol("SPY", "after ex-date", "K/S < 0.93", "mean_market_minus_model")), dol("SPY", "after ex-date", "K/S < 0.93", "mean_quoted_half_spread_sum"), 100 * dol("SPY", "after ex-date", "K/S < 0.93", "frac_gap_within_half_spread_sum"),
                   dol("SPY", "after ex-date", "K/S < 0.93", "median_abs_implied_call_iv_shift_vp"), dol("SPY", "after ex-date", "K/S < 0.93", "median_call_vega")]),
    Claim("Q4.4 H4 deep ITM", r"\(American ([\d.]+) vs BS ([\d.]+) vol points\) is therefore", lambda: [wing_row("K/S < 0.93", "mean_abs_gap_american_vp"), wing_row("K/S < 0.93", "mean_abs_gap_bs_vp")]),
    Claim("Q4.4 middle wing", r"The gap is \$([\d.]+) after the ex-date and \$([\d.]+) before it, about (\d+) to (\d+) times the sum",
          lambda: [abs(dol("SPY", "after ex-date", "0.93 <= K/S <= 1.03", "mean_market_minus_model")), abs(dol("SPY", "before ex-date", "0.93 <= K/S <= 1.03", "mean_market_minus_model")),
                   dol("SPY", "after ex-date", "0.93 <= K/S <= 1.03", "median_abs_gap_over_half_spread_sum"), dol("SPY", "before ex-date", "0.93 <= K/S <= 1.03", "median_abs_gap_over_half_spread_sum")]),
    Claim("Q4.4 constant gap", r"\(about \$([\d.]+) for SPY before the ex-date, \$([\d.]+) to \$([\d.]+) after it,", lambda: [abs(dol("SPY", "before ex-date", "0.93 <= K/S <= 1.03", "mean_market_minus_model"))] + [min(abs(dol("SPY", "after ex-date", w, "mean_market_minus_model")) for w in ("K/S < 0.93", "0.93 <= K/S <= 1.03", "K/S > 1.03")), max(abs(dol("SPY", "after ex-date", w, "mean_market_minus_model")) for w in ("K/S < 0.93", "0.93 <= K/S <= 1.03", "K/S > 1.03"))]),
    Claim("Q4.4 dividend pv", r"present value ([\d.]+) at the snapshot", lambda: [par("SPY", "registered", "pv_dividends").min() if False else c("diag_parity.csv").query("underlying == 'SPY' and pv_dividends > 0").pv_dividends.min()]),
    Claim("Q1 PDE vs tree", r"observed order about ([\d.]+), agreeing with the tree reference to ([\d.]+e-\d+) after Richardson",
          lambda: [pde("observed order from the three prices"), pde("abs(Richardson at observed order - tree reference)")]),
    Claim("LSM common-set note", r"that set's own standard error \(([\d.]+)\) to every out-of-sample level", lambda: [c("bias_decomposition.csv").query("component.str.contains('ONE run')").value.iloc[0]]),
    Claim("finding 1", r"at (6\.0904), and the PDE error", lambda: [c("bermudan_vs_nex.csv").american_reference.iloc[0]]),
    Claim("finding 2", r"low by ([\d.]+) ± ([\d.]+) with a million paths\. Restricting exercise to those dates costs another ([\d.]+)\.",
          lambda: [c("bias_decomposition.csv").value.iloc[1], c("bias_decomposition.csv").error.iloc[1], c("bias_decomposition.csv").value.iloc[0]]),
    Claim("finding 3", r"moves the put price by ([\d.]+)\. The gap between the PDE and the tree is about ([\d.]+e-\d+)\.",
          lambda: [abs(c("dividend_model_risk.csv").model_difference_american.iloc[0]), c("dividend_model_risk.csv").method_uncertainty_pde_vs_tree.iloc[0]]),
    Claim("finding 4", r"differ by about ([\d.]+) vol points at the money, as I predicted, and my American IVs match OptionMetrics' to a median of ([\d.]+) to ([\d.]+) vol points",
          lambda: [50 * (hyp("AMZN median put diff, ATM") + hyp("SPY median put diff, ATM")), 100 * vend("AMZN", "median_abs_diff_american"), 100 * vend("SPY", "median_abs_diff_american")]),
    Claim("finding 5", r"about \$([\d.]+) above the model's, and SPY's call-put IV gap is larger under the American model \(([\d.]+) vol points\) than under Black-Scholes \(([\d.]+)\)",
          lambda: [boot(PREM).estimate, 100 * boot("SPY: mean |call-put gap| American").estimate, 100 * boot("SPY: mean |call-put gap| BS").estimate]),
]