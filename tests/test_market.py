"""Market-data layer on synthetic data only: no network, no vendor data. Tolerances are set from measurements."""
from datetime import date, datetime
from pathlib import Path
import math
import numpy as np
import pandas as pd
import pytest
from aopt.bs import bs_price
from aopt.market.clean import CleanConfig, add_model_inputs, clean_quotes, contract_of
from aopt.market.dividends import UnsupportedDistribution, dividends_for_expiry, load_distributions
from aopt.market.iv import Contract, american_iv, bs_iv, bs_vega, iv_interval
from aopt.market.loader import build_quotes, load_option_prices, underlying_close
from aopt.market.rates import ZeroCurve, curve_from_zerocd
from aopt.market.schema import QUOTE_COLUMNS, validate_quotes
from aopt.market.synthetic import distributions_table, make_chain
from aopt.market.timeutil import expiry_timestamp, year_fraction

FIXTURES = Path(__file__).parent / "fixtures"
QT = datetime(2024, 3, 15, 16, 0)
CURVE = ZeroCurve(np.array([30.0, 365.0]), np.array([0.045, 0.05]))
DIV = ((50, 1.0), (140, 1.0))
GRID_TOL = 5e-4  # chains are priced at M=N=800 and inverted at 400; measured worst IV error 3.0e-4 (no noise)


@pytest.fixture(scope="module")
def clean_chain():
    return make_chain(strikes_over_spot=(0.9, 1.0, 1.1), expiry_days=(35, 182), curve=CURVE)


@pytest.fixture(scope="module")
def div_chain():
    return make_chain(strikes_over_spot=(0.9, 1.0, 1.1), expiry_days=(35, 182), curve=CURVE, dividends_abs=DIV)


def inputs(chain, div=()):
    return add_model_inputs(chain, CURVE, distributions_table(QT, div) if div else load_distributions(pd.DataFrame(), None))


@pytest.mark.parametrize("div", [(), DIV], ids=["no_dividends", "dividends"])
def test_american_iv_recovers_true_vol_within_noise(div):
    noise_rel, tick = 0.002, 0.01
    chain = make_chain(strikes_over_spot=(0.9, 0.95, 1.0, 1.05, 1.1), expiry_days=(35, 182), curve=CURVE, dividends_abs=div, rel_spread=0.02,
                       noise_rel=noise_rel, tick=tick, seed=3)
    d = inputs(chain, div)
    worst = 0.0
    for _, row in d.iterrows():
        c = contract_of(row)
        vega = bs_vega(c, row["true_iv"])
        if vega < 1.0:  # IV is not identified from a price this insensitive to vol
            continue
        iv, why = american_iv(row["mid"], c)
        assert why == "ok"
        allowed = (4 * noise_rel * row["true_mid"] + tick) / vega + GRID_TOL  # 4-sigma price noise plus tick rounding, mapped through vega
        worst = max(worst, abs(iv - row["true_iv"]) / allowed)
        assert abs(iv - row["true_iv"]) < allowed
    print(f"[iv recovery] div={bool(div)}: worst |error| / allowed = {worst:.2f}")


def test_zero_noise_iv_grid_error_is_small(clean_chain, div_chain):
    errs = []
    for chain, div in ((clean_chain, ()), (div_chain, DIV)):
        for _, row in inputs(chain, div).iterrows():
            iv, why = american_iv(row["true_mid"], contract_of(row))
            if why == "ok":
                errs.append(abs(iv - row["true_iv"]))
    print(f"[iv grid] max |american_iv - true| = {max(errs):.2e} over {len(errs)} contracts")
    assert max(errs) < GRID_TOL


def test_bs_iv_at_least_american_iv_for_puts_and_equal_for_calls_without_dividends(clean_chain):
    d = inputs(clean_chain)
    put_gap, call_gap = [], []
    for _, row in d.iterrows():
        c = contract_of(row)
        am, _ = american_iv(row["true_mid"], c)
        bs, _ = bs_iv(row["true_mid"], c)
        if not (math.isfinite(am) and math.isfinite(bs)):
            continue
        (put_gap if row["cp_flag"] == "P" else call_gap).append(bs - am)
    print(f"[bs vs american] puts: min {min(put_gap):+.2e} max {max(put_gap):+.2e}; calls: max |gap| {max(map(abs, call_gap)):.2e}")
    assert min(put_gap) > -GRID_TOL  # BS IV >= American IV (an American put is worth more, so it needs less vol)
    assert max(put_gap) > 5e-3  # and the gap is real for ITM puts, not just grid noise
    assert max(map(abs, call_gap)) < GRID_TOL  # q = 0, no dividends: American call = European call


def test_parity_premium_equals_model_premium_on_zero_noise_chain(clean_chain):
    d = inputs(clean_chain)
    diffs = []
    for (_, _), g in d.groupby(["expiry", "strike"]):
        c_row, p_row = g[g.cp_flag == "C"].iloc[0], g[g.cp_flag == "P"].iloc[0]
        cc, pc = contract_of(c_row), contract_of(p_row)
        market = p_row["true_mid"] - (c_row["true_mid"] - cc.S + cc.K * math.exp(-cc.r * cc.T))
        sigma_c, _ = american_iv(c_row["true_mid"], cc)
        model = american_iv_price(pc, sigma_c) - float(bs_price(pc.S, pc.K, pc.T, pc.r, 0.0, sigma_c, "put"))
        diffs.append(market - model)
    print(f"[parity premium] market - model: max |diff| = {max(map(abs, diffs)):.2e} (premiums are O(0.1-1))")
    assert max(map(abs, diffs)) < 6e-3  # measured 3.8e-3: the M=N=400 vs 800 PDE grid error of the put premium; premiums are 0.1-1


def american_iv_price(c: Contract, sigma: float) -> float:
    from aopt.pde import solve_pde
    return solve_pde(c.params(sigma), 400, 400, american_method="penalty").price


def test_bs_iv_reason_codes():
    c = Contract(100.0, 100.0, 0.5, 0.05, "P")
    assert bs_iv(-1.0, c)[1] == "nonpositive_price"
    itm = Contract(100.0, 80.0, 0.5, 0.05, "C")
    assert bs_iv(5.0, itm)[1] == "price_below_intrinsic"  # below the European lower bound S - K e^{-rT}
    assert bs_iv(0.1, Contract(100.0, 102.53, 0.5, 0.05, "C"))[1] == "price_below_min_vol_value"  # an at-the-forward option is worth ~0.28 even at 1% vol
    assert bs_iv(90.0, c)[1] == "price_above_max_vol_value"  # a put at sigma = 3 is worth only ~69
    iv, why = bs_iv(float(bs_price(100.0, 100.0, 0.5, 0.05, 0.0, 0.3, "put")), c)
    assert why == "ok" and iv == pytest.approx(0.3, abs=1e-6)
    assert american_iv(0.0, c)[1] == "nonpositive_price"


def test_iv_interval_orders_bid_mid_ask():
    c = Contract(100.0, 100.0, 0.5, 0.05, "P")
    r = iv_interval("american", 4.0, 4.1, 4.2, c)
    assert r["iv_bid"] < r["iv_mid"] < r["iv_ask"] and r["half_width"] == pytest.approx(0.5 * (r["iv_ask"] - r["iv_bid"]))


def injected(chain, mutate):
    d = inputs(chain.iloc[:12]).copy()
    row = d.index[5]
    mutate(d, row)
    d["mid"] = 0.5 * (d["bid"] + d["ask"])
    return d, row


@pytest.mark.parametrize("name,mutate,reason", [
    ("zero bid", lambda d, i: d.__setitem__("bid", d["bid"].where(d.index != i, 0.0)), "bid_nonpositive"),
    ("crossed", lambda d, i: d.__setitem__("ask", d["ask"].where(d.index != i, d.loc[i, "bid"] - 0.05)), "crossed_market"),
    ("wide spread", lambda d, i: d.__setitem__("ask", d["ask"].where(d.index != i, d.loc[i, "bid"] * 2)), "wide_spread"),
    ("low open interest", lambda d, i: d.__setitem__("open_interest", d["open_interest"].where(d.index != i, 1.0)), "low_open_interest"),
    ("AM settled", lambda d, i: d.__setitem__("am_settlement", d["am_settlement"].where(d.index != i, True)), "am_settled"),
])
def test_cleaner_rejects_injected_quote_with_right_reason(clean_chain, name, mutate, reason):
    d, row = injected(clean_chain, mutate)
    kept, log = clean_quotes(d)
    counts = dict(zip(log["reason"], log["count"]))
    assert counts[reason] >= 1, (name, counts)
    assert row not in kept.index
    assert len(kept) < len(d)


def test_cleaner_rejects_below_intrinsic_and_above_upper_bound(clean_chain):
    d = inputs(clean_chain.iloc[:12]).copy()
    itm_put = d[(d.cp_flag == "P") & (d.strike > d.spot)].index[0]
    otm_call = d[(d.cp_flag == "C")].index[0]
    d.loc[itm_put, ["bid", "ask"]] = [d.loc[itm_put, "strike"] - d.loc[itm_put, "spot"] - 1.1, d.loc[itm_put, "strike"] - d.loc[itm_put, "spot"] - 0.9]
    d.loc[otm_call, ["bid", "ask"]] = [d.loc[otm_call, "spot"] + 1.0, d.loc[otm_call, "spot"] + 1.2]
    d["mid"] = 0.5 * (d["bid"] + d["ask"])
    # make the call quote wide-spread-free and not below intrinsic so only the upper bound can catch it
    kept, log = clean_quotes(d, CleanConfig(max_rel_spread=10.0))
    counts = dict(zip(log["reason"], log["count"]))
    assert counts["below_intrinsic"] >= 1 and counts["above_upper_bound"] >= 1
    assert itm_put not in kept.index and otm_call not in kept.index


def test_cleaner_maturity_moneyness_and_waterfall_adds_up(clean_chain):
    d = inputs(clean_chain)
    kept, log = clean_quotes(d, CleanConfig(min_days=40.0, min_moneyness=0.95, max_moneyness=1.05))
    counts = dict(zip(log["reason"], log["count"]))
    assert counts["maturity_out_of_range"] == (d["days"] < 40).sum()
    assert counts["moneyness_out_of_range"] > 0
    assert len(d) - log["count"].sum() == len(kept)
    assert log["remaining"].iloc[-1] == len(kept)


def test_rate_interpolation_linear_in_days_flat_outside_and_shift():
    zc = ZeroCurve(np.array([30.0, 90.0, 365.0]), np.array([0.04, 0.05, 0.06]))
    assert zc.rate_at(60.0) == pytest.approx(0.045)
    assert zc.rate_at(227.5) == pytest.approx(0.055)
    assert zc.rate_at(5.0) == pytest.approx(0.04) and zc.rate_at(1000.0) == pytest.approx(0.06)
    assert zc.shifted(50).rate_at(60.0) == pytest.approx(0.05)


def test_zero_curve_units_are_asserted():
    ok = pd.DataFrame({"date": ["2024-03-15"] * 3, "days": [30, 90, 365], "rate": [5.2, 5.1, 4.9]})
    assert curve_from_zerocd(ok).rate_at(30.0) == pytest.approx(0.052)
    for bad_rate in ([0.052, 0.051, 0.049], [52.0, 51.0, 49.0]):  # decimals / implausibly large
        with pytest.raises(ValueError):
            curve_from_zerocd(ok.assign(rate=bad_rate))
    with pytest.raises(ValueError):
        curve_from_zerocd(ok.drop(columns="rate"))


def test_time_to_expiry_act365_and_am_settlement():
    assert year_fraction(QT, expiry_timestamp(date(2024, 4, 19), False)) == pytest.approx(35 / 365)
    am = year_fraction(QT, expiry_timestamp(date(2024, 4, 19), True))
    assert am == pytest.approx(35 / 365 - 6.5 / 24 / 365)


def test_dividend_mapping_and_unsupported_types():
    raw = pd.DataFrame({"ex_date": ["2024-05-03", "2024-08-02", "2024-03-01"], "amount": [0.8, 0.8, 0.5], "distr_type": ["1", "1", "1"]})
    table = load_distributions(raw, {"1"})
    assert list(table["is_forecast"]) == [False] * 3
    got = dividends_for_expiry(table, QT, expiry_timestamp(date(2024, 7, 12), False))  # only 2024-05-03 is after the quote and before expiry
    assert len(got) == 1
    assert got[0][0] == pytest.approx((datetime(2024, 5, 3, 9, 30) - QT).total_seconds() / (365 * 86400))
    assert dividends_for_expiry(table, QT, expiry_timestamp(date(2024, 7, 12), False), scale=1.1)[0][1] == pytest.approx(0.88)
    assert dividends_for_expiry(table, QT, expiry_timestamp(date(2024, 5, 3), True)) == ()  # AM expiry at the ex-date open: no drop before it
    with pytest.raises(UnsupportedDistribution, match="not confirmed as cash"):
        load_distributions(raw.assign(distr_type=["1", "5", "1"]), {"1"})
    with pytest.raises(UnsupportedDistribution):
        load_distributions(raw, None)  # no confirmed cash codes: stop and ask
    with pytest.raises(UnsupportedDistribution, match="cancel_flag"):
        load_distributions(raw.assign(cancel_flag=[0, 1, 0]), {"1"})
    assert len(load_distributions(raw.assign(cancel_flag=[0, 0, 0], approx_flag=[0.0, 0.0, 0.0]), {"1"})) == 3
    assert load_distributions(pd.DataFrame(), None).empty  # an empty table means no dividends


def fake_opprcd(**over) -> pd.DataFrame:
    base = dict(secid=1, date="2024-03-15", exdate=["2024-04-19", "2024-04-19", "2024-04-19", "2024-04-19"], cp_flag=["C", "P", "C", "P"],
                strike_price=[100000, 100000, 105000, 105000], best_bid=[3.0, 2.8, 1.0, 6.0], best_offer=[3.1, 2.9, 1.1, 6.2], volume=[10, 10, 10, 10],
                open_interest=[100, 100, 100, 100], impl_volatility=[0.25, -99.99, 0.24, 0.26], delta=[0.5, -99.99, 0.3, -0.7],
                ss_flag=[0, 0, 0, 0], am_settlement=[0, 0, 1, 0], contract_size=[100, 100, 100, 100],
                cfadj=[1.0, 1.0, 1.0, 1.0])
    base.update(over)
    return pd.DataFrame(base)


INFO = pd.DataFrame({"secid": [1], "div_convention": [1], "exercise_style": ["A"], "am_set_flag": [0]})


def test_loader_scales_strikes_handles_sentinels_and_flags_am_settlement():
    q, rej = build_quotes(load_option_prices(fake_opprcd(), INFO), 101.0, "X")
    validate_quotes(q)
    assert list(q["strike"]) == [100.0, 100.0, 105.0, 105.0]
    assert q["vendor_iv"].isna().tolist() == [False, True, False, False] and q["vendor_delta"].isna().tolist() == [False, True, False, False]
    assert q["am_settlement"].tolist() == [False, False, True, False] and q["am_settlement"].dtype == bool  # flagged, not dropped
    assert sum(rej.values()) == 0 and list(q.columns) == QUOTE_COLUMNS


def test_loader_applies_schema_flags_and_counts_rejects():
    d = fake_opprcd(ss_flag=[0, 1, 0, 0], cfadj=[1.0, 1.0, 0.5, 1.0], contract_size=[100, 100, 100, 50])
    q, rej = build_quotes(load_option_prices(d, INFO), 101.0, "X")
    assert rej == {"nonstandard_settlement": 1, "adjusted_contract": 1, "nonstandard_contract_size": 1} and len(q) == 1


@pytest.mark.parametrize("bad", [
    dict(strike_price=[100.5, 100000, 105000, 105000]),  # not integer-valued: scaling assumption broken
    dict(impl_volatility=[0.25, -5.0, 0.24, 0.26]),  # negative, not the -99.99 sentinel
    dict(impl_volatility=[25.0, -99.99, 0.24, 0.26]),  # percent instead of decimal
    dict(ss_flag=[0, 2, 0, 0]),
    dict(delta=[1.5, -99.99, 0.3, -0.7]),
])
def test_loader_fails_loudly_on_violated_assumptions(bad):
    with pytest.raises(ValueError):
        load_option_prices(fake_opprcd(**bad), INFO)
    for style in ("X", "AE"):  # exercise style comes from opinfd and must be exactly A or E
        with pytest.raises(ValueError, match="exercise_style"):
            load_option_prices(fake_opprcd(), INFO.assign(exercise_style=style))


def test_loader_fails_on_missing_column_wrong_scaling_and_bad_spot():
    with pytest.raises(ValueError, match="missing columns"):
        load_option_prices(fake_opprcd().drop(columns=["cfadj"]), INFO)
    with pytest.raises(ValueError, match="scaling"):
        build_quotes(load_option_prices(fake_opprcd(strike_price=[100, 100, 105, 105]), INFO), 101.0, "X")  # strikes look like $0.1-0.105: ratio to spot is far from 1
    sec = pd.DataFrame({"secid": [1, 1], "date": ["2024-03-15", "2024-03-14"], "close": [-101.0, 100.0]})
    with pytest.raises(ValueError, match="not positive"):
        underlying_close(sec, 1, pd.Timestamp("2024-03-15"))
    assert underlying_close(sec, 1, pd.Timestamp("2024-03-14")) == 100.0


def test_schema_validation_catches_bad_tables(clean_chain):
    ok = clean_chain[QUOTE_COLUMNS]
    validate_quotes(ok)
    with pytest.raises(ValueError, match="missing"):
        validate_quotes(ok.drop(columns="spot"))
    with pytest.raises(ValueError, match="cp_flag"):
        validate_quotes(ok.assign(cp_flag="X"))
    with pytest.raises(ValueError, match="non-negative"):
        validate_quotes(ok.assign(bid=-1.0))
    with pytest.raises(ValueError, match="datetime"):
        validate_quotes(ok.assign(expiry="2024-04-19"))


def test_fixture_round_trips_through_schema():
    chain = pd.read_csv(FIXTURES / "synthetic_chain.csv", parse_dates=["quote_time", "spot_time", "expiry"])
    validate_quotes(chain[QUOTE_COLUMNS])
    dist = pd.read_csv(FIXTURES / "synthetic_distributions.csv", parse_dates=["ex_date"])
    assert len(chain) == 30 and len(dist) == 2