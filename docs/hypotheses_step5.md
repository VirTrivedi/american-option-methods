# Step 5 hypotheses and snapshot rules

## 1. Snapshot

```
snapshot_date: 2025-05-15     # Thursday, 14 days after AMZN's Q1 2025 report (2025-05-01)
ticker_nondividend: AMZN      # Amazon pays no dividend
ticker_dividend: SPY          # American-style ETF options, quarterly dividend; exactly one ex-date falls in the window
min_days: 14                  # expiry window in calendar days after the snapshot: 2025-05-29 ...
max_days: 64                  # ... to 2025-07-18, before AMZN's Q2 2025 report (2025-07-31)
expiry_targets_days: 14, 28, 42, 56, 64   # per ticker, the listed expiry nearest each target (at most 5 expiries), same rule for both tickers
next_earnings_date: 2025-07-31            # selected AMZN expiries on/after this date abort the run
cash_distr_types: 1             # distr_type 1 = ordinary cash dividend
```

## 2. Pre-registered cleaning thresholds (`CleanConfig`)

Relative spread <= 10% (sensitivity table at 5% and 20%), open interest >= 10, 7 <= days to expiry <= 365, 0.8 <= K/S <= 1.2,
BS vega at the mid IV >= 0.5 (price per 1.00 of vol), reject bid <= 0, ask < bid, mid below intrinsic, mid above K (puts) or S (calls),
AM-settled expiries. American IV: PDE M=N=400, Rannacher on, penalty method, sigma in [0.01, 3].

## 3. Hypotheses

**H1. BS vs American IV for puts.** For the AMZN-like chain the gap (BS minus American) is 0.03 to 0.16 for OTM puts (K/S 0.85 to 0.95), 0.05 at 14 days
rising to 0.30 at 65 days at the money, 0.5 to 0.6 at K/S 1.05, 3.4 falling to 1.3 at K/S 1.10, and 5.8 falling to 2.8 at K/S 1.15. So I expect a fraction
of a vol point at the money, and a gap that exceeds a typical half-width only for ITM puts. It should be larger at short maturities there, because deep ITM puts have tiny vega.
I would be surprised by a median ATM gap (K/S 0.98 to 1.02) above 0.6, a median gap above 0.3 for puts with K/S at or below 0.95, or an ITM gap that never exceeds the
half-width. I can't predict the gap relative to the half-width until I see the spreads, so I report that as measured.

**H2. AMZN calls.** With q = 0 the American call equals the European call, so BS and American IV should agree. In the synthetic chain the largest difference is 0.06
vol points for K/S between 0.9 and 1.1, which is grid error. I would be surprised by a median difference above the real-chain discretization check or above 0.1 vol points
for contracts that pass the vega filter.

**H3. AMZN parity premium.** The model's put early-exercise premium is 0.009 to 0.052 per 100 of spot at the money, rising to 0.16 to 0.30 at K/S 1.15, and it grows with moneyness
and maturity. That is 0.01% to 0.3% of spot, probably smaller than the two propagated half-spreads near the money, so I expect the test to be inconclusive there and informative only for deep ITM puts.
I would be surprised by a negative market premium far outside the propagated band.

**H4. SPY call-put IV gap.** With correct rates and dividends the American-model gap should be close to zero. On the synthetic chain the mean absolute gap is 0.03 vol points before the ex-date and
0.005 after, against 5.8 and 3.5 for BS (those BS averages are dominated by the wings). Near ATM the BS gap is only about 0.01 to 0.3. On the call side it is large only for K/S at or below 0.95 in the
first expiry after the ex-date. I would be surprised if the American model did not reduce the call-put gap, because that would point to the dividend, rate or quote-timing inputs rather than the model.

**H5. Vendor IV.** I expect my American IV at the mid price to be within 1 vol point of OptionMetrics' `impl_volatility` at the median, with larger differences in the deep ITM wings and for
short-dated contracts. I would be surprised by a systematic signed offset, which would mean the vendor's rate or dividend inputs differ from mine. I don't tune my inputs to match the vendor.