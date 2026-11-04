# Running everything

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e . --no-deps
make test            # pytest, synthetic data only, no network, ~1 min
make steps-1-4       # figures and tables for the model-only steps (~15 min, mostly Step 3)
make step5-synthetic # pipeline demo on the synthetic fixture
```

Market-data steps (5 and 6) need OptionMetrics data from WRDS under your own licence:

```bash
.venv/bin/pip install -r requirements-wrds.txt
.venv/bin/python scripts/pull_wrds.py --help   # reads your ~/.pgpass; credentials are never printed or stored; files go to data/raw/ (gitignored)
make step5 step6 audit
```

Rules the repository follows (please keep them):

- `data/` is gitignored. Vendor data and per-contract tables derived from it (bid/ask/vendor IV by strike) live in `data/` only; `results/tables/` holds aggregates.
- Tests use synthetic data only (`tests/fixtures/`). Tolerances come from measurements, with the measured value and the reason in the test comment.
- `docs/hypotheses_step5.md` holds the predictions I stated in advance. Do not edit predictions or thresholds; add a dated amendment instead.
- Anything done after seeing results is labelled "exploratory (post-hoc)" in code, tables and figures, and flagged once at the top of any README or docs section that reports it.
- Results in README tables are generated (`make readme-tables`) and checked (`make audit`); do not edit them by hand.

## Pulling WRDS data

Install `requirements-wrds.txt`, put your WRDS credentials in `~/.pgpass` (`wrds-pgdata.wharton.upenn.edu:9737:wrds:<username>:<password>`, mode 600) and run `python scripts/pull_wrds.py --date 2025-05-15 --tickers AMZN SPY --username <you>`. The script never reads, prints or stores the password itself. Files land in `data/raw/` (gitignored) with the pull timestamp in the name. The snapshot, tickers, expiry window and cash `distr_type` codes come from `docs/hypotheses_step5.md`.