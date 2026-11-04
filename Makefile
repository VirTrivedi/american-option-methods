PY ?= .venv/bin/python

.PHONY: validation test steps-1-4 step5 step5-synthetic step6 readme-tables audit all clean-derived

test:
	$(PY) -m pytest

steps-1-4:
	$(PY) scripts/step1_convergence.py
	$(PY) scripts/step2_pde.py
	$(PY) scripts/step3_lsm.py        # ~5 min
	$(PY) scripts/step4_dividends.py

step5-synthetic:
	$(PY) scripts/step5_market.py --synthetic

# market data steps need data/raw/ (see CONTRIBUTING.md); ~10 min on 10 cores
step5:
	$(PY) scripts/step5_market.py

step6:
	$(PY) scripts/step6_diagnostics.py

readme-tables:
	$(PY) scripts/build_readme_tables.py

audit:
	$(PY) scripts/audit_readme_numbers.py
	$(PY) scripts/check_links.py

all: test steps-1-4 step5 step6 readme-tables audit

validation:
	$(PY) scripts/validation.py

validation:
	$(PY) scripts/validation.py