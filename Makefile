PY ?= python

.PHONY: setup data quality twin backtest replay train charts test lint validate-infra app

setup:            ## virtualenv with dev dependencies
	$(PY) -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

data:             ## download NOAA weather (pinned, SHA-256 checked)
	$(PY) -m bijuli download

quality: data     ## weather data-quality report
	$(PY) -m bijuli quality

twin: data        ## build the grid twin, check it against 40 real NEA daily reports
	$(PY) -m bijuli twin

backtest: data    ## walk-forward backtest, Jan 2023 - Aug 2025 (~3 min)
	$(PY) -m bijuli backtest

replay: data      ## replay 26 months of daily operations under 6 retraining policies (~15 min)
	$(PY) -m bijuli replay

train: data       ## train on the trailing year, register as champion in artifacts/registry
	$(PY) -m bijuli train --until 2025-08-01

charts:           ## figures in docs/img from the reports
	$(PY) scripts/make_charts.py

test:
	$(PY) -m pytest -q

lint:
	ruff check .

validate-infra:
	cfn-lint infra/template.yaml

app:              ## the Streamlit demo on http://localhost:8501 (pip install -r requirements-app.txt first)
	$(PY) -m streamlit run app.py
