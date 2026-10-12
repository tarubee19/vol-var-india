.PHONY: install test lint synth fetch run api dash eval-assistant docker-build clean

PY ?= python

install:
	$(PY) -m pip install -e ".[dashboard,llm,data,dev]"

test:
	$(PY) -m pytest -q

lint:
	ruff check .

# simulated data for development and CI (clearly labelled SIM* tickers)
synth:
	$(PY) scripts/make_synthetic.py --out data/synthetic

# real data: run on a machine with internet access
fetch:
	$(PY) scripts/fetch_data.py --out data/raw

# full walk-forward run; pass DATA=data/synthetic to use simulated data
DATA ?= data/raw
run:
	$(PY) scripts/run_pipeline.py --data-dir $(DATA) --results-dir results

api:
	uvicorn volvar.api:app --host 0.0.0.0 --port 8000

dash:
	streamlit run app/dashboard.py

eval-assistant:
	$(PY) scripts/eval_assistant.py

docker-build:
	docker build -t vol-var-india .

clean:
	rm -rf .pytest_cache .ruff_cache results_tmp
