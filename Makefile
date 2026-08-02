.PHONY: help setup all collect process analyze features train figures dashboard test clean reset

PYTHON ?= python3
PORT   ?= 8501

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies and create .env from the template
	$(PYTHON) -m pip install -r requirements.txt
	@test -f .env || (cp .env.example .env && echo "Created .env - add your AIESEC_ACCESS_TOKEN")

all:  ## Run the full pipeline (collect -> process -> analyze -> features -> train -> figures)
	$(PYTHON) run_pipeline.py --step all

offline:  ## Run the full pipeline forcing the offline reference dataset
	$(PYTHON) run_pipeline.py --step all --use-reference-data

collect:   ## Collect raw data from the AIESEC Analytics API
	$(PYTHON) run_pipeline.py --step collect

process:   ## Parse and validate raw responses into the processed dataset
	$(PYTHON) run_pipeline.py --step process

analyze:   ## Run exploratory analysis and write report tables
	$(PYTHON) run_pipeline.py --step analyze

features:  ## Build the supervised feature matrix
	$(PYTHON) run_pipeline.py --step features

train:     ## Backtest, select, persist the model and forecast 2026
	$(PYTHON) run_pipeline.py --step train

figures:   ## Export static PNG figures
	$(PYTHON) run_pipeline.py --step figures

dashboard:  ## Launch the Streamlit dashboard
	$(PYTHON) -m streamlit run app.py --server.port $(PORT)

test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

clean:  ## Remove Python caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

reset: clean  ## Remove every generated artefact (raw data, models, outputs)
	rm -f data/raw/*.json data/processed/*.csv data/interim/* models/*.pkl
	rm -f outputs/predictions_2026.csv outputs/figures/*.png outputs/reports/*.csv
	@echo "Artefacts cleared. Rebuild with: make all"
