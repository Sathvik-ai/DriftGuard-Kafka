# DriftGuard-Kafka M1 developer commands. Run `make help` for a list.

SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
COUNT ?= 1000
INTERVAL_MS ?= 2

# Load optional .env and export its variables to every command.
ifneq (,$(wildcard .env))
include .env
export
endif

.DEFAULT_GOAL := help
.PHONY: help setup kafka-up kafka-down kafka-clean kafka-logs kafka-verify topic-create \
        producer consumer test coverage lint format typecheck check demo clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv (Python >= 3.10) and install dependencies
	@if command -v uv >/dev/null 2>&1; then \
	  uv venv --python 3.11 $(VENV) && uv pip install --python $(PY) -r requirements-dev.txt; \
	else \
	  python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
	    || { echo "Python >= 3.10 required (try: brew install python@3.11)"; exit 1; }; \
	  python3 -m venv $(VENV) && $(PY) -m pip install --upgrade pip && \
	  $(PY) -m pip install -r requirements-dev.txt; \
	fi
	@test -f .env || cp .env.example .env
	@$(PY) --version

kafka-up: ## Start the Kafka broker and wait until healthy
	docker compose up -d
	./scripts/verify_kafka.sh --wait 120

kafka-down: ## Stop the broker (data volume is kept)
	docker compose down

kafka-clean: ## Stop the broker AND delete all Kafka data
	docker compose down -v

kafka-logs: ## Follow broker logs
	docker compose logs -f kafka

kafka-verify: ## Check broker health and list topics
	./scripts/verify_kafka.sh

topic-create: ## Create research.telemetry.raw with 3 partitions
	./scripts/create_topic.sh

producer: ## Publish COUNT events (default 1000) every INTERVAL_MS ms
	$(PY) -m app.producer --count $(COUNT) --interval-ms $(INTERVAL_MS)

consumer: ## Consume events (Ctrl+C to stop; LIMIT=N to stop after N)
	$(PY) -m app.consumer $(if $(LIMIT),--limit $(LIMIT),)

test: ## Run unit tests with coverage
	$(PY) -m pytest --cov --cov-report=term-missing

lint: ## Ruff lint + format check + mypy type check
	$(PY) -m ruff check app tests
	$(PY) -m ruff format --check app tests
	$(PY) -m mypy

format: ## Auto-format and auto-fix lint issues
	$(PY) -m ruff format app tests
	$(PY) -m ruff check --fix app tests

check: lint test ## Run all static checks and tests

demo: ## Full integration demo with PASS/FAIL result
	./scripts/run_demo.sh $(COUNT)

clean: ## Remove caches and run artefacts (keeps .venv and Kafka data)
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov runs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
