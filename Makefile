.DEFAULT_GOAL := help
SHELL := /bin/bash

VENV := .venv
PY := $(VENV)/bin/python
UV := uv

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Create the venv and install dependencies
	$(UV) sync --all-extras

.PHONY: up
up: ## Start Postgres (pgvector)
	docker compose up -d
	@echo "waiting for postgres..."
	@until docker compose exec -T db pg_isready -q; do sleep 1; done
	@echo "ready"

.PHONY: down
down: ## Stop Postgres (keeps the volume)
	docker compose down

.PHONY: migrate
migrate: ## Apply migrations
	$(PY) -m alembic upgrade head

.PHONY: revision
revision: ## Autogenerate a migration: make revision m="message"
	$(PY) -m alembic revision --autogenerate -m "$(m)"

.PHONY: api
api: ## Run the API with reload
	$(PY) -m uvicorn hontology.api.main:app --reload --port 8100

.PHONY: ui
ui: ## Run the Streamlit UI (needs the API running)
	$(VENV)/bin/streamlit run src/hontology/ui/Home.py

.PHONY: test
test: ## Run the test suite
	$(PY) -m pytest -q

.PHONY: test-fast
test-fast: ## Run only tests that need no services
	$(PY) -m pytest -q -m "not requires_db and not requires_llm"

.PHONY: lint
lint: ## Lint and type-check
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests

.PHONY: fmt
fmt: ## Autoformat
	$(VENV)/bin/ruff check --fix src tests
	$(VENV)/bin/ruff format src tests

.PHONY: doctor
doctor: ## Check that the database and the LLM provider are reachable
	$(PY) -m hontology.cli doctor

.PHONY: ingest
ingest: ## Catch up to the newest published feed slice, then exit
	$(PY) -m hontology.cli ingest once

.PHONY: ingest-status
ingest-status: ## Show the ingest watermark and how far behind it is
	$(PY) -m hontology.cli ingest status

.PHONY: watch
watch: ## OPTIONAL: follow the feed continuously (Ctrl-C to stop)
	$(PY) -m hontology.cli ingest watch

.PHONY: watch-docker
watch-docker: ## OPTIONAL: run the watcher as a container
	docker compose --profile watch up -d --build watcher
