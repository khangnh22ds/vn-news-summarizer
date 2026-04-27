# vn-news-summarizer — developer entrypoints.
# Run `make help` for a list.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# --- Tunables -------------------------------------------------------------
UV ?= uv
PY ?= $(UV) run python
COMPOSE ?= docker compose

# --------------------------------------------------------------------------
.PHONY: help
help: ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

# --- Setup ----------------------------------------------------------------
.PHONY: setup
setup: ## Install dev dependencies via uv (creates .venv).
	$(UV) sync --all-extras --dev
	@echo "Run 'cp .env.example .env' if you haven't yet."

.PHONY: precommit
precommit: ## Install pre-commit hooks.
	$(UV) run pre-commit install

# --- Quality --------------------------------------------------------------
.PHONY: lint
lint: ## Ruff lint + format check + mypy.
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy

.PHONY: format
format: ## Apply ruff format and import-sorting fixes.
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

.PHONY: test
test: ## Run pytest suite.
	$(UV) run pytest

.PHONY: cov
cov: ## Run pytest with coverage.
	$(UV) run pytest --cov --cov-report=term-missing

# --- Database -------------------------------------------------------------
.PHONY: db-up
db-up: ## Start Postgres + Redis via docker-compose.
	$(COMPOSE) up -d db redis

.PHONY: db-down
db-down: ## Stop and remove db/redis containers.
	$(COMPOSE) stop db redis

.PHONY: db-migrate
db-migrate: ## Run Alembic migrations (placeholder until Phase 1).
	@echo "Alembic not initialized yet. Will be wired up in Phase 1 (TICKET-002)."

# --- Pipeline entrypoints (placeholders until later phases) ---------------
.PHONY: crawl
crawl: ## Run a one-shot crawl (Phase 1).
	$(PY) scripts/run_crawler.py

.PHONY: label
label: ## Run LLM labeling worker (Phase 2).
	$(PY) scripts/run_labeler.py

.PHONY: train
train: ## Run training (Phase 4).
	$(PY) scripts/run_training.py

.PHONY: eval
eval: ## Run evaluation harness (Phase 3+).
	$(PY) scripts/run_eval.py

# --- Services -------------------------------------------------------------
.PHONY: api
api: ## Run the FastAPI dev server.
	$(UV) run uvicorn vn_news_api.app:app --reload --host 0.0.0.0 --port 8000

.PHONY: web
web: ## Run the Next.js dev server (after `cd apps/web && npm install`).
	cd apps/web && npm run dev

.PHONY: up
up: ## docker-compose up everything in the background.
	$(COMPOSE) up -d --build

.PHONY: down
down: ## docker-compose stop everything.
	$(COMPOSE) down

.PHONY: logs
logs: ## Follow docker-compose logs.
	$(COMPOSE) logs -f --tail=100

# --- Cleanup --------------------------------------------------------------
.PHONY: clean
clean: ## Remove caches and build artifacts (keeps .venv).
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +

.PHONY: distclean
distclean: clean ## Also remove the venv. Re-run `make setup` afterwards.
	rm -rf .venv
