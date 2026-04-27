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
db-migrate: ## Apply Alembic migrations to the configured DATABASE_URL.
	$(UV) run alembic upgrade head

.PHONY: db-revision
db-revision: ## Autogenerate a new Alembic revision (use MSG="...").
	$(UV) run alembic revision --autogenerate -m "$(MSG)"

.PHONY: db-reset
db-reset: ## DEV ONLY: drop SQLite file and re-apply all migrations.
	rm -f data/app.db
	$(UV) run alembic upgrade head

# --- Pipeline entrypoints --------------------------------------------------
.PHONY: crawl
crawl: ## Run a one-shot crawl over all enabled sources.
	$(UV) run python scripts/run_crawler.py

.PHONY: crawl-schedule
crawl-schedule: ## Run the crawler as a long-lived scheduler (every 30 min).
	$(UV) run python scripts/run_crawler.py --schedule

.PHONY: label
label: ## Run LLM labeling worker (Phase 2). Use LIMIT=N to bound the batch.
	$(PY) scripts/run_labeler.py --limit $(or $(LIMIT),50)

.PHONY: label-export
label-export: ## Export QC-passed labels to data/datasets/<NAME>. Use NAME=v1.
	$(PY) scripts/run_labeler.py --no-label --export $(or $(NAME),v1)

.PHONY: train
train: ## Run training (Phase 4).
	$(PY) scripts/run_training.py

.PHONY: eval
eval: ## Run baseline eval. Use BASELINE=lexrank|textrank DATASET=v1 SPLIT=test.
	$(PY) scripts/run_eval.py \
	  --baseline $(or $(BASELINE),lexrank) \
	  --dataset  $(or $(DATASET),v1) \
	  --split    $(or $(SPLIT),test)

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
