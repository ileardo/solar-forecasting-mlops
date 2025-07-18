# Solar Forecasting MLOps - Streamlined Development Automation
# Author: MLOps Team
# Description: Essential automation for development workflow

.PHONY: help setup install install-dev test test-unit test-integration lint format format-check clean clean-all clean-conda docker-build docker-up docker-down docker-logs docker-clean mlflow-ui mlflow-ui-conda mlflow-clean pre-commit-install pre-commit-run conda-create conda-install conda-install-dev conda-remove conda-info db-init db-init-conda db-migrate db-migrate-conda db-reset dev-setup dev-check dev-check-conda prod-test prod-test-conda prod-build info conda-activate-help

# Default target
.DEFAULT_GOAL := help

# Python and environment settings
PYTHON := python3
CONDA := conda
CONDA_ENV := solar-forecasting-mlops
PROJECT_NAME := solar-forecasting-mlops

# Directories
SRC_DIR := src
TEST_DIR := tests

# Docker settings
DOCKER_COMPOSE := docker-compose.yml

help: ## Show this help message
	@echo "Solar Forecasting MLOps - Available Commands:"
	@echo "============================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment Setup
setup: ## Complete environment setup (conda env + install + pre-commit)
	@echo "Setting up complete development environment..."
	$(MAKE) conda-install-dev
	$(MAKE) pre-commit-install
	@echo "Environment setup complete!"

install: ## Install production requirements in current environment
	@echo "Installing production requirements..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Production requirements installed!"

install-dev: ## Install development requirements in current environment
	@echo "Installing development requirements..."
	pip install --upgrade pip
	pip install -r requirements-dev.txt
	@echo "Development requirements installed!"

# Conda Environment Management
conda-create: ## Create conda environment
	@echo "Creating conda environment: $(CONDA_ENV)..."
	$(CONDA) create -n $(CONDA_ENV) python=3.11 -y
	@echo "Conda environment created! Activate with: conda activate $(CONDA_ENV)"

conda-install: conda-create ## Create conda env and install production requirements
	@echo "Installing production requirements in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) pip install --upgrade pip
	$(CONDA) run -n $(CONDA_ENV) pip install -r requirements.txt
	@echo "Conda environment ready with production requirements!"

conda-install-dev: conda-create ## Create conda env and install development requirements
	@echo "Installing development requirements in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) pip install --upgrade pip
	$(CONDA) run -n $(CONDA_ENV) pip install -r requirements-dev.txt
	@echo "Conda environment ready with development requirements!"

conda-remove: ## Remove conda environment
	@echo "Removing conda environment: $(CONDA_ENV)..."
	$(CONDA) env remove -n $(CONDA_ENV) -y
	@echo "Conda environment removed!"

conda-info: ## Show conda environment information
	@echo "Conda environment information:"
	@echo "==============================="
	@$(CONDA) info --envs | grep $(CONDA_ENV) || echo "Environment $(CONDA_ENV) not found"
	@echo ""
	@echo "To activate environment: conda activate $(CONDA_ENV)"

# Testing
test: ## Run all tests (unit + integration)
	@echo "Running all tests..."
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "All tests completed!"

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	pytest $(TEST_DIR)/unit -v --cov=$(SRC_DIR) --cov-report=term-missing
	@echo "Unit tests completed!"

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	pytest $(TEST_DIR)/integration -v --tb=short
	@echo "Integration tests completed!"

# Code Quality
lint: ## Run all linting checks
	@echo "Running linting checks..."
	pylint $(SRC_DIR)
	black --check --diff $(SRC_DIR) $(TEST_DIR)
	isort --check-only --diff $(SRC_DIR) $(TEST_DIR)
	mypy $(SRC_DIR)
	@echo "All linting checks completed!"

format: ## Format code with black and isort
	@echo "Formatting code..."
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)
	@echo "Code formatting completed!"

format-check: ## Check if code needs formatting
	@echo "Checking code formatting..."
	black --check $(SRC_DIR) $(TEST_DIR)
	isort --check-only $(SRC_DIR) $(TEST_DIR)

# Pre-commit Hooks
pre-commit-install: ## Install pre-commit hooks
	@echo "Installing pre-commit hooks..."
	pre-commit install
	@echo "Pre-commit hooks installed!"

pre-commit-run: ## Run pre-commit on all files
	@echo "Running pre-commit checks..."
	pre-commit run --all-files

# MLflow Management
mlflow-ui: ## Start MLflow UI
	@echo "Starting MLflow UI..."
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --port 5000

mlflow-ui-conda: ## Start MLflow UI in conda environment
	@echo "Starting MLflow UI in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --port 5000

mlflow-clean: ## Clean MLflow tracking data
	@echo "Cleaning MLflow data..."
	rm -rf mlruns/
	rm -f mlflow.db
	rm -rf artifacts/

# Infrastructure Management
docker-build: ## Build Docker containers
	@echo "Building Docker containers..."
	docker-compose -f $(DOCKER_COMPOSE) build

docker-up: ## Start all services with Docker Compose
	@echo "Starting all services..."
	docker-compose -f $(DOCKER_COMPOSE) up -d
	@echo "Services started! Check with: docker-compose ps"

docker-down: ## Stop all services
	@echo "Stopping all services..."
	docker-compose -f $(DOCKER_COMPOSE) down

docker-logs: ## Show logs from all services
	@echo "Showing service logs..."
	docker-compose -f $(DOCKER_COMPOSE) logs -f

docker-clean: ## Clean Docker containers and volumes
	@echo "Cleaning Docker resources..."
	docker-compose -f $(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f

# Database Management
db-init: ## Initialize database schema
	@echo "Initializing database..."
	$(PYTHON) -c "from $(SRC_DIR).utils.database import init_database; init_database()"

db-init-conda: ## Initialize database in conda environment
	@echo "Initializing database in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) python -c "from $(SRC_DIR).utils.database import init_database; init_database()"

db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	$(PYTHON) -c "from $(SRC_DIR).utils.database import run_migrations; run_migrations()"

db-migrate-conda: ## Run database migrations in conda environment
	@echo "Running database migrations in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) python -c "from $(SRC_DIR).utils.database import run_migrations; run_migrations()"

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "WARNING: Resetting database..."
	@read -p "Are you sure? This will destroy all data [y/N]: " confirm && [ "$$confirm" = "y" ]
	$(PYTHON) -c "from $(SRC_DIR).utils.database import reset_database; reset_database()"

# Cleaning
clean: ## Clean cache and temporary files
	@echo "Cleaning cache and temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	@echo "Cleanup completed!"

clean-all: clean mlflow-clean docker-clean ## Complete cleanup (cache + mlflow + docker)
	@echo "Complete cleanup performed!"

clean-conda: clean ## Clean cache and remove conda environment
	@echo "Cleaning cache and conda environment..."
	$(MAKE) conda-remove
	@echo "Complete conda cleanup performed!"

# Development Workflow
dev-setup: ## Complete development setup with conda
	@echo "Setting up development environment with conda..."
	$(MAKE) conda-install-dev
	$(MAKE) pre-commit-install
	@echo "Development environment ready!"

dev-check: ## Run all development checks
	@echo "Running all development checks..."
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) format-check
	@echo "All checks passed!"

dev-check-conda: ## Run all development checks in conda environment
	@echo "Running all development checks in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) pylint $(SRC_DIR)
	$(CONDA) run -n $(CONDA_ENV) pytest $(TEST_DIR) -v --tb=short --disable-warnings
	$(CONDA) run -n $(CONDA_ENV) black --check $(SRC_DIR) $(TEST_DIR)
	$(CONDA) run -n $(CONDA_ENV) isort --check-only $(SRC_DIR) $(TEST_DIR)
	@echo "All conda checks passed!"

# Production Helpers
prod-test: ## Run production-like tests
	@echo "Running production tests..."
	pytest $(TEST_DIR) -v --tb=short --disable-warnings
	$(MAKE) lint

prod-test-conda: ## Run production tests in conda environment
	@echo "Running production tests in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) pytest $(TEST_DIR) -v --tb=short --disable-warnings
	$(CONDA) run -n $(CONDA_ENV) pylint $(SRC_DIR)

prod-build: ## Build production artifacts
	@echo "Building production artifacts..."
	$(MAKE) clean
	$(MAKE) test
	$(MAKE) docker-build

# Project Information
info: ## Show project information
	@echo "Solar Forecasting MLOps Project"
	@echo "==============================="
	@echo "Python Version: $$($(PYTHON) --version)"
	@echo "Conda Version: $$($(CONDA) --version)"
	@echo "Current Environment: $$(conda info --envs | grep '*' | awk '{print $$1}' || echo 'Not in conda env')"
	@echo "Project Environment: $(CONDA_ENV)"
	@echo "Project Structure:"
	@tree -L 2 -I '__pycache__|*.pyc|.git' || ls -la

conda-activate-help: ## Show how to activate conda environment
	@echo "To activate the conda environment, run:"
	@echo "conda activate $(CONDA_ENV)"
	@echo ""
	@echo "To deactivate the environment, run:"
	@echo "conda deactivate"
