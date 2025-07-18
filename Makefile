# Solar Forecasting MLOps - Development Automation
# Author: MLOps Team
# Description: Comprehensive automation for development workflow

.PHONY: help setup install install-dev test test-unit test-integration lint format clean clean-all docker-build docker-up docker-down

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
DATA_DIR := data
DOCKER_DIR := infrastructure/docker

# Docker settings
DOCKER_COMPOSE := docker-compose.yml
SERVICES := postgres mlflow localstack grafana

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

test-watch: ## Run tests in watch mode
	@echo "Running tests in watch mode..."
	pytest-watch $(TEST_DIR) --runner "pytest -v"

test-conda: ## Run all tests in conda environment
	@echo "Running tests in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

# Code Quality
lint: ## Run all linting checks
	@echo "Running linting checks..."
	$(MAKE) lint-pylint
	$(MAKE) lint-black-check
	$(MAKE) lint-isort-check
	$(MAKE) lint-mypy
	@echo "All linting checks completed!"

lint-pylint: ## Run pylint
	@echo "Running pylint..."
	pylint $(SRC_DIR)

lint-black-check: ## Check code formatting with black
	@echo "Checking code formatting..."
	black --check --diff $(SRC_DIR) $(TEST_DIR)

lint-isort-check: ## Check import sorting with isort
	@echo "Checking import sorting..."
	isort --check-only --diff $(SRC_DIR) $(TEST_DIR)

lint-mypy: ## Run type checking with mypy
	@echo "Running type checks..."
	mypy $(SRC_DIR)

lint-conda: ## Run all linting checks in conda environment
	@echo "Running linting checks in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) pylint $(SRC_DIR)
	$(CONDA) run -n $(CONDA_ENV) black --check --diff $(SRC_DIR) $(TEST_DIR)
	$(CONDA) run -n $(CONDA_ENV) isort --check-only --diff $(SRC_DIR) $(TEST_DIR)
	$(CONDA) run -n $(CONDA_ENV) mypy $(SRC_DIR)

# Code Formatting
format: ## Format code with black and isort
	@echo "Formatting code..."
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)
	@echo "Code formatting completed!"

format-check: ## Check if code needs formatting
	@echo "Checking code formatting..."
	black --check $(SRC_DIR) $(TEST_DIR)
	isort --check-only $(SRC_DIR) $(TEST_DIR)

format-conda: ## Format code in conda environment
	@echo "Formatting code in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) black $(SRC_DIR) $(TEST_DIR)
	$(CONDA) run -n $(CONDA_ENV) isort $(SRC_DIR) $(TEST_DIR)

# Pre-commit Hooks
pre-commit-install: ## Install pre-commit hooks
	@echo "Installing pre-commit hooks..."
	pre-commit install
	@echo "Pre-commit hooks installed!"

pre-commit-run: ## Run pre-commit on all files
	@echo "Running pre-commit checks..."
	pre-commit run --all-files

pre-commit-conda: ## Install and run pre-commit in conda environment
	@echo "Setting up pre-commit in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) pre-commit install
	$(CONDA) run -n $(CONDA_ENV) pre-commit run --all-files

# Data Management
data-download: ## Download dataset from source
	@echo "Downloading dataset..."
	@mkdir -p $(DATA_DIR)/raw
	@echo "Manual download required from Kaggle"
	@echo "URL: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data"

data-validate: ## Validate raw dataset
	@echo "Validating dataset..."
	$(PYTHON) -c "from $(SRC_DIR).data.loader import SolarDataLoader; SolarDataLoader.validate_raw_data('$(DATA_DIR)/raw')"

data-validate-conda: ## Validate dataset in conda environment
	@echo "Validating dataset in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) python -c "from $(SRC_DIR).data.loader import SolarDataLoader; SolarDataLoader.validate_raw_data('$(DATA_DIR)/raw')"

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
	$(MAKE) pre-commit-conda
	@echo "Development environment ready!"

dev-check: ## Run all development checks
	@echo "Running all development checks..."
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) format-check
	@echo "All checks passed!"

dev-check-conda: ## Run all development checks in conda environment
	@echo "Running all development checks in conda environment..."
	$(MAKE) lint-conda
	$(MAKE) test-conda
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
	$(MAKE) lint-conda

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
