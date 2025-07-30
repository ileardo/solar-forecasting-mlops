# Solar Forecasting MLOps - Streamlined Development Automation
# Author: ileardo
# Description: Essential automation for development workflow

.PHONY: help setup check-prerequisites setup-env conda-create conda-install conda-install-dev conda-remove conda-info pre-commit-install pre-commit-run check-services lint format format-check test test-unit test-integration docker-build docker-up docker-ps docker-logs docker-down docker-clean aws-list-s3 aws-delete-all-s3 train predict-date predict-tomorrow info

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

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

# Environment files
ENV_FILE := .env
ENV_EXAMPLE := .env.example

help: ## Show this help message
	@echo "Solar Forecasting MLOps - Available Commands:"
	@echo "============================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment Setup
setup: ## Complete environment setup (prerequisites + env + conda + pre-commit)
	@echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Starting complete environment setup..."
	$(MAKE) check-prerequisites
	$(MAKE) setup-env
	$(MAKE) conda-install-dev
	$(MAKE) pre-commit-install
	@echo ""
	@echo "========================================"
	@echo "  Environment Setup Complete!"
	@echo "========================================"
	@echo "Activate the environment:"
	@echo "   conda activate $(CONDA_ENV)"

# Prerequisites check
check-prerequisites: ## Check if all required tools are installed
	@echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Checking prerequisites..."
	@command -v python >/dev/null 2>&1 || { echo -e "$(RED)[$(shell date +'%Y-%m-%d %H:%M:%S')] ERROR:$(NC) Python is not installed"; exit 1; }
	@command -v conda >/dev/null 2>&1 || { echo -e "$(RED)[$(shell date +'%Y-%m-%d %H:%M:%S')] ERROR:$(NC) Conda is not installed"; exit 1; }
	@command -v git >/dev/null 2>&1 || { echo -e "$(RED)[$(shell date +'%Y-%m-%d %H:%M:%S')] ERROR:$(NC) Git is not installed"; exit 1; }
	@command -v docker >/dev/null 2>&1 || echo -e "$(YELLOW)[$(shell date +'%Y-%m-%d %H:%M:%S')] WARNING:$(NC) Docker is not installed - some features will be unavailable"
	@echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Prerequisites check completed"

# Create .env file from template with secure random values
setup-env:
	@echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Setting up environment file..."
	@if [ ! -f "$(ENV_EXAMPLE)" ]; then \
		echo -e "$(RED)[$(shell date +'%Y-%m-%d %H:%M:%S')] ERROR:$(NC) $(ENV_EXAMPLE) file not found"; \
		exit 1; \
	fi
	@if [ -f "$(ENV_FILE)" ]; then \
		echo -e "$(YELLOW)[$(shell date +'%Y-%m-%d %H:%M:%S')] WARNING:$(NC) $(ENV_FILE) file already exists"; \
		read -p "Do you want to overwrite it? (y/N): " REPLY; \
		if [ "$$REPLY" != "y" ] && [ "$$REPLY" != "Y" ]; then \
			echo -e "$(BLUE)[$(shell date +'%Y-%m-%d %H:%M:%S')] INFO:$(NC) Skipping .env file creation"; \
			exit 0; \
		fi; \
	fi
	@cp "$(ENV_EXAMPLE)" "$(ENV_FILE)"
	@DB_PASSWORD=$$(openssl rand -base64 32 | tr -d "=+/\n" | head -c 20); \
	sed -i.bak "s/your_secure_password/$$DB_PASSWORD/g" "$(ENV_FILE)" && rm "$(ENV_FILE).bak"; \
	echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Environment file created with secure random values"; \
	echo -e "$(BLUE)[$(shell date +'%Y-%m-%d %H:%M:%S')] INFO:$(NC) Database password: $$DB_PASSWORD"

# Conda Environment Management
# Create conda environment
conda-create:
	@echo "Creating conda environment: $(CONDA_ENV)..."
	$(CONDA) create -n $(CONDA_ENV) python=3.11 -y
	@echo "Conda environment created! Activate with: conda activate $(CONDA_ENV)"

# Create conda env and install production requirements
conda-install: conda-create
	@echo "Installing production requirements in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) pip install --upgrade pip
	$(CONDA) run -n $(CONDA_ENV) pip install -r requirements.txt
	@echo "Conda environment ready with production requirements!"

# Create conda env and install development requirements
conda-install-dev: conda-create
	@echo "Installing development requirements in conda environment..."
	$(CONDA) run -n $(CONDA_ENV) pip install --upgrade pip
	$(CONDA) run -n $(CONDA_ENV) pip install -r requirements-dev.txt
	@echo "Conda environment ready with development requirements!"

# Remove conda environment
conda-remove:
	@echo "Removing conda environment: $(CONDA_ENV)..."
	$(CONDA) env remove -n $(CONDA_ENV) -y
	@echo "Conda environment removed!"

# Pre-commit Hooks
# Install pre-commit hooks
pre-commit-install:
	@echo "Installing pre-commit hooks..."
	$(CONDA) run -n $(CONDA_ENV) pre-commit install
	@echo "Pre-commit hooks installed!"

# Run pre-commit on all files
pre-commit-run:
	@echo "Running pre-commit checks..."
	$(CONDA) run -n $(CONDA_ENV) pre-commit run --all-files

# Services check
check-services: ## Check if all MLOps services are running
	@echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Checking MLOps services..."
	@echo "PostgreSQL:" && docker exec solar-postgres pg_isready -U $(DB_USER) -d $(DB_NAME) || echo "PostgreSQL not ready"
	@echo "LocalStack S3:" && docker exec solar-localstack awslocal s3 ls || echo "LocalStack not ready"
	@echo "MLflow:" && curl -f http://localhost:5000/health >/dev/null 2>&1 && echo "MLflow ready" || echo "MLflow not ready"
	@echo "Prefect:" && curl -f http://localhost:4200/api/health >/dev/null 2>&1 && echo "Prefect ready" || echo "Prefect not ready"

# Code Quality
lint: ## Run all linting checks
	@echo "Running pylint..."
	pylint $(SRC_DIR)

format: ## Format code with black and isort
	@echo "Formatting code..."
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)
	@echo "Code formatting completed!"

format-check: ## Check if code needs formatting
	@echo "Checking code formatting..."
	black --check $(SRC_DIR) $(TEST_DIR)
	isort --check-only $(SRC_DIR) $(TEST_DIR)

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

# Infrastructure Management
docker-build: ## Build Docker containers
	@echo "Building Docker containers..."
	docker-compose -f $(DOCKER_COMPOSE) build

docker-up: ## Start all services with Docker Compose
	@echo "Starting all services..."
	docker-compose -f $(DOCKER_COMPOSE) up -d
	@echo "Services started! Check with: docker-compose ps"

docker-ps: ## Show status of Docker containers
	@echo "Showing status of Docker containers..."
	docker-compose -f $(DOCKER_COMPOSE) ps

docker-logs: ## Show logs from all services
	@echo "Showing service logs..."
	docker-compose -f $(DOCKER_COMPOSE) logs -f

docker-down: ## Stop all services
	@echo "Stopping all services..."
	docker-compose -f $(DOCKER_COMPOSE) down

docker-clean: ## Clean Docker containers and volumes
	@echo "Cleaning Docker resources..."
	docker-compose -f $(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f

# LocalStack S3 Management
aws-list-s3: ## List all LocalStack S3 buckets
	@echo "Listing LocalStack S3 buckets..."
	@awslocal s3 ls

aws-delete-all-s3: ## Delete all LocalStack S3 buckets
	@echo "Deleting all LocalStack S3 buckets..."
	@for bucket in $$(awslocal s3 ls | awk '{print $$3}'); do \
		echo "Deleting bucket $$bucket..."; \
		awslocal s3 rb s3://$$bucket --force; \
	done
	@echo "All buckets deleted."

# Training and Prediction
train: ## Run complete training pipeline (train + eval + registry)
	@echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Starting training pipeline..."
	python -m src.model.run_pipeline
	@echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Training pipeline completed!"

predict-date: ## Run batch prediction for specific date (usage: make predict-date DATE=2020-06-15)
	@echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Running batch prediction for $(DATE)..."
	$(CONDA) run -n $(CONDA_ENV) python -m src.batch.orchestrator $(DATE)
	@echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Batch prediction completed!"

predict-tomorrow: ## Run batch prediction for tomorrow
	@echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Running batch prediction for tomorrow..."
	$(CONDA) run -n $(CONDA_ENV) python -m src.batch.orchestrator
	@echo -e "$(GREEN)[$(shell date +'%Y-%m-%d %H:%M:%S')]$(NC) Batch prediction completed!"

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
