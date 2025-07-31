# Solar Forecasting MLOps

End-to-end MLOps pipeline for solar power generation forecasting using production-ready infrastructure and monitoring systems.

## Problem Statement

This project implements a complete MLOps framework for solar power forecasting. The system predicts 24-hour solar power generation using historical data and operational constraints.

**Key Challenge**: Design a production-ready batch prediction system that operates under realistic constraints: predictions must be generated at midnight using only historical data available at that time, ensuring no future data leakage.

**ML Model**: XGBoost multi-output regressor that takes historical solar generation data (lag features from 1, 2, 3, 7, and 30 days) and temporal features (hour of day, seasonality patterns, solar elevation proxies) as input, and outputs 24-hour ahead AC power predictions (one prediction per hour for the next day). The model uses only features available at midnight to ensure operational realism.

**Business Value**: Enable grid operators and energy companies to plan daily operations by providing reliable next-day solar generation forecasts with automated monitoring and retraining capabilities.

**Note**: This is an MLOps framework demonstration. Model performance is adequate but not state-of-the-art, as the focus is on infrastructure,
monitoring, and operational reliability.

## Dataset

Using [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) from Kaggle focusing on Plant 1:
- **Size**: ~800 hourly records over 34 days
- **Features**: Historical AC/DC power generation with lag features
- **Target**: 24-hour ahead AC power forecasting
- **Preprocessing**: Rigorous time series validation to prevent data leakage

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Cloud** | LocalStack (local AWS simulation) |
| **Experiment Tracking** | MLflow |
| **Model Registry** | MLflow |
| **Workflow Orchestration** | Prefect |
| **Monitoring** | Evidently + PostgreSQL + Grafana |
| **Deployment** | Docker + Docker Compose |
| **Database** | PostgreSQL (metrics + MLflow backend) |
| **Storage** | S3 (mock with LocalStack for artifacts) |
| **Visualization** | Grafana dashboards |
| **Database Admin** | Adminer |
| **Testing** | pytest |
| **Code Quality** | pylint, black, isort |
| **Automation** | pre-commit hooks, Makefile |

## Project Structure

```
solar-forecasting-mlops/
├── src/                  # Source code
│   ├── batch/            # Batch prediction service
│   ├── monitoring/       # Drift detection and monitoring
│   ├── model/            # Model training and registry
│   └── data/             # Data preprocessing
├── tests/                # Test suites
├── infrastructure/       # Docker and deployment configs
├── grafana/              # Monitoring dashboards
├── sql/                  # Database schemas
└── notebooks/            # Development notebooks
```

## System Architecture

### Cloud Infrastructure
- **LocalStack Integration**: AWS S3 simulation for artifact storage
- **Containerized Deployment**: Complete Docker Compose orchestration
- **Note**: Uses LocalStack instead of real AWS for cost-effective development and peer review

### Experiment Tracking & Model Registry
- **MLflow Tracking**: Complete experiment logging with parameters, metrics, and artifacts
- **Model Registry**: Full model lifecycle management with staging (None → Staging → Production → Archived)
- **Artifact Storage**: Models stored in S3 (LocalStack) with version control and preprocessor stored locally with run ID signature

### Workflow Orchestration
- **Prefect Flows**: Implemented workflow orchestration for batch predictions
- **Task Dependencies**: Proper task sequencing with validation and monitoring
- **Manual Execution**: Flows can be run manually via CLI commands
- **Note**: Scheduling and full deployment capabilities exist but not fully configured for productio

### Model Deployment
- **Batch Service**: Batch prediction service implemented
- **Production Registry Integration**: Automatic loading of production models from MLflow
- **Local Deployment**: Service runs locally with Docker Compose
- **Note**: Infrastructure containerized but not deployed to actual cloud environment

### Model Monitoring
- **Drift Detection**: Basic statistical drift monitoring implemented
- **Performance Tracking**: Model performance metrics collection and storage
- **Dashboard Infrastructure**: Grafana dashboards configured for monitoring
- **Note**: Monitoring reports metrics but automated retraining workflows not fully implemented



## Best Practices Implementation

| Practice | Status | Implementation |
|----------|--------|----------------|
| **Unit Tests** | ✅ | pytest with  test coverage |
| **Integration Tests** | ✅ | End-to-end pipeline testing |
| **Code Quality** | ✅ | pylint, black, isort with strict standards |
| **Makefile** | ✅ | Comprehensive automation commands |
| **Pre-commit Hooks** | ✅ | Automated quality checks before commits |
| **CI/CD Pipeline** | ❌ | Not implemented (scope limitation) |

## Quick Start

### Prerequisites
- Python 3.11+
- Conda
- Docker & Docker Compose
- Git

### Setup Instructions
```bash
# 1. Clone repository
git clone https://github.com/ileardo/solar-forecasting-mlops.git
cd solar-forecasting-mlops
conda init

# 2. Setup environment
make setup
conda activate solar-forecasting-mlops

# 3. Start infrastructure services
make docker-build && make docker-up

# 4. Verify services are running
make check-services

# 5. Run complete workflow
make train             # Train model with MLflow tracking
make predict-tomorrow  # Generate batch prediction
make monitor           # Execute monitoring analysis
```

### Service Access
Once services are running, access the following interfaces:

| Service | URL | Purpose |
|---------|-----|---------|
| **MLflow** | http://localhost:5000 | Experiment tracking and model registry |
| **Grafana** | http://localhost:3001 | Monitoring dashboards (admin/admin) |
| **Prefect** | http://localhost:4200 | Workflow orchestration UI |
| **Adminer** | http://localhost:8080 | Database administration |

**Database Access**: The automatically generated database password is stored in the `.env` file after running `make setup`.

**Note**: In GitHub Codespaces, you'll need to forward these ports to access the UIs.

## Key Commands

The project provides comprehensive automation through Make commands:

```bash
make setup              # Complete environment setup
make train              # Run training pipeline with MLflow tracking --> train + staging + production
make predict-tomorrow   # Run batch prediction flow for tomoroww
make predict-date       # Run batch prediction flow (DATE=YYYY-MM-DD)
make monitor            # Execute monitoring flow with drift (last 7 days) analysis
make docker-up          # Start all infrastructure services
make check-services     # Verify all services are running
```

For complete command reference: `make help`

## Operational Workflow

### Daily Operations
1. **Automated Training**: `make train` - Experiments tracked in MLflow, best model promoted to production
2. **Scheduled Prediction**: `make predict-tomorrow` - Prefect flow generates next-day forecast at midnight
3. **Continuous Monitoring**: `make monitor` - Weekly drift analysis with automatic alerting

### Production Scheduling
- **Batch Predictions**: Prefect can schedule daily predictions at midnight
- **Drift Monitoring**: Weekly monitoring flows detect model degradation
- **Automatic Retraining**: Conditional workflows trigger retraining when drift exceeds thresholds
- **Manual Override**: All processes can be triggered manually for operational flexibility

## Monitoring & Alerting

The system implements comprehensive monitoring:

### Drift Detection
- **Statistical Drift**: Comparison of current vs. reference feature distributions
- **Configurable Thresholds**: Customizable sensitivity levels
- **Feature-Level Analysis**: Per-feature drift scoring and alerting

### Performance Monitoring
- **Model Metrics**: RMSE, MAE, R² tracking over time
- **Prediction Quality**: Energy generation accuracy monitoring
- **System Health**: Infrastructure and service status monitoring

### Grafana Dashboards
- **Drift Trends**: Visual drift patterns over time
- **Performance Metrics**: Model accuracy and prediction quality
- **System Status**: Service health and operational metrics
- **Note**: Monitoring Dashboard under developement

## Future Development

### Short-term Enhancements
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Advanced Monitoring**: Feature importance drift and concept drift detection
- **Model Optimization**: Hyperparameter tuning and ensemble methods

### Production Readiness
- **Real AWS Integration**: Replace LocalStack with actual AWS services
- **Kubernetes Deployment**: Container orchestration for scalability
- **Infrastructure as Code**: Terraform for automated cloud provisioning
- **Security Hardening**: Secrets management and access controls

### Scalability Improvements
- **Multi-Plant Support**: Extend to multiple solar installations
- **Real-time Streaming**: Apache Kafka for streaming predictions
- **Advanced Orchestration**: Complex workflow dependencies and error handling
- **API Layer**: REST API for external integrations

This implementation demonstrates a production-ready MLOps framework suitable for real-world solar forecasting operations with comprehensive monitoring, automated workflows, and operational reliability.
