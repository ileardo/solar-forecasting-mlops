#!/bin/bash

# Solar Forecasting MLOps - Environment Setup Script
# This script automates the complete environment setup process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="solar-forecasting-mlops"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
ENV_EXAMPLE="${PROJECT_ROOT}/.env.example"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# Check if running from project root
check_project_root() {
    if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
        log_error "Must be run from project root directory"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check Python
    if ! command -v python &> /dev/null; then
        log_error "Python is not installed"
        exit 1
    fi

    # Check Conda
    if ! command -v conda &> /dev/null; then
        log_error "Conda is not installed"
        exit 1
    fi

    # Check Docker (optional)
    if ! command -v docker &> /dev/null; then
        log_warning "Docker is not installed - some features will be unavailable"
    fi

    # Check Git
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed"
        exit 1
    fi

    log "Prerequisites check completed"
}

# Create .env file from template
setup_env_file() {
    log "Setting up environment file..."

    if [ ! -f "${ENV_EXAMPLE}" ]; then
        log_error ".env.example file not found"
        exit 1
    fi

    if [ -f "${ENV_FILE}" ]; then
        log_warning ".env file already exists"
        read -p "Do you want to overwrite it? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping .env file creation"
            return 0
        fi
    fi

    # Copy template to .env
    cp "${ENV_EXAMPLE}" "${ENV_FILE}"

    # Generate secure random values
    DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    API_SECRET_KEY=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-64)
    JUPYTER_TOKEN=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)

    # Replace placeholder values
    sed -i "s/your_secure_password/${DB_PASSWORD}/g" "${ENV_FILE}"
    sed -i "s/your_super_secret_key_change_in_production/${API_SECRET_KEY}/g" "${ENV_FILE}"
    sed -i "s/your_jupyter_token/${JUPYTER_TOKEN}/g" "${ENV_FILE}"

    log "Environment file created with secure random values"
    log_info "Database password: ${DB_PASSWORD}"
    log_info "API secret key: ${API_SECRET_KEY:0:20}..."
    log_info "Jupyter token: ${JUPYTER_TOKEN:0:20}..."
}

# Create required directories
create_directories() {
    log "Creating required directories..."

    local dirs=(
        "data/raw"
        "data/processed"
        "artifacts"
        "mlruns"
        "prefect_home"
        "evidently_workspace"
        "monitoring_logs"
        "htmlcov"
        "notebooks"
        "sql/migrations"
        "grafana/dashboards"
        "grafana/provisioning/datasources"
        "grafana/provisioning/dashboards"
    )

    for dir in "${dirs[@]}"; do
        mkdir -p "${PROJECT_ROOT}/${dir}"
        log_info "Created directory: ${dir}"
    done

    # Create .gitkeep files for empty directories
    touch "${PROJECT_ROOT}/data/raw/.gitkeep"
    touch "${PROJECT_ROOT}/monitoring_logs/.gitkeep"

    log "Required directories created"
}

# Setup conda environment
setup_conda_environment() {
    log "Setting up Conda environment..."

    # Check if environment exists
    if conda env list | grep -q "^${PROJECT_NAME}"; then
        log_warning "Conda environment '${PROJECT_NAME}' already exists"
        read -p "Do you want to recreate it? (y/N): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n "${PROJECT_NAME}" -y
            log_info "Removed existing environment"
        else
            log_info "Using existing environment"
            return 0
        fi
    fi

    # Create new environment
    conda create -n "${PROJECT_NAME}" python=3.11 -y
    log_info "Created Conda environment: ${PROJECT_NAME}"

    # Activate environment and install dependencies
    log "Installing dependencies..."
    conda run -n "${PROJECT_NAME}" pip install --upgrade pip
    conda run -n "${PROJECT_NAME}" pip install -r "${PROJECT_ROOT}/requirements-dev.txt"

    log "Conda environment setup completed"
}

# Setup pre-commit hooks
setup_pre_commit() {
    log "Setting up pre-commit hooks..."

    conda run -n "${PROJECT_NAME}" pre-commit install
    log_info "Pre-commit hooks installed"

    # Run pre-commit on all files
    log "Running pre-commit checks..."
    conda run -n "${PROJECT_NAME}" pre-commit run --all-files || true

    log "Pre-commit setup completed"
}

# Validate environment
validate_environment() {
    log "Validating environment setup..."

    # Test Python imports
    if ! conda run -n "${PROJECT_NAME}" python -c "from src.config.settings import get_settings; print('Settings loaded successfully')" > /dev/null 2>&1; then
        log_error "Failed to load settings"
        return 1
    fi

    # Test validation
    local validation_result
    validation_result=$(conda run -n "${PROJECT_NAME}" python -c "from src.config.settings import validate_required_env_vars; import json; print(json.dumps(validate_required_env_vars()))")

    log_info "Environment validation result:"
    echo "${validation_result}" | conda run -n "${PROJECT_NAME}" python -m json.tool

    # Check if overall validation passed
    if echo "${validation_result}" | grep -q '"overall": true'; then
        log "Environment validation passed"
        return 0
    else
        log_warning "Environment validation has some issues (expected for development setup)"
        return 0
    fi
}

# Generate activation script
generate_activation_script() {
    log "Generating activation script..."

    cat > "${PROJECT_ROOT}/activate_env.sh" << 'EOF'
#!/bin/bash
# Solar Forecasting MLOps - Environment Activation Script

echo "Activating Solar Forecasting MLOps environment..."

# Activate conda environment
conda activate solar-forecasting-mlops

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Show environment info
echo "Environment activated successfully!"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "Python path includes: ${PYTHONPATH}"

# Show available make commands
echo ""
echo "Available make commands:"
make help
EOF

    chmod +x "${PROJECT_ROOT}/activate_env.sh"
    log_info "Created activation script: activate_env.sh"
}

# Show completion message
show_completion() {
    echo ""
    echo "========================================"
    echo "  Environment Setup Complete!"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "1. Activate the environment:"
    echo "   conda activate ${PROJECT_NAME}"
    echo "   # OR"
    echo "   source activate_env.sh"
    echo ""
    echo "2. Verify installation:"
    echo "   make dev-check"
    echo ""
    echo "3. Start development:"
    echo "   make help  # See all available commands"
    echo ""
    echo "4. Download dataset:"
    echo "   make data-download  # Follow manual instructions"
    echo ""
    echo "Environment file created: .env"
    echo "Check the file for your database password and API keys"
    echo ""
}

# Main execution
main() {
    log "Starting Solar Forecasting MLOps environment setup..."

    check_project_root
    check_prerequisites
    setup_env_file
    create_directories
    setup_conda_environment
    setup_pre_commit
    validate_environment
    generate_activation_script
    show_completion

    log "Environment setup completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-conda)
            SKIP_CONDA=true
            shift
            ;;
        --skip-precommit)
            SKIP_PRECOMMIT=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-conda      Skip conda environment setup"
            echo "  --skip-precommit  Skip pre-commit hooks setup"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"
