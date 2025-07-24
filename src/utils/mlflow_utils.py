"""
MLflow utilities for solar forecasting project.

Essential helper functions for MLflow tracking, experiments, and model registry.
Provides simple interface for common MLflow operations.
- setup MLflow tracking
- get or create experiments
- start runs
- log metrics and parameters
- register models
- validate MLflow connection
- cleanup test experiments
"""

from src.config.settings import get_settings

import logging
from typing import (
    Any,
    Dict,
    Optional,
    Tuple
)

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


# Configure logging
logger = logging.getLogger(__name__)


def setup_mlflow_tracking() -> None:
    """
    Setup MLflow tracking with configuration from settings.

    Configures tracking URI and S3 artifacts storage based on application settings.
    Must be called before any MLflow operations.

    Raises:
        MlflowException: When MLflow setup fails.
    """
    settings = get_settings()

    # Set tracking URI
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    # Set S3 configuration for artifacts
    import os

    os.environ["AWS_ACCESS_KEY_ID"] = settings.aws.access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws.secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.mlflow.s3_endpoint_url or ""

    logger.info(f"MLflow tracking URI set to: {settings.mlflow.tracking_uri}")
    logger.info(f"MLflow S3 endpoint: {settings.mlflow.s3_endpoint_url}")


def get_mlflow_client() -> MlflowClient:
    """
    Get configured MLflow client instance.

    Returns:
        MlflowClient: Configured MLflow client for API operations.
    """
    setup_mlflow_tracking()
    return MlflowClient()


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Get existing experiment or create new one.

    Args:
        experiment_name: Name of the experiment to get or create.

    Returns:
        str: Experiment ID.

    Raises:
        MlflowException: When experiment operations fail.
    """
    client = get_mlflow_client()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            logger.info(f"Using existing experiment: {experiment_name}")
            return experiment.experiment_id
    except MlflowException:
        pass

    # Create new experiment
    experiment_id = client.create_experiment(experiment_name)
    logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    return experiment_id


def start_mlflow_run(
    experiment_name: str, run_name: Optional[str] = None
) -> mlflow.ActiveRun:
    """
    Start MLflow run in specified experiment.

    Args:
        experiment_name: Name of the experiment.
        run_name: Optional name for the run.

    Returns:
        mlflow.ActiveRun: Active MLflow run context.
    """
    setup_mlflow_tracking()
    experiment_id = get_or_create_experiment(experiment_name)

    run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)

    logger.info(f"Started MLflow run: {run.info.run_id}")
    return run


def log_model_metrics(metrics: Dict[str, float]) -> None:
    """
    Log model evaluation metrics to current MLflow run.

    Args:
        metrics: Dictionary of metric names and values.
    """
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    logger.info(f"Logged {len(metrics)} metrics to MLflow")


def log_model_params(params: Dict[str, Any]) -> None:
    """
    Log model parameters to current MLflow run.

    Args:
        params: Dictionary of parameter names and values.
    """
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)

    logger.info(f"Logged {len(params)} parameters to MLflow")


def register_model(model_name: str, model_uri: str) -> str:
    """
    Register model in MLflow model registry.

    Args:
        model_name: Name for the registered model.
        model_uri: URI of the model to register (e.g., from current run).

    Returns:
        str: Version of the registered model.

    Raises:
        MlflowException: When model registration fails.
    """
    client = get_mlflow_client()

    try:
        model_version = client.create_model_version(name=model_name, source=model_uri)

        logger.info(f"Registered model {model_name} version {model_version.version}")
        return model_version.version

    except MlflowException as e:
        logger.error(f"Failed to register model: {str(e)}")
        raise


def get_latest_model_version(
    model_name: str, stage: str = "Production"
) -> Optional[str]:
    """
    Get latest model version from registry for specified stage.

    Args:
        model_name: Name of the registered model.
        stage: Model stage ('Production', 'Staging', 'None').

    Returns:
        Optional[str]: Latest model version URI, None if not found.
    """
    client = get_mlflow_client()

    try:
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        if latest_versions:
            version = latest_versions[0]
            model_uri = f"models:/{model_name}/{version.version}"
            logger.info(f"Found {stage} model: {model_uri}")
            return model_uri
        else:
            logger.warning(f"No {stage} model found for {model_name}")
            return None

    except MlflowException as e:
        logger.error(f"Failed to get model version: {str(e)}")
        return None


def validate_mlflow_connection() -> Tuple[bool, Dict[str, Any]]:
    """
    Validate MLflow server connection and configuration.

    Returns:
        Tuple containing:
            - bool: True if connection is successful.
            - Dict[str, Any]: Validation results with details.
    """
    results = {
        "tracking_server": False,
        "experiments_access": False,
        "s3_artifacts": False,
        "error_details": [],
    }

    try:
        # Test tracking server connection
        client = get_mlflow_client()
        experiments = client.search_experiments()
        results["tracking_server"] = True
        results["experiments_access"] = True

        logger.info(
            f"MLflow connection successful. Found {len(experiments)} experiments."
        )

        # Test S3 artifacts by creating a test run
        settings = get_settings()
        test_experiment = get_or_create_experiment("connection-test")

        with mlflow.start_run(experiment_id=test_experiment):
            mlflow.log_param("test_param", "connection_test")
            results["s3_artifacts"] = True

        logger.info("S3 artifacts connection successful")

    except Exception as e:
        error_msg = f"MLflow validation failed: {str(e)}"
        logger.error(error_msg)
        results["error_details"].append(error_msg)

    overall_success = all(
        [
            results["tracking_server"],
            results["experiments_access"],
            results["s3_artifacts"],
        ]
    )

    return overall_success, results


def cleanup_test_experiments() -> None:
    """
    Clean up test experiments created during validation.

    Removes experiments created for testing purposes.
    """
    client = get_mlflow_client()

    try:
        test_experiment = client.get_experiment_by_name("connection-test")
        if test_experiment:
            client.delete_experiment(test_experiment.experiment_id)
            logger.info("Cleaned up test experiment")
    except MlflowException:
        pass  # Experiment doesn't exist, nothing to clean
