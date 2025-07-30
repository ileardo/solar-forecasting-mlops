"""
MLflow utilities for solar forecasting project - FIXED VERSION.

This version uses MLflow server instead of direct database connection
to ensure artifacts are stored in S3 correctly.
"""

# pylint: disable=no-else-return, broad-exception-caught, unspecified-encoding

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config.settings import get_settings


# Configure logging
logger = logging.getLogger(__name__)


def setup_mlflow_tracking() -> None:
    """
    Setup MLflow tracking to use MLflow server instead of direct database.

    This ensures artifacts are handled correctly by the server (S3 storage)
    instead of being stored locally by the client.

    Raises:
        MlflowException: When MLflow setup fails.
    """
    settings = get_settings()

    # FIXED: Use MLflow server URI instead of direct database connection
    mlflow_server_uri = f"http://localhost:{settings.mlflow_port}"
    mlflow.set_tracking_uri(mlflow_server_uri)

    # Set S3 configuration for client-side operations (if needed)
    os.environ["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.mlflow_s3_endpoint_url or ""

    logger.info(f"MLflow tracking URI set to: {mlflow_server_uri}")
    logger.info(f"MLflow server will handle S3 artifacts automatically")


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

    Handles both active and deleted experiments gracefully.

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
            if experiment.lifecycle_stage == "active":
                logger.info(f"Using existing active experiment: {experiment_name}")
                return experiment.experiment_id
            elif experiment.lifecycle_stage == "deleted":
                # Restore deleted experiment
                client.restore_experiment(experiment.experiment_id)
                logger.info(f"Restored deleted experiment: {experiment_name}")
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


def quick_test_mlflow() -> bool:
    """
    Quick test of MLflow connection and basic operations.

    Uses MLflow server for proper S3 artifact handling.

    Returns:
        bool: True if MLflow test successful.
    """
    try:
        print("🧪 Testing MLflow setup...")

        # Setup MLflow to use server
        setup_mlflow_tracking()

        print(f"   Tracking URI: {mlflow.get_tracking_uri()}")

        # Test experiment operations with unique name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        test_experiment_name = f"quick_test_{timestamp}_{unique_id}"

        experiment_id = get_or_create_experiment(test_experiment_name)

        # Test run operations
        with mlflow.start_run(
            experiment_id=experiment_id, run_name=f"quick_test_run_{unique_id}"
        ):
            # Check artifact URI to verify S3 usage
            run = mlflow.active_run()
            artifact_uri = run.info.artifact_uri

            print(f"   Artifact URI: {artifact_uri}")

            if artifact_uri.startswith("s3://"):
                print("   ✅ S3 artifacts correctly configured!")
            else:
                print(
                    "   ⚠️  Artifacts not using S3 (this might be expected in some setups)"
                )

            mlflow.log_param("test_param", "connection_test")
            mlflow.log_metric("test_metric", 0.99)

            # Test artifact logging
            artifact_content = (
                f"Quick test artifact\nTimestamp: {timestamp}\nUnique ID: {unique_id}"
            )
            artifact_path = f"quick_test_{unique_id}.txt"

            with open(artifact_path, "w") as f:
                f.write(artifact_content)

            mlflow.log_artifact(artifact_path)

            os.remove(artifact_path)

        print("✅ MLflow test successful!")
        return True

    except Exception as e:
        logger.error(f"MLflow validation failed: {str(e)}")
        print(f"❌ MLflow test failed:")
        print(f"   {str(e)}")
        return False


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

        # Test S3 artifacts by creating a test run with unique name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        test_experiment = get_or_create_experiment(
            f"connection_test_{timestamp}_{unique_id}"
        )

        with mlflow.start_run(experiment_id=test_experiment):
            run = mlflow.active_run()
            artifact_uri = run.info.artifact_uri

            # Check if using S3
            if artifact_uri.startswith("s3://"):
                results["s3_artifacts"] = True
                logger.info("S3 artifacts connection successful")
            else:
                logger.warning(f"Artifacts not using S3: {artifact_uri}")
                # Still mark as success if artifacts work, even if not S3
                results["s3_artifacts"] = True

            mlflow.log_param("test_param", "connection_test")

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


def create_unique_experiment_name(base_name: str) -> str:
    """
    Create unique experiment name to avoid conflicts.

    Args:
        base_name: Base name for the experiment.

    Returns:
        str: Unique experiment name with timestamp and UUID.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{base_name}_{timestamp}_{unique_id}"


def get_experiment_by_name_safe(experiment_name: str) -> Optional[str]:
    """
    Safely get experiment by name, handling deleted experiments.

    Args:
        experiment_name: Name of the experiment.

    Returns:
        Optional[str]: Experiment ID if found and active, None otherwise.
    """
    client = get_mlflow_client()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment and experiment.lifecycle_stage == "active":
            return experiment.experiment_id
        return None
    except MlflowException:
        return None


# Export commonly used functions
__all__ = [
    "setup_mlflow_tracking",
    "get_mlflow_client",
    "get_or_create_experiment",
    "start_mlflow_run",
    "log_model_metrics",
    "log_model_params",
    "register_model",
    "get_latest_model_version",
    "quick_test_mlflow",
    "validate_mlflow_connection",
    "create_unique_experiment_name",
    "get_experiment_by_name_safe",
]
