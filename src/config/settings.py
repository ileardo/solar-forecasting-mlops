"""
Solar Forecasting MLOps - Simplified Configuration Management.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv


class Settings:
    """
    Simple, direct configuration class.

    Loads environment variables explicitly and provides clean access
    to all configuration needed for MLOps pipeline.
    """

    def __init__(self) -> None:
        """Initialize settings with explicit .env loading."""
        # Load .env file explicitly - no magic
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)

        # Database configuration (core essentials)
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = int(os.getenv("DB_PORT", "5432"))
        self.db_name = os.getenv("DB_NAME", "solar_forecasting")
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "password")

        # AWS/LocalStack configuration (minimal)
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "test")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "test")
        self.aws_default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.aws_endpoint_url = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")

        # S3 buckets
        self.s3_bucket_artifacts = os.getenv("S3_BUCKET_ARTIFACTS", "mlflow-artifacts")

        # MLflow configuration (derived from database)
        self.mlflow_port = int(os.getenv("MLFLOW_PORT", "5000"))
        self.mlflow_tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", f"http://localhost:{self.mlflow_port}"
        )
        self.mlflow_s3_endpoint_url = os.getenv(
            "MLFLOW_S3_ENDPOINT_URL", self.aws_endpoint_url
        )
        self.mlflow_artifact_root = os.getenv(
            "MLFLOW_ARTIFACT_ROOT", f"s3://{self.s3_bucket_artifacts}"
        )
        self.mlflow_experiment_name = os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "solar-forecasting-experiment"
        )

        # Application settings (simple)
        self.app_name = os.getenv("APP_NAME", "solar-forecasting-mlops")
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"

    @property
    def database_url(self) -> str:
        """Get database connection URL."""
        # pylint: disable=line-too-long
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def is_development(self) -> bool:
        """Check if in development mode."""
        return self.environment.lower() == "development"

    @property
    def is_localstack(self) -> bool:
        """Check if using LocalStack."""
        return "localhost" in self.aws_endpoint_url

    def get_aws_config(self) -> Dict[str, str]:
        """Get AWS configuration for boto3."""
        return {
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "region_name": self.aws_default_region,
            "endpoint_url": self.aws_endpoint_url if self.is_localstack else None,
        }

    def validate(self) -> Dict[str, Any]:
        """
        Simple validation of critical settings.

        Returns:
            Dict with validation results.
        """
        results = {
            "env_file_exists": Path(".env").exists(),
            "db_password_set": self.db_password != "password",
            "aws_configured": bool(
                self.aws_access_key_id and self.aws_secret_access_key
            ),
            "mlflow_uri_valid": bool(self.mlflow_tracking_uri),
        }

        results["overall"] = all(
            [
                results["env_file_exists"],
                results["db_password_set"],
                results["aws_configured"],
                results["mlflow_uri_valid"],
            ]
        )

        return results

    def debug_info(self) -> Dict[str, Any]:
        """Get debug information (safe for logging)."""
        return {
            "app_name": self.app_name,
            "environment": self.environment,
            "debug": self.debug,
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_name": self.db_name,
            "db_user": self.db_user,
            "db_password_length": len(self.db_password),
            "aws_endpoint": self.aws_endpoint_url,
            "is_localstack": self.is_localstack,
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
            "s3_bucket_artifacts": self.s3_bucket_artifacts,
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Configured settings instance.
    """
    return Settings()


def setup_environment_for_mlflow() -> None:
    """
    Setup environment variables specifically for MLflow.

    This function sets the necessary environment variables that MLflow
    expects, using our configuration.
    """
    settings = get_settings()

    # Set MLflow environment variables
    os.environ["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
    os.environ["AWS_DEFAULT_REGION"] = settings.aws_default_region
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.mlflow_s3_endpoint_url


def validate_configuration() -> bool:
    """
    Quick validation of configuration.

    Returns:
        bool: True if configuration is valid.
    """
    settings = get_settings()
    validation = settings.validate()

    if not validation["overall"]:
        print("❌ Configuration validation failed:")
        for key, value in validation.items():
            if key != "overall" and not value:
                print(f"   - {key}: {value}")
        return False

    print("✅ Configuration validation passed")
    return True


# Export commonly used functions
__all__ = [
    "Settings",
    "get_settings",
    "setup_environment_for_mlflow",
    "validate_configuration",
]
