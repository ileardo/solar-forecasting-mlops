"""
Solar Forecasting MLOps - Configuration Management.

This module handles all environment variables and configuration settings
for the MLOps pipeline, providing type-safe access to all service configurations.
"""

# pylint: disable=line-too-long, broad-exception-caught

from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional
)

from pydantic import (
    Field,
    field_validator,
    model_validator
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="DB_", env_file=".env", case_sensitive=False
    )

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="solar_forecasting", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(
        default="password", description="Database password"
    )  # Made optional with default
    db_schema: str = Field(
        default="public", description="Database schema", alias="DB_SCHEMA"
    )  # Fixed shadow issue

    # Connection pool settings
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max pool overflow")
    pool_timeout: int = Field(default=30, description="Pool timeout seconds")
    pool_recycle: int = Field(default=3600, description="Pool recycle seconds")

    @property
    def url(self) -> str:
        """Generate database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def async_url(self) -> str:
        """Generate async database URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def test_url(self) -> str:
        """Generate test database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}_test"


class MLflowSettings(BaseSettings):
    """MLflow configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="MLFLOW_", env_file=".env", case_sensitive=False
    )

    tracking_uri: Optional[str] = Field(default=None, description="MLflow tracking URI")
    artifact_root: str = Field(
        default="s3://mlflow-artifacts", description="Artifact storage root"
    )
    s3_endpoint_url: Optional[str] = Field(
        default="http://localhost:4566", description="S3 endpoint URL"
    )
    experiment_name: str = Field(
        default="solar-forecasting-experiment", description="Default experiment name"
    )

    # UI settings
    ui_host: str = Field(default="0.0.0.0", description="MLflow UI host")
    ui_port: int = Field(default=5000, description="MLflow UI port")
    backend_store_uri: Optional[str] = Field(
        default=None, description="Backend store URI"
    )
    default_artifact_root: Optional[str] = Field(
        default=None, description="Default artifact root"
    )

    @field_validator("tracking_uri", mode="before")
    @classmethod
    def set_default_tracking_uri(cls, v: Optional[str]) -> str:
        """Set default tracking URI if not provided."""
        if v is None:
            return "sqlite:///mlflow.db"
        return v


class AWSSettings(BaseSettings):
    """AWS/Localstack configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="AWS_", env_file=".env", case_sensitive=False
    )

    access_key_id: str = Field(default="test", description="AWS access key ID")
    secret_access_key: str = Field(default="test", description="AWS secret access key")
    default_region: str = Field(default="us-east-1", description="AWS default region")
    endpoint_url: Optional[str] = Field(
        default="http://localhost:4566", description="AWS endpoint URL"
    )

    # S3 Buckets
    s3_bucket_artifacts: str = Field(
        default="mlflow-artifacts", alias="S3_BUCKET_ARTIFACTS"
    )
    s3_bucket_data: str = Field(default="solar-data", alias="S3_BUCKET_DATA")
    s3_bucket_models: str = Field(default="solar-models", alias="S3_BUCKET_MODELS")
    s3_bucket_monitoring: str = Field(
        default="monitoring-data", alias="S3_BUCKET_MONITORING"
    )

    # Localstack settings
    localstack_host: str = Field(default="localhost", alias="LOCALSTACK_HOST")
    localstack_port: int = Field(default=4566, alias="LOCALSTACK_PORT")
    localstack_services: str = Field(
        default="s3,rds,lambda", alias="LOCALSTACK_SERVICES"
    )

    @property
    def is_localstack(self) -> bool:
        """Check if using localstack."""
        return self.endpoint_url is not None and "localhost" in self.endpoint_url

    @property
    def s3_client_kwargs(self) -> Dict[str, Any]:
        """Get S3 client configuration."""
        kwargs = {
            "aws_access_key_id": self.access_key_id,
            "aws_secret_access_key": self.secret_access_key,
            "region_name": self.default_region,
        }
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        return kwargs


class PrefectSettings(BaseSettings):
    """Prefect workflow orchestration settings."""

    model_config = SettingsConfigDict(
        env_prefix="PREFECT_", env_file=".env", case_sensitive=False
    )

    api_url: str = Field(
        default="http://localhost:4200/api", description="Prefect API URL"
    )
    ui_url: str = Field(default="http://localhost:4200", description="Prefect UI URL")
    logging_level: str = Field(default="INFO", description="Prefect logging level")
    home: str = Field(default="./prefect_home", description="Prefect home directory")
    orion_database_connection_url: Optional[str] = Field(
        default=None, description="Prefect database URL"
    )


class MonitoringSettings(BaseSettings):
    """Monitoring and drift detection settings."""

    model_config = SettingsConfigDict(
        env_prefix="MONITORING_", env_file=".env", case_sensitive=False
    )

    evidently_workspace: str = Field(
        default="./evidently_workspace", description="Evidently workspace path"
    )
    batch_size: int = Field(default=1000, description="Monitoring batch size")
    drift_detection_threshold: float = Field(
        default=0.1, description="Drift detection threshold"
    )
    performance_threshold_rmse: float = Field(
        default=0.5, description="Performance threshold RMSE"
    )

    # Table names
    metrics_table_name: str = Field(
        default="monitoring_metrics", alias="METRICS_TABLE_NAME"
    )
    drift_table_name: str = Field(default="drift_reports", alias="DRIFT_TABLE_NAME")
    performance_table_name: str = Field(
        default="model_performance", alias="PERFORMANCE_TABLE_NAME"
    )


class GrafanaSettings(BaseSettings):
    """Grafana dashboard configuration."""

    model_config = SettingsConfigDict(
        env_prefix="GRAFANA_", env_file=".env", case_sensitive=False
    )

    host: str = Field(default="localhost", description="Grafana host")
    port: int = Field(default=3000, description="Grafana port")
    user: str = Field(default="admin", description="Grafana user")
    password: str = Field(default="admin", description="Grafana password")
    database_url: Optional[str] = Field(
        default=None, description="Grafana database URL"
    )

    # Organization settings
    org_name: str = Field(default="Solar Forecasting MLOps", alias="GRAFANA_ORG_NAME")
    org_id: int = Field(default=1, alias="GRAFANA_ORG_ID")

    @property
    def url(self) -> str:
        """Generate Grafana URL."""
        return f"http://{self.host}:{self.port}"


class ModelSettings(BaseSettings):
    """Model training and registry configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MODEL_", env_file=".env", case_sensitive=False
    )

    model_type: str = Field(default="random_forest", description="Model type")
    target_column: str = Field(default="AC_POWER", description="Target column name")
    features_weather: str = Field(
        default="AMBIENT_TEMPERATURE,MODULE_TEMPERATURE,IRRADIATION",
        description="Weather features (comma-separated)",
    )
    features_temporal: str = Field(
        default="hour,day_of_year,season",
        description="Temporal features (comma-separated)",
    )

    # Training parameters
    train_test_split_ratio: float = Field(
        default=0.8, description="Train/test split ratio"
    )
    validation_split_ratio: float = Field(
        default=0.2, description="Validation split ratio"
    )
    random_seed: int = Field(default=42, description="Random seed")
    cv_folds: int = Field(default=5, description="Cross-validation folds")

    # Model registry
    registry_name: str = Field(
        default="solar-forecasting-models", alias="MODEL_REGISTRY_NAME"
    )
    stage_staging: str = Field(default="Staging", alias="MODEL_STAGE_STAGING")
    stage_production: str = Field(default="Production", alias="MODEL_STAGE_PRODUCTION")
    version_alias: str = Field(default="latest", alias="MODEL_VERSION_ALIAS")

    @property
    def weather_features_list(self) -> List[str]:
        """Get weather features as list."""
        return [f.strip() for f in self.features_weather.split(",")]

    @property
    def temporal_features_list(self) -> List[str]:
        """Get temporal features as list."""
        return [f.strip() for f in self.features_temporal.split(",")]


class BatchSettings(BaseSettings):
    """Batch prediction configuration."""

    model_config = SettingsConfigDict(
        env_prefix="BATCH_", env_file=".env", case_sensitive=False
    )

    prediction_schedule: str = Field(
        default="0 23 * * *", description="Prediction cron schedule"
    )
    batch_size: int = Field(default=1000, description="Batch processing size")
    prediction_horizon_hours: int = Field(
        default=24, description="Prediction horizon in hours"
    )
    output_table: str = Field(default="predictions", alias="BATCH_OUTPUT_TABLE")

    # Data processing
    data_refresh_schedule: str = Field(
        default="0 22 * * *", alias="DATA_REFRESH_SCHEDULE"
    )
    data_retention_days: int = Field(default=90, alias="DATA_RETENTION_DAYS")
    data_validation_enabled: bool = Field(default=True, alias="DATA_VALIDATION_ENABLED")


class APISettings(BaseSettings):
    """API service configuration."""

    model_config = SettingsConfigDict(
        env_prefix="API_", env_file=".env", case_sensitive=False
    )

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=4, description="API workers")
    reload: bool = Field(default=True, description="API reload on changes")

    # Security (made optional for development)
    secret_key: str = Field(
        default="dev-secret-key-change-in-production", description="API secret key"
    )
    access_token_expire_minutes: int = Field(
        default=30, description="Token expiration minutes"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",
    )

    # Application settings
    app_name: str = Field(default="solar-forecasting-mlops", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    aws: AWSSettings = Field(default_factory=AWSSettings)
    prefect: PrefectSettings = Field(default_factory=PrefectSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    grafana: GrafanaSettings = Field(default_factory=GrafanaSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    batch: BatchSettings = Field(default_factory=BatchSettings)
    api: APISettings = Field(default_factory=APISettings)

    @property
    def is_development(self) -> bool:
        """Check if in development environment."""
        return self.environment.lower() == "development"

    @property
    def is_production(self) -> bool:
        """Check if in production environment."""
        return self.environment.lower() == "production"

    @property
    def is_testing(self) -> bool:
        """Check if in testing environment."""
        return self.environment.lower() == "testing"

    @model_validator(mode="after")
    def setup_cross_references(self) -> "Settings":
        """Set up cross-references between settings."""
        # Set MLflow tracking URI to database URL if using default
        if self.mlflow.tracking_uri == "sqlite:///mlflow.db":
            self.mlflow.tracking_uri = self.database.url

        # Set Prefect database URL if not explicitly set
        if self.prefect.orion_database_connection_url is None:
            self.prefect.orion_database_connection_url = self.database.url

        # Set Grafana database URL if not explicitly set
        if self.grafana.database_url is None:
            self.grafana.database_url = self.database.url

        return self


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns:
        Settings: The application settings instance.
    """
    return Settings()


def validate_required_env_vars() -> Dict[str, bool]:
    """
    Validate that all required environment variables are set.

    Returns:
        Dict[str, bool]: Validation results for each component.
    """
    results = {}

    try:
        settings = get_settings()

        # Test database connection settings
        results["database"] = bool(
            settings.database.password != "password"
        )  # Check if changed from default

        # Test AWS/S3 settings
        results["aws"] = bool(
            settings.aws.access_key_id and settings.aws.secret_access_key
        )

        # Test API security
        results["api"] = (
            settings.api.secret_key != "dev-secret-key-change-in-production"
        )

        # Test critical paths
        results["paths"] = all(
            [
                Path(settings.monitoring.evidently_workspace).parent.exists()
                or settings.monitoring.evidently_workspace.startswith("./"),
                Path(settings.prefect.home).parent.exists()
                or settings.prefect.home.startswith("./"),
            ]
        )

        # Environment file check
        results["env_file"] = Path(".env").exists()

        results["overall"] = all(results.values())

    except Exception as exception_error:
        results["error"] = str(exception_error)
        results["overall"] = False

    return results


def get_connection_urls() -> Dict[str, str]:
    """
    Get all service connection URLs.

    Returns:
        Dict[str, str]: Service URLs for health checks.
    """
    settings = get_settings()

    return {
        "database": settings.database.url,
        "mlflow_tracking": settings.mlflow.tracking_uri,
        "mlflow_ui": f"http://{settings.mlflow.ui_host}:{settings.mlflow.ui_port}",
        "prefect_api": settings.prefect.api_url,
        "prefect_ui": settings.prefect.ui_url,
        "grafana": settings.grafana.url,
        "localstack": f"http://{settings.aws.localstack_host}:{settings.aws.localstack_port}",
    }


# Export commonly used settings
__all__ = [
    "Settings",
    "get_settings",
    "validate_required_env_vars",
    "get_connection_urls",
]
