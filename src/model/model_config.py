"""
Model configuration for solar forecasting MLOps pipeline.

This module centralizes all model-related configuration parameters
for XGBoost time series forecasting.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List
)


def get_project_root() -> Path:
    """
    Get the project root directory by looking for key project files.

    Returns:
        Path: Path to the project root directory.
    """
    current_dir = Path(__file__).resolve().parent

    # Go up until we find the project root (contains data/, src/, etc.)
    while current_dir != current_dir.parent:
        if (current_dir / "src").exists() and (current_dir / "data").exists():
            return current_dir
        current_dir = current_dir.parent

    # Fallback: assume we're in src/models and go up two levels
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class XGBoostConfig:
    """
    XGBoost model configuration parameters.

    These are the optimal parameters identified through hyperparameter tuning
    in the model experiments phase.
    """

    n_estimators: int = 50
    max_depth: int = 4
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    n_jobs: int = -1
    verbosity: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for MLflow logging.

        Returns:
            Dict[str, Any]: Configuration parameters as dictionary.
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": self.verbosity,
        }


@dataclass
class TimeSeriesValidationConfig:
    """
    Configuration for time series cross-validation.
    """

    n_splits: int = 3
    test_size: int = 7
    min_train_size: int = 20

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert validation configuration to dictionary.

        Returns:
            Dict[str, Any]: Validation parameters as dictionary.
        """
        return {
            "n_splits": self.n_splits,
            "test_size": self.test_size,
            "min_train_size": self.min_train_size,
        }


@dataclass
class PreprocessorConfig:
    """
    Configuration for solar forecasting preprocessor.
    """

    forecast_horizon: int = 24
    lag_days: List[int] = None
    rolling_windows: List[int] = None
    scaling_method: str = "standard"
    target_frequency: str = "1H"

    def __post_init__(self):
        """Set default values for mutable defaults."""
        if self.lag_days is None:
            self.lag_days = [1, 2, 3, 7, 30]
        if self.rolling_windows is None:
            self.rolling_windows = [7, 30]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert preprocessor configuration to dictionary.

        Returns:
            Dict[str, Any]: Preprocessor parameters as dictionary.
        """
        return {
            "forecast_horizon": self.forecast_horizon,
            "lag_days": self.lag_days,
            "rolling_windows": self.rolling_windows,
            "scaling_method": self.scaling_method,
            "target_frequency": self.target_frequency,
        }


@dataclass
class TrainingConfig:
    """
    Complete training configuration combining all components.
    """

    model: XGBoostConfig = None
    validation: TimeSeriesValidationConfig = None
    preprocessor: PreprocessorConfig = None

    # MLflow configuration
    experiment_name: str = "solar-forecasting-production"
    run_name_prefix: str = "xgboost_training"

    # Data paths (will be resolved to absolute paths)
    generation_data_path: str = "data/raw/Plant_1_Generation_Data.csv"
    weather_data_path: str = "data/raw/Plant_1_Weather_Sensor_Data.csv"

    def __post_init__(self):
        """Initialize default configurations and resolve paths."""
        if self.model is None:
            self.model = XGBoostConfig()
        if self.validation is None:
            self.validation = TimeSeriesValidationConfig()
        if self.preprocessor is None:
            self.preprocessor = PreprocessorConfig()

        # Resolve data paths to absolute paths based on project root
        project_root = get_project_root()
        self.generation_data_path = str(project_root / self.generation_data_path)
        self.weather_data_path = str(project_root / self.weather_data_path)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert complete training configuration to dictionary.

        Returns:
            Dict[str, Any]: Complete configuration as dictionary.
        """
        return {
            "model_config": self.model.to_dict(),
            "validation_config": self.validation.to_dict(),
            "preprocessor_config": self.preprocessor.to_dict(),
            "experiment_name": self.experiment_name,
            "run_name_prefix": self.run_name_prefix,
            "generation_data_path": self.generation_data_path,
            "weather_data_path": self.weather_data_path,
        }


# Note: DEFAULT_TRAINING_CONFIG will be created on first access to avoid circular issues
_DEFAULT_TRAINING_CONFIG = None


def get_training_config(
    generation_data_path: str = None, weather_data_path: str = None
) -> TrainingConfig:
    """
    Get training configuration with optional path overrides.

    Args:
        generation_data_path: Optional override for generation data path.
        weather_data_path: Optional override for weather data path.

    Returns:
        TrainingConfig: Configuration for training pipeline with resolved paths.
    """
    global _DEFAULT_TRAINING_CONFIG

    if _DEFAULT_TRAINING_CONFIG is None:
        _DEFAULT_TRAINING_CONFIG = TrainingConfig()

    # Create a new config based on default
    config = TrainingConfig()

    # Override paths if provided
    if generation_data_path:
        config.generation_data_path = generation_data_path
    if weather_data_path:
        config.weather_data_path = weather_data_path

    return config


def get_default_training_config() -> TrainingConfig:
    """Get default training configuration (function for lazy loading)."""
    global _DEFAULT_TRAINING_CONFIG
    if _DEFAULT_TRAINING_CONFIG is None:
        _DEFAULT_TRAINING_CONFIG = TrainingConfig()
    return _DEFAULT_TRAINING_CONFIG


# For backward compatibility
DEFAULT_TRAINING_CONFIG = get_default_training_config


def get_model_config() -> XGBoostConfig:
    """
    Get default model configuration.

    Returns:
        XGBoostConfig: Default XGBoost parameters.
    """
    return get_default_training_config().model


def get_validation_config() -> TimeSeriesValidationConfig:
    """
    Get default validation configuration.

    Returns:
        TimeSeriesValidationConfig: Default validation parameters.
    """
    return get_default_training_config().validation


def get_preprocessor_config() -> PreprocessorConfig:
    """
    Get default preprocessor configuration.

    Returns:
        PreprocessorConfig: Default preprocessor parameters.
    """
    return get_default_training_config().preprocessor


# Export main classes and functions
__all__ = [
    "XGBoostConfig",
    "TimeSeriesValidationConfig",
    "PreprocessorConfig",
    "TrainingConfig",
    "get_training_config",
    "get_default_training_config",
    "DEFAULT_TRAINING_CONFIG",
    "get_model_config",
    "get_validation_config",
    "get_preprocessor_config",
    "get_project_root",
]
