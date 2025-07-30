"""
Pytest configuration and fixtures for solar forecasting tests.
"""

from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest


@pytest.fixture
def mock_settings():
    """Mock settings for database configuration."""
    settings = Mock()
    settings.db_host = "localhost"
    settings.db_port = 5432
    settings.db_name = "test_db"
    settings.db_user = "test_user"
    settings.db_password = "test_password"
    return settings


@pytest.fixture
def sample_prediction_results():
    """Sample prediction results for testing."""
    return {
        "prediction_date": "2020-06-15",
        "model_name": "solar-forecasting-prod",
        "forecast_24h": [100.0] * 24,
        "peak_power": 500.0,
        "peak_hour": 12,
        "total_energy": 2400.0,
        "daylight_energy": 1200.0,
        "features_count": 31,
        "prediction_timestamp": "2020-06-15T00:00:00",
    }


@pytest.fixture
def mock_model():
    """Mock trained model for testing."""
    model = Mock()
    model.predict.return_value = [[100.0] * 24]
    return model


@pytest.fixture
def mock_preprocessor():
    """Mock preprocessor for testing."""
    preprocessor = Mock()
    preprocessor.get_feature_names.return_value = [
        "feature_" + str(i) for i in range(31)
    ]
    preprocessor.forecast_horizon = 24
    preprocessor.scaling_method = "standard"
    return preprocessor
