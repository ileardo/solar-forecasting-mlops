"""
Minimal tests for batch predictor functionality.
"""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.batch.predictor import BatchPredictor


class TestBatchPredictor:
    """Minimal tests for BatchPredictor class."""

    @patch("src.batch.predictor.ModelRegistry")
    @patch("src.batch.predictor.SimpleDriftDetector")
    def test_init_with_drift_detector(self, mock_drift_detector, mock_registry):
        """Test BatchPredictor initialization with drift detector."""
        mock_drift_detector.return_value = Mock()

        predictor = BatchPredictor("test-model")

        assert predictor.model_name == "test-model"
        assert predictor.is_loaded is False
        assert predictor.drift_monitoring_enabled is True

    @patch("src.batch.predictor.ModelRegistry")
    @patch("src.batch.predictor.SimpleDriftDetector")
    def test_init_without_drift_detector(self, mock_drift_detector, mock_registry):
        """Test BatchPredictor initialization when drift detector fails."""
        mock_drift_detector.side_effect = Exception("Drift detector failed")

        predictor = BatchPredictor("test-model")

        assert predictor.model_name == "test-model"
        assert predictor.drift_monitoring_enabled is False

    @patch("src.batch.predictor.ModelRegistry")
    @patch("src.batch.predictor.SimpleDriftDetector")
    def test_load_production_model(
        self, mock_drift_detector, mock_registry, mock_model, mock_preprocessor
    ):
        """Test loading production model."""
        mock_registry_instance = Mock()
        mock_registry_instance.load_production_model.return_value = (
            mock_model,
            mock_preprocessor,
        )
        mock_registry.return_value = mock_registry_instance

        predictor = BatchPredictor("test-model")
        predictor.load_production_model()

        assert predictor.is_loaded is True
        assert predictor.model == mock_model
        assert predictor.preprocessor == mock_preprocessor

    @patch("src.batch.predictor.ModelRegistry")
    @patch("src.batch.predictor.SimpleDriftDetector")
    def test_get_model_info_when_loaded(
        self, mock_drift_detector, mock_registry, mock_model, mock_preprocessor
    ):
        """Test getting model info when model is loaded."""
        predictor = BatchPredictor("test-model")
        predictor.model = mock_model
        predictor.preprocessor = mock_preprocessor
        predictor.is_loaded = True

        info = predictor.get_model_info()

        assert info["model_name"] == "test-model"
        assert info["is_loaded"] is True
        assert info["features_count"] == 31
        assert info["forecast_horizon"] == 24
