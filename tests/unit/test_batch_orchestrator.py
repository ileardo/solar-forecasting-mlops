"""
Minimal tests for batch orchestrator functionality.
"""

from unittest.mock import Mock, patch

import pytest

from src.batch.orchestrator import log_results_task, validate_results_task


class TestOrchestrator:
    """Minimal tests for orchestrator tasks."""

    def test_validate_results_task_valid_data(self, sample_prediction_results):
        """Test validation task with valid prediction results."""
        # Create a mock logger
        with patch("src.batch.orchestrator.get_run_logger") as mock_logger:
            mock_logger.return_value = Mock()

            validated_results = validate_results_task(sample_prediction_results)

            assert "validation" in validated_results
            assert validated_results["validation"]["validation_passed"] is True
            assert validated_results["validation"]["negative_predictions"] == 0

    def test_validate_results_task_invalid_length(self):
        """Test validation task with invalid forecast length."""
        invalid_results = {
            "prediction_date": "2020-06-15",
            "forecast_24h": [100.0] * 12,  # Wrong length
            "peak_power": 500.0,
            "peak_hour": 12,
        }

        with patch("src.batch.orchestrator.get_run_logger") as mock_logger:
            mock_logger.return_value = Mock()

            with pytest.raises(
                ValueError, match="Invalid forecast length: 12, expected 24"
            ):
                validate_results_task(invalid_results)

    def test_log_results_task(self, sample_prediction_results):
        """Test logging task runs without errors."""
        with patch("src.batch.orchestrator.get_run_logger") as mock_logger:
            mock_task_logger = Mock()
            mock_logger.return_value = mock_task_logger

            # Should not raise any exception
            log_results_task(sample_prediction_results)

            # Verify logger was called multiple times
            assert mock_task_logger.info.call_count > 5

    def test_log_results_task_with_validation(self, sample_prediction_results):
        """Test logging task with validation results."""
        # Add validation info to results
        sample_prediction_results["validation"] = {
            "validation_passed": True,
            "negative_predictions": 2,
        }

        with patch("src.batch.orchestrator.get_run_logger") as mock_logger:
            mock_task_logger = Mock()
            mock_logger.return_value = mock_task_logger

            log_results_task(sample_prediction_results)

            # Verify validation info was logged
            assert mock_task_logger.info.call_count > 7
