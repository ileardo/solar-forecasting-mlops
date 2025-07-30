"""
Minimal tests for batch storage functionality.
"""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.batch.storage import PredictionStorage


class TestPredictionStorage:
    """Minimal tests for PredictionStorage class."""

    @patch("src.batch.storage.get_settings")
    @patch("src.batch.storage.psycopg2.connect")
    def test_init_creates_table(self, mock_connect, mock_get_settings, mock_settings):
        """Test that PredictionStorage initializes and creates table."""
        mock_get_settings.return_value = mock_settings
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn

        storage = PredictionStorage()

        assert storage.settings == mock_settings
        mock_conn.cursor.assert_called()

    @patch("src.batch.storage.get_settings")
    @patch("src.batch.storage.psycopg2.connect")
    def test_save_prediction_success(
        self, mock_connect, mock_get_settings, mock_settings, sample_prediction_results
    ):
        """Test successful prediction save."""
        mock_get_settings.return_value = mock_settings
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [123]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_conn

        storage = PredictionStorage()
        result_id = storage.save_prediction(sample_prediction_results)

        assert result_id == 123
        mock_cursor.execute.assert_called()

    @patch("src.batch.storage.get_settings")
    @patch("src.batch.storage.psycopg2.connect")
    def test_health_check_healthy(self, mock_connect, mock_get_settings, mock_settings):
        """Test health check returns healthy status."""
        mock_get_settings.return_value = mock_settings
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [42]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_conn

        storage = PredictionStorage()
        health = storage.health_check()

        assert health["healthy"] is True
        assert health["total_predictions"] == 42
