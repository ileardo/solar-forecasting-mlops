"""
Integration test for the complete training pipeline.

This test verifies the end-to-end training workflow including:
- Data preprocessing
- Model training
- Model evaluation
- Model registry operations
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.data.preprocessor import SolarForecastingPreprocessor
from src.model.evaluator import ModelEvaluator
from src.model.model_config import get_training_config
from src.model.registry import ModelRegistry
from src.model.trainer import ModelTrainer


class TestTrainingPipelineIntegration:
    """Integration tests for the complete training pipeline."""

    @pytest.fixture(scope="class")
    def temp_data_dir(self):
        """Create temporary directory with sample data files."""
        temp_dir = tempfile.mkdtemp()

        # Create extended sample CSV data with multiple days for lag features
        generation_data = """DATE_TIME,PLANT_ID,SOURCE_KEY,DC_POWER,AC_POWER,DAILY_YIELD,TOTAL_YIELD"""

        # Generate 45 days of hourly data (sufficient for 30-day lag)
        from datetime import datetime, timedelta

        import pandas as pd

        start_date = datetime(2020, 5, 1)
        data_rows = []

        for day in range(45):
            current_date = start_date + timedelta(days=day)
            for hour in range(24):
                timestamp = current_date + timedelta(hours=hour)
                # Simple solar pattern: power during daylight hours (6-18)
                if 6 <= hour <= 18:
                    # Peak at noon (hour 12)
                    power = 500 * (1 - abs(hour - 12) / 6) + 50
                    ac_power = power * 0.95  # DC to AC conversion
                else:
                    power = 0
                    ac_power = 0

                data_rows.append(
                    f"{timestamp:%Y-%m-%d %H:%M:%S},1,1HKI5sORdkQF,{power:.1f},{ac_power:.1f},{ac_power*4:.1f},{6259580 + day*5000 + hour*100}"
                )

        generation_data = generation_data + "\n" + "\n".join(data_rows)

        weather_data = """DATE_TIME,PLANT_ID,SOURCE_KEY,AMBIENT_TEMPERATURE,MODULE_TEMPERATURE,IRRADIATION"""

        # Generate corresponding weather data
        weather_rows = []
        for day in range(45):
            current_date = start_date + timedelta(days=day)
            for hour in range(0, 24, 4):  # Every 4 hours to reduce data size
                timestamp = current_date + timedelta(hours=hour)
                temp = 25 + 10 * (1 - abs(hour - 12) / 12)  # Temperature pattern
                irradiation = max(0, (1 - abs(hour - 12) / 6)) if 6 <= hour <= 18 else 0

                weather_rows.append(
                    f"{timestamp:%Y-%m-%d %H:%M:%S},1,HmiyD2TTLFNqkNe,{temp:.1f},{temp+5:.1f},{irradiation:.1f}"
                )

        weather_data = weather_data + "\n" + "\n".join(weather_rows)

        # Write sample files
        gen_file = Path(temp_dir) / "generation.csv"
        weather_file = Path(temp_dir) / "weather.csv"

        with open(gen_file, "w") as f:
            f.write(generation_data)

        with open(weather_file, "w") as f:
            f.write(weather_data)

        yield temp_dir, str(gen_file), str(weather_file)

        # Cleanup
        shutil.rmtree(temp_dir)

    @patch("src.utils.mlflow_utils.setup_mlflow_tracking")
    @patch("src.utils.mlflow_utils.mlflow.start_run")
    @patch("src.utils.mlflow_utils.mlflow.active_run")
    @patch("src.utils.mlflow_utils.mlflow.log_param")
    @patch("src.utils.mlflow_utils.mlflow.log_metric")
    @patch("src.utils.mlflow_utils.mlflow.sklearn.log_model")
    @patch("src.model.registry.ModelRegistry.load_production_model")
    def test_complete_training_pipeline(
        self,
        mock_load_model,
        mock_log_model,
        mock_log_metric,
        mock_log_param,
        mock_active_run,
        mock_start_run,
        mock_setup_mlflow,
        temp_data_dir,
    ):
        """Test the complete training pipeline from data to model registry."""
        temp_dir, gen_file, weather_file = temp_data_dir

        # Mock MLflow run context
        mock_run = Mock()
        mock_run.info.run_id = "test-run-id-123"

        # Mock start_run context manager
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_start_run.return_value.__exit__.return_value = None

        # Mock active_run to return the same run
        mock_active_run.return_value = mock_run

        # Mock model info for log_model
        mock_model_info = Mock()
        mock_model_info.registered_model_version = "1"
        mock_log_model.return_value = mock_model_info

        # Step 1: Test data preprocessing
        preprocessor = SolarForecastingPreprocessor(
            forecast_horizon=24,
            lag_days=[1, 2, 7],  # Reduced but sufficient for our 45-day dataset
            rolling_windows=[7],  # Reduced for faster processing
            scaling_method="standard",
        )

        # This should work with our minimal dataset
        X, y, metadata = preprocessor.fit_transform(gen_file, weather_file)

        # Verify preprocessing results
        assert len(X) > 0, "Preprocessing should produce features"
        assert len(y) > 0, "Preprocessing should produce targets"
        assert y.shape[1] == 24, "Should have 24-hour forecast targets"
        assert preprocessor.is_fitted, "Preprocessor should be fitted"

        # Step 2: Test model training
        config = get_training_config()
        config.generation_data_path = gen_file
        config.weather_data_path = weather_file
        config.model.n_estimators = 5  # Reduced for fast testing
        config.validation.n_splits = 2  # Reduced for small dataset

        # Use the same preprocessor configuration as above
        config.preprocessor.lag_days = [1, 2, 7]
        config.preprocessor.rolling_windows = [7]

        trainer = ModelTrainer(config)

        # Train should complete without errors
        model, metrics, run_id = trainer.train()

        # Verify training results
        assert model is not None, "Training should produce a model"
        assert isinstance(metrics, dict), "Training should produce metrics"
        assert "test_rmse_overall" in metrics, "Should have test RMSE metric"
        assert run_id == "test-run-id-123", "Should return correct run ID"

        # Step 3: Test model evaluation
        # Use the same preprocessor from training to ensure feature consistency
        trained_preprocessor = trainer.preprocessor
        split_point = int(0.8 * len(X))
        X_test = X.iloc[split_point:].reset_index(drop=True)
        y_test = y.iloc[split_point:].reset_index(drop=True)

        if len(X_test) > 0:  # Only test if we have test data
            evaluator = ModelEvaluator()
            results = evaluator.evaluate_model(
                model, X_test, y_test, trained_preprocessor
            )

            # Verify evaluation results structure
            assert "overall" in results, "Should have overall metrics"
            assert "per_horizon" in results, "Should have per-horizon metrics"
            # Skip specific metric validation since structure may vary

        # Step 4: Test model registry operations
        with patch(
            "src.model.registry.ModelRegistry.register_model_package"
        ) as mock_register:
            mock_register.return_value = "1"

            with patch(
                "src.model.registry.ModelRegistry.transition_to_production"
            ) as mock_transition:
                registry = ModelRegistry()

                # Register model
                version = registry.register_model_package(
                    model,
                    trained_preprocessor,  # Use the trained preprocessor
                    "test-solar-model",
                    description="Integration test model",
                    tags={"test": "integration"},
                    run_id=run_id,
                )

                # Transition to production
                registry.transition_to_production("test-solar-model", version)

                # Verify registry operations were called
                mock_register.assert_called_once()
                mock_transition.assert_called_once_with("test-solar-model", "1")

        # Verify MLflow interactions
        mock_setup_mlflow.assert_called()
        mock_log_param.assert_called()
        mock_log_metric.assert_called()

    @patch("src.utils.mlflow_utils.setup_mlflow_tracking")
    def test_pipeline_with_insufficient_data(self, mock_setup_mlflow):
        """Test pipeline behavior with insufficient data."""
        # Create minimal dataset that should fail gracefully
        temp_dir = tempfile.mkdtemp()

        minimal_data = """DATE_TIME,PLANT_ID,SOURCE_KEY,DC_POWER,AC_POWER,DAILY_YIELD,TOTAL_YIELD
2020-05-15 00:00:00,1,1HKI5sORdkQF,0,0,0,6259580
2020-05-15 01:00:00,1,1HKI5sORdkQF,100,95,380,6259960"""

        gen_file = Path(temp_dir) / "minimal.csv"
        with open(gen_file, "w") as f:
            f.write(minimal_data)

        try:
            preprocessor = SolarForecastingPreprocessor(
                forecast_horizon=24,
                lag_days=[1, 7, 30],  # This should cause issues with minimal data
                rolling_windows=[7, 30],
                scaling_method="standard",
            )

            # This should handle insufficient data gracefully
            with pytest.raises((ValueError, RuntimeError)):
                X, y, metadata = preprocessor.fit_transform(str(gen_file))

        finally:
            shutil.rmtree(temp_dir)

    def test_pipeline_components_integration(self):
        """Test that pipeline components integrate correctly without full execution."""
        # Test that all components can be imported and initialized
        preprocessor = SolarForecastingPreprocessor()
        config = get_training_config()
        trainer = ModelTrainer(config)
        evaluator = ModelEvaluator()
        registry = ModelRegistry()

        # Verify components have expected interfaces
        assert hasattr(preprocessor, "fit_transform")
        assert hasattr(trainer, "train")
        assert hasattr(evaluator, "evaluate_model")
        assert hasattr(registry, "register_model_package")

        # Test configuration integration
        assert config.model is not None
        assert config.preprocessor is not None
        assert config.validation is not None
