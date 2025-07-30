"""
Batch prediction service for solar forecasting models.

This module provides a streamlined batch prediction service that loads
production models from MLflow registry and generates 24-hour solar forecasts
for operational deployment.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from src.data.preprocessor import SolarForecastingPreprocessor
from src.model.registry import ModelRegistry


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchPredictor:
    """
    Streamlined batch prediction service for solar forecasting.

    This class handles the complete batch prediction workflow including
    production model loading, feature preparation, and 24-hour forecasting
    for operational solar power prediction.

    Example:
        >>> predictor = BatchPredictor()
        >>> results = predictor.run_batch_prediction("2020-06-15")
        >>> print(f"Peak power: {results['peak_power']:.1f} kW")
    """

    def __init__(self, model_name: str = "solar-forecasting-prod") -> None:
        """
        Initialize the batch predictor.

        Args:
            model_name: Name of the registered model to use for predictions.
        """
        self.model_name = model_name
        self.model: Optional[MultiOutputRegressor] = None
        self.preprocessor: Optional[SolarForecastingPreprocessor] = None
        self.registry = ModelRegistry()
        self.is_loaded = False

        logger.info(f"BatchPredictor initialized for model: {model_name}")

    def load_production_model(self) -> None:
        """
        Load the production model and preprocessor from MLflow registry.

        Raises:
            RuntimeError: When model loading fails.
        """
        logger.info(f"Loading production model: {self.model_name}")

        try:
            self.model, self.preprocessor = self.registry.load_production_model(
                self.model_name
            )
            self.is_loaded = True

            logger.info("Production model loaded successfully")
            logger.info(f"Model type: {type(self.model).__name__}")
            logger.info(f"Features: {len(self.preprocessor.get_feature_names())}")
            logger.info(f"Forecast horizon: {self.preprocessor.forecast_horizon}h")

        except Exception as e:
            logger.error(f"Failed to load production model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e

    def predict_for_date(self, prediction_date: str, data_path: str) -> Dict[str, Any]:
        """
        Generate 24-hour forecast for a specific date.

        Args:
            prediction_date: Date to predict (YYYY-MM-DD format).
            data_path: Path to historical generation data CSV.

        Returns:
            Dict[str, Any]: Prediction results with forecast and metadata.

        Raises:
            RuntimeError: When prediction fails or model not loaded.

        Example:
            >>> results = predictor.predict_for_date("2020-06-15", "data/raw/Plant_1_Generation_Data.csv")
            >>> print(f"Total energy: {results['total_energy']:.1f} kWh")
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_production_model() first.")

        logger.info(f"Generating 24h forecast for date: {prediction_date}")

        try:
            # Load historical data
            data, _ = self.preprocessor.load_and_prepare_data(data_path)

            # Prepare midnight prediction features
            features = self.preprocessor.prepare_midnight_prediction(
                data, prediction_date
            )

            # Generate 24-hour forecast
            forecast_features = features.drop("DATE_TIME", axis=1)
            forecast_24h = self.model.predict(forecast_features)

            # Extract forecast array (handle multiple rows if present)
            forecast_array = (
                forecast_24h[0] if forecast_24h.shape[0] > 0 else forecast_24h.flatten()
            )

            # Calculate key metrics
            peak_power = float(forecast_array.max())
            peak_hour = int(np.argmax(forecast_array) + 1)
            total_energy = float(forecast_array.sum())
            daylight_energy = float(forecast_array[6:19].sum())  # Hours 7-19 (6am-6pm)

            # Prepare results
            results = {
                "prediction_date": prediction_date,
                "forecast_24h": forecast_array.tolist(),
                "peak_power": peak_power,
                "peak_hour": peak_hour,
                "total_energy": total_energy,
                "daylight_energy": daylight_energy,
                "features_count": len(self.preprocessor.get_feature_names()),
                "model_name": self.model_name,
                "prediction_timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Prediction successful for {prediction_date}")
            logger.info(f"Peak power: {peak_power:.1f} kW at hour {peak_hour}")
            logger.info(f"Total energy: {total_energy:.1f} kWh")

            return results

        except Exception as e:
            logger.error(f"Prediction failed for {prediction_date}: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}") from e

    def run_batch_prediction(
        self, prediction_date: str, data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete batch prediction workflow: load model and generate forecast.

        Args:
            prediction_date: Date to predict (YYYY-MM-DD format).
            data_path: Optional path to data. Uses default if None.

        Returns:
            Dict[str, Any]: Complete prediction results.

        Example:
            >>> predictor = BatchPredictor()
            >>> results = predictor.run_batch_prediction("2020-06-15")
            >>> print(f"Forecast generated: {len(results['forecast_24h'])} hours")
        """
        logger.info(f"Starting batch prediction workflow for {prediction_date}")

        # Set default data path if not provided
        if data_path is None:
            project_root = Path(__file__).parent.parent.parent
            data_path = str(
                project_root / "data" / "raw" / "Plant_1_Generation_Data.csv"
            )

        # Load model if not already loaded
        if not self.is_loaded:
            self.load_production_model()

        # Generate prediction
        results = self.predict_for_date(prediction_date, data_path)

        logger.info("Batch prediction workflow completed successfully")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dict[str, Any]: Model information and status.

        Raises:
            RuntimeError: When model is not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_production_model() first.")

        return {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "is_loaded": self.is_loaded,
            "features_count": len(self.preprocessor.get_feature_names()),
            "forecast_horizon": self.preprocessor.forecast_horizon,
            "scaling_method": self.preprocessor.scaling_method,
        }


# Export main class
__all__ = ["BatchPredictor"]
