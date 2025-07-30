"""
Prefect orchestrator for solar forecasting batch prediction workflows.

This module provides Prefect-based orchestration for automated solar
forecasting batch predictions with task scheduling and monitoring.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from prefect import flow, task
from prefect.logging import get_run_logger

from src.batch.predictor import BatchPredictor
from src.batch.storage import PredictionStorage


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task(retries=2, retry_delay_seconds=30)
def run_prediction_task(
    prediction_date: str, model_name: str = "solar-forecasting-prod"
) -> Dict[str, Any]:
    """
    Prefect task to run batch prediction for a specific date.

    Args:
        prediction_date: Date to predict (YYYY-MM-DD format).
        model_name: Name of the model to use for prediction.

    Returns:
        Dict[str, Any]: Prediction results with forecast and metadata.

    Raises:
        RuntimeError: When prediction fails after retries.
    """
    task_logger = get_run_logger()
    task_logger.info(f"Starting prediction task for {prediction_date}")

    try:
        # Initialize and run batch predictor
        predictor = BatchPredictor(model_name)
        results = predictor.run_batch_prediction(prediction_date)

        task_logger.info(f"Prediction successful: {results['peak_power']:.1f} kW peak")
        return results

    except Exception as e:
        task_logger.error(f"Prediction task failed: {str(e)}")
        raise


@task(retries=1)
def validate_results_task(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate prediction results for data quality.

    Args:
        results: Prediction results to validate.

    Returns:
        Dict[str, Any]: Validated results with quality metrics.

    Raises:
        ValueError: When validation fails.
    """
    task_logger = get_run_logger()
    task_logger.info("Validating prediction results")

    try:
        # Basic validation checks
        forecast = results["forecast_24h"]

        # Check forecast length
        if len(forecast) != 24:
            raise ValueError(f"Invalid forecast length: {len(forecast)}, expected 24")

        # Check for negative values
        negative_count = sum(1 for x in forecast if x < 0)
        if negative_count > 0:
            task_logger.warning(f"Found {negative_count} negative predictions")

        # Check for reasonable power values (0-1000 kW)
        max_power = max(forecast)
        if max_power > 1000:
            task_logger.warning(f"Unusually high power prediction: {max_power:.1f} kW")

        # Add validation metadata
        validation_info = {
            "validation_timestamp": datetime.now().isoformat(),
            "negative_predictions": negative_count,
            "max_power": max_power,
            "validation_passed": True,
        }

        results["validation"] = validation_info
        task_logger.info("Results validation passed")

        return results

    except Exception as e:
        task_logger.error(f"Results validation failed: {str(e)}")
        raise


@task(retries=1)
def save_predictions_task(results: Dict[str, Any]) -> int:
    """
    Save prediction results to PostgreSQL database.

    Args:
        results: Validated prediction results to save.

    Returns:
        int: ID of the saved prediction record.

    Raises:
        RuntimeError: When save operation fails.
    """
    task_logger = get_run_logger()
    task_logger.info(f"Saving prediction for {results['prediction_date']} to database")

    try:
        storage = PredictionStorage()
        prediction_id = storage.save_prediction(results)

        task_logger.info(f"Prediction saved successfully with ID: {prediction_id}")
        return prediction_id

    except Exception as e:
        task_logger.error(f"Failed to save prediction: {str(e)}")
        raise


@task
def log_results_task(results: Dict[str, Any]) -> None:
    """
    Log prediction results summary.

    Args:
        results: Prediction results to log.
    """
    task_logger = get_run_logger()

    # Log summary information
    task_logger.info("=" * 60)
    task_logger.info(f"SOLAR FORECAST SUMMARY - {results['prediction_date']}")
    task_logger.info("=" * 60)
    task_logger.info(
        f"Peak Power: {results['peak_power']:.1f} kW (Hour {results['peak_hour']})"
    )
    task_logger.info(f"Total Energy: {results['total_energy']:.1f} kWh")
    task_logger.info(f"Daylight Energy: {results['daylight_energy']:.1f} kWh")
    task_logger.info(f"Model: {results['model_name']}")
    task_logger.info(f"Features: {results['features_count']}")

    if "validation" in results:
        validation = results["validation"]
        task_logger.info(
            f"Validation: {'PASSED' if validation['validation_passed'] else 'FAILED'}"
        )
        if validation["negative_predictions"] > 0:
            task_logger.info(
                f"Negative predictions: {validation['negative_predictions']}"
            )

    task_logger.info("=" * 60)


@flow(
    name="solar-batch-prediction",
    description="Solar forecasting batch prediction workflow",
    version="1.0",
)
def solar_batch_prediction_flow(
    prediction_date: Optional[str] = None, model_name: str = "solar-forecasting-prod"
) -> Dict[str, Any]:
    """
    Main Prefect flow for solar forecasting batch predictions.

    This flow orchestrates the complete batch prediction workflow including
    model loading, prediction generation, validation, and logging.

    Args:
        prediction_date: Date to predict (YYYY-MM-DD). Uses tomorrow if None.
        model_name: Name of the registered model to use.

    Returns:
        Dict[str, Any]: Complete prediction results with metadata.

    Example:
        >>> # Run for specific date
        >>> results = solar_batch_prediction_flow("2020-06-15")
        >>>
        >>> # Run for tomorrow (default)
        >>> results = solar_batch_prediction_flow()
    """
    flow_logger = get_run_logger()

    # Set default prediction date to tomorrow if not provided
    if prediction_date is None:
        tomorrow = datetime.now() + timedelta(days=1)
        prediction_date = tomorrow.strftime("%Y-%m-%d")

    flow_logger.info(f"Starting solar batch prediction flow for {prediction_date}")
    flow_logger.info(f"Using model: {model_name}")

    try:
        # Task 1: Run batch prediction
        prediction_results = run_prediction_task(prediction_date, model_name)

        # Task 2: Validate results
        validated_results = validate_results_task(prediction_results)

        # Task 3: Save to database
        prediction_id = save_predictions_task(validated_results)

        # Task 4: Log summary
        log_results_task(validated_results)

        # Add database info to results
        validated_results["database_id"] = prediction_id

        flow_logger.info(f"Solar batch prediction flow completed successfully")
        flow_logger.info(f"Prediction saved to database with ID: {prediction_id}")
        return validated_results

    except Exception as e:
        flow_logger.error(f"Solar batch prediction flow failed: {str(e)}")
        raise


@flow(
    name="solar-batch-prediction-schedule",
    description="Scheduled solar forecasting for next day",
)
def daily_solar_prediction_flow() -> Dict[str, Any]:
    """
    Daily scheduled flow for next-day solar predictions.

    This flow runs the batch prediction for tomorrow's solar forecast,
    typically scheduled to run at midnight.

    Returns:
        Dict[str, Any]: Prediction results for tomorrow.
    """
    flow_logger = get_run_logger()

    # Always predict tomorrow
    tomorrow = datetime.now() + timedelta(days=1)
    prediction_date = tomorrow.strftime("%Y-%m-%d")

    flow_logger.info(f"Running daily solar prediction for {prediction_date}")

    # Run the main prediction flow
    return solar_batch_prediction_flow(prediction_date)


def main() -> None:
    """
    CLI entry point for running batch prediction flows.

    Example usage:
        python -m src.batch.orchestrator
        python -m src.batch.orchestrator 2020-06-15
    """
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        prediction_date = sys.argv[1]
        print(f"Running batch prediction for: {prediction_date}")
        results = solar_batch_prediction_flow(prediction_date)
    else:
        print("Running daily solar prediction (tomorrow)")
        results = daily_solar_prediction_flow()

    print(f"\nFlow completed successfully!")
    print(f"Peak power: {results['peak_power']:.1f} kW")
    print(f"Total energy: {results['total_energy']:.1f} kWh")


if __name__ == "__main__":
    main()


# Export main flows
__all__ = [
    "solar_batch_prediction_flow",
    "daily_solar_prediction_flow",
    "run_prediction_task",
    "validate_results_task",
    "save_predictions_task",
    "log_results_task",
]
