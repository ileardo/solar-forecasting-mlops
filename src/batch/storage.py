"""
PostgreSQL storage for solar forecasting batch predictions.

This module provides database storage functionality for saving and retrieving
batch prediction results for monitoring and analysis.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from src.config.settings import get_settings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionStorage:
    """
    PostgreSQL storage manager for solar forecasting predictions.

    This class handles database operations for storing and retrieving
    batch prediction results, including forecast data and metadata.

    Example:
        >>> storage = PredictionStorage()
        >>> storage.save_prediction(prediction_results)
        >>> recent = storage.get_recent_predictions(days=7)
    """

    def __init__(self) -> None:
        """Initialize the prediction storage manager."""
        self.settings = get_settings()
        self._ensure_table_exists()

        logger.info("PredictionStorage initialized")

    def _get_connection(self) -> psycopg2.extensions.connection:
        """
        Get database connection.

        Returns:
            psycopg2.extensions.connection: Database connection.

        Raises:
            RuntimeError: When connection fails.
        """
        try:
            conn = psycopg2.connect(
                host=self.settings.db_host,
                port=self.settings.db_port,
                database=self.settings.db_name,
                user=self.settings.db_user,
                password=self.settings.db_password,
            )
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise RuntimeError(f"Database connection failed: {str(e)}") from e

    def _ensure_table_exists(self) -> None:
        """
        Create predictions table if it doesn't exist.

        Raises:
            RuntimeError: When table creation fails.
        """
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            prediction_date DATE NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            forecast_24h JSONB NOT NULL,
            peak_power FLOAT NOT NULL,
            peak_hour INTEGER NOT NULL,
            total_energy FLOAT NOT NULL,
            daylight_energy FLOAT NOT NULL,
            features_count INTEGER NOT NULL,
            prediction_timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Create unique constraint to prevent duplicate predictions
            UNIQUE(prediction_date, model_name, prediction_timestamp)
        );

        -- Create indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_predictions_date
            ON predictions(prediction_date);
        CREATE INDEX IF NOT EXISTS idx_predictions_model
            ON predictions(model_name);
        CREATE INDEX IF NOT EXISTS idx_predictions_created
            ON predictions(created_at);
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_sql)
                    conn.commit()

            logger.info("Predictions table ensured to exist")

        except Exception as e:
            logger.error(f"Failed to create predictions table: {str(e)}")
            raise RuntimeError(f"Table creation failed: {str(e)}") from e

    def save_prediction(self, results: Dict[str, Any]) -> int:
        """
        Save prediction results to database.

        Args:
            results: Prediction results from BatchPredictor.

        Returns:
            int: ID of the saved prediction record.

        Raises:
            RuntimeError: When save operation fails.

        Example:
            >>> results = predictor.run_batch_prediction("2020-06-15")
            >>> prediction_id = storage.save_prediction(results)
            >>> print(f"Saved prediction with ID: {prediction_id}")
        """
        logger.info(f"Saving prediction for {results['prediction_date']}")

        insert_sql = """
        INSERT INTO predictions (
            prediction_date, model_name, forecast_24h, peak_power, peak_hour,
            total_energy, daylight_energy, features_count, prediction_timestamp
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id;
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        insert_sql,
                        (
                            results["prediction_date"],
                            results["model_name"],
                            json.dumps(results["forecast_24h"]),  # Store as JSON
                            results["peak_power"],
                            results["peak_hour"],
                            results["total_energy"],
                            results["daylight_energy"],
                            results["features_count"],
                            results["prediction_timestamp"],
                        ),
                    )

                    prediction_id = cursor.fetchone()[0]
                    conn.commit()

            logger.info(f"Prediction saved successfully with ID: {prediction_id}")
            return prediction_id

        except psycopg2.IntegrityError:
            logger.warning(
                f"Prediction for {results['prediction_date']} already exists"
            )
            raise RuntimeError("Duplicate prediction - already exists in database")
        except Exception as e:
            logger.error(f"Failed to save prediction: {str(e)}")
            raise RuntimeError(f"Save operation failed: {str(e)}") from e

    def get_recent_predictions(
        self, days: int = 7, model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve recent predictions from database.

        Args:
            days: Number of days to look back. Defaults to 7.
            model_name: Optional model name filter.

        Returns:
            pd.DataFrame: Recent predictions with all fields.
        """
        logger.info(f"Retrieving predictions from last {days} days")

        base_sql = """
        SELECT
            id, prediction_date, model_name, forecast_24h, peak_power, peak_hour,
            total_energy, daylight_energy, features_count, prediction_timestamp, created_at
        FROM predictions
        WHERE prediction_date >= %s
        """
        params = [datetime.now().date() - timedelta(days=days)]

        if model_name:
            base_sql += " AND model_name = %s"
            params.append(model_name)

        base_sql += " ORDER BY prediction_date DESC, created_at DESC"

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(base_sql, params)
                    rows = cursor.fetchall()

            if rows:
                df = pd.DataFrame([dict(row) for row in rows])

                logger.info(f"Retrieved {len(df)} predictions")
                return df
            else:
                logger.info("No predictions found for the specified period")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to retrieve predictions: {str(e)}")
            raise RuntimeError(f"Retrieval operation failed: {str(e)}") from e

    def get_prediction_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get summary statistics for recent predictions.

        Args:
            days: Number of days to analyze. Defaults to 30.

        Returns:
            Dict[str, Any]: Summary statistics including averages and trends.

        Example:
            >>> stats = storage.get_prediction_stats(days=30)
            >>> print(f"Average peak power: {stats['avg_peak_power']:.1f} kW")
        """
        logger.info(f"Calculating prediction statistics for last {days} days")

        stats_sql = """
        SELECT
            COUNT(*) as total_predictions,
            COUNT(DISTINCT prediction_date) as unique_dates,
            COUNT(DISTINCT model_name) as unique_models,
            AVG(peak_power) as avg_peak_power,
            AVG(total_energy) as avg_total_energy,
            AVG(daylight_energy) as avg_daylight_energy,
            MIN(peak_power) as min_peak_power,
            MAX(peak_power) as max_peak_power,
            MIN(prediction_date) as earliest_date,
            MAX(prediction_date) as latest_date
        FROM predictions
        WHERE prediction_date >= %s
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(
                        stats_sql, [datetime.now().date() - timedelta(days=days)]
                    )
                    stats = dict(cursor.fetchone())

            logger.info(
                f"Statistics calculated for {stats['total_predictions']} predictions"
            )
            return stats

        except Exception as e:
            logger.error(f"Failed to calculate statistics: {str(e)}")
            raise RuntimeError(f"Statistics calculation failed: {str(e)}") from e

    def delete_old_predictions(self, days: int = 90) -> int:
        """
        Delete predictions older than specified days.

        Args:
            days: Keep predictions newer than this many days. Defaults to 90.

        Returns:
            int: Number of predictions deleted.

        Example:
            >>> deleted_count = storage.delete_old_predictions(days=180)
            >>> print(f"Deleted {deleted_count} old predictions")
        """
        logger.info(f"Deleting predictions older than {days} days")

        delete_sql = """
        DELETE FROM predictions
        WHERE prediction_date < %s
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        delete_sql, [datetime.now().date() - timedelta(days=days)]
                    )
                    deleted_count = cursor.rowcount
                    conn.commit()

            logger.info(f"Deleted {deleted_count} old predictions")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete old predictions: {str(e)}")
            raise RuntimeError(f"Delete operation failed: {str(e)}") from e

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the storage system.

        Returns:
            Dict[str, Any]: Health check results including connection status.

        Example:
            >>> health = storage.health_check()
            >>> print(f"Storage healthy: {health['healthy']}")
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM predictions")
                    total_predictions = cursor.fetchone()[0]

            return {
                "healthy": True,
                "total_predictions": total_predictions,
                "last_check": datetime.now().isoformat(),
                "database_host": self.settings.db_host,
                "database_name": self.settings.db_name,
            }

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "healthy": False,
                "error": str(e),
                "last_check": datetime.now().isoformat(),
            }


def main() -> None:
    """
    CLI entry point for testing storage functionality.

    Example usage:
        python -m src.batch.storage
    """
    storage = PredictionStorage()

    print("Running storage health check...")
    health = storage.health_check()

    if health["healthy"]:
        print(f"Storage healthy - {health['total_predictions']} predictions stored")

        print("\nRecent predictions:")
        recent = storage.get_recent_predictions(days=30)
        print(f"Found {len(recent)} predictions in last 30 days")

        if len(recent) > 0:
            print(
                f"Latest: {recent.iloc[0]['prediction_date']} - {recent.iloc[0]['peak_power']:.1f} kW"
            )

        print("\nStatistics:")
        stats = storage.get_prediction_stats(days=30)
        print(
            f"Average peak power: {stats['avg_peak_power']:.1f} kW"
            if stats["avg_peak_power"]
            else "No data"
        )

    else:
        print(f"Storage unhealthy: {health['error']}")


if __name__ == "__main__":
    main()


# Export main class
__all__ = ["PredictionStorage"]
