"""
Database writer for solar forecasting monitoring data.

This module provides database operations for storing and retrieving
reference data, performance metrics, and drift detection results
for model monitoring purposes.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from src.config.settings import get_settings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonitoringDBWriter:
    """
    Database writer for model monitoring data.

    This class handles database operations for storing reference statistics,
    performance metrics, and drift detection results in PostgreSQL.

    Example:
        >>> db_writer = MonitoringDBWriter()
        >>> db_writer.save_reference_data(reference_stats, "solar-forecasting-prod", "1")
        >>> reference = db_writer.get_reference_data("solar-forecasting-prod", "1")
    """

    def __init__(self) -> None:
        """Initialize the monitoring database writer."""
        self.settings = get_settings()
        logger.info("MonitoringDBWriter initialized")

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

    def save_reference_data(
        self, reference_data: Dict[str, Any], model_name: str, model_version: str
    ) -> int:
        """
        Save reference statistics to database.

        Args:
            reference_data: Complete reference data from ReferenceDataCollector.
            model_name: Name of the model.
            model_version: Version of the model.

        Returns:
            int: ID of the saved reference data record.

        Raises:
            RuntimeError: When save operation fails.

        Example:
            >>> reference_id = db_writer.save_reference_data(
            ...     reference_stats, "solar-forecasting-prod", "1"
            ... )
            >>> print(f"Reference data saved with ID: {reference_id}")
        """
        logger.info(f"Saving reference data for model {model_name} v{model_version}")

        # Extract components from reference data
        summary = reference_data.get("summary", {})
        feature_stats = reference_data.get("feature_statistics", [])
        target_stats = reference_data.get("target_statistics", {})
        correlation_analysis = reference_data.get("correlation_analysis", {})
        data_quality = reference_data.get("data_quality", {})
        model_metadata = reference_data.get("model_metadata", {})

        # Create simplified drift reference for faster queries
        drift_reference = {}
        for feature_stat in feature_stats:
            feature_name = feature_stat["feature_name"]
            if feature_stat.get("mean") is not None:
                drift_reference[feature_name] = {
                    "mean": feature_stat["mean"],
                    "std": feature_stat["std"],
                    "min": feature_stat["min"],
                    "max": feature_stat["max"],
                    "q1": feature_stat["q1"],
                    "q3": feature_stat["q3"],
                    "median": feature_stat["median"],
                }

        insert_sql = """
        INSERT INTO monitoring.reference_data (
            model_name, model_version, training_samples, feature_count, target_horizons,
            collection_timestamp, feature_statistics, target_statistics,
            correlation_analysis, data_quality, model_metadata, drift_reference
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (model_name, model_version)
        DO UPDATE SET
            training_samples = EXCLUDED.training_samples,
            feature_count = EXCLUDED.feature_count,
            target_horizons = EXCLUDED.target_horizons,
            collection_timestamp = EXCLUDED.collection_timestamp,
            feature_statistics = EXCLUDED.feature_statistics,
            target_statistics = EXCLUDED.target_statistics,
            correlation_analysis = EXCLUDED.correlation_analysis,
            data_quality = EXCLUDED.data_quality,
            model_metadata = EXCLUDED.model_metadata,
            drift_reference = EXCLUDED.drift_reference,
            timestamp = CURRENT_TIMESTAMP
        RETURNING id;
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        insert_sql,
                        (
                            model_name,
                            model_version,
                            summary.get("training_samples", 0),
                            summary.get("feature_count", 0),
                            summary.get("target_horizons", 0),
                            datetime.fromisoformat(
                                reference_data["collection_timestamp"]
                            ),
                            json.dumps(feature_stats),
                            json.dumps(target_stats),
                            json.dumps(correlation_analysis),
                            json.dumps(data_quality),
                            json.dumps(model_metadata),
                            json.dumps(drift_reference),
                        ),
                    )

                    reference_id = cursor.fetchone()[0]
                    conn.commit()

            logger.info(f"Reference data saved successfully with ID: {reference_id}")
            return reference_id

        except Exception as e:
            logger.error(f"Failed to save reference data: {str(e)}")
            raise RuntimeError(f"Reference data save failed: {str(e)}") from e

    def get_reference_data(
        self, model_name: str, model_version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve reference data for a model.

        Args:
            model_name: Name of the model.
            model_version: Version of the model. If None, gets latest version.

        Returns:
            Optional[Dict[str, Any]]: Reference data if found, None otherwise.

        Example:
            >>> reference = db_writer.get_reference_data("solar-forecasting-prod")
            >>> if reference:
            ...     print(f"Training samples: {reference['training_samples']}")
        """
        logger.info(f"Retrieving reference data for model {model_name}")

        if model_version:
            query_sql = """
            SELECT * FROM monitoring.reference_data
            WHERE model_name = %s AND model_version = %s
            """
            params = [model_name, model_version]
        else:
            # Get latest version
            query_sql = """
            SELECT * FROM monitoring.reference_data
            WHERE model_name = %s
            ORDER BY timestamp DESC
            LIMIT 1
            """
            params = [model_name]

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query_sql, params)
                    row = cursor.fetchone()

            if row:
                reference_data = dict(row)
                logger.info(
                    f"Reference data retrieved for {model_name} v{reference_data['model_version']}"
                )
                return reference_data
            else:
                logger.info(f"No reference data found for model {model_name}")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve reference data: {str(e)}")
            raise RuntimeError(f"Reference data retrieval failed: {str(e)}") from e

    def get_drift_reference(
        self, model_name: str, model_version: Optional[str] = None
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get simplified drift reference statistics for efficient drift detection.

        Args:
            model_name: Name of the model.
            model_version: Version of the model. If None, gets latest version.

        Returns:
            Optional[Dict[str, Dict[str, float]]]: Drift reference statistics if found.

        Example:
            >>> drift_ref = db_writer.get_drift_reference("solar-forecasting-prod")
            >>> if drift_ref:
            ...     mean_power = drift_ref["ac_power_lag_1d"]["mean"]
        """
        logger.info(f"Retrieving drift reference for model {model_name}")

        if model_version:
            query_sql = """
            SELECT drift_reference FROM monitoring.reference_data
            WHERE model_name = %s AND model_version = %s
            """
            params = [model_name, model_version]
        else:
            query_sql = """
            SELECT drift_reference FROM monitoring.reference_data
            WHERE model_name = %s
            ORDER BY timestamp DESC
            LIMIT 1
            """
            params = [model_name]

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query_sql, params)
                    row = cursor.fetchone()

            if row and row[0]:
                # psycopg2 already decodes JSON, so row[0] is a dict.
                drift_reference = row[0]
                logger.info(
                    f"Drift reference retrieved for {len(drift_reference)} features"
                )
                return drift_reference
            else:
                logger.info(f"No drift reference found for model {model_name}")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve drift reference: {str(e)}")
            raise RuntimeError(f"Drift reference retrieval failed: {str(e)}") from e

    def save_performance_metrics(
        self,
        model_name: str,
        model_version: Optional[str],
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """
        Save performance metrics to database.

        Args:
            model_name: Name of the model.
            model_version: Version of the model.
            metrics: Dictionary of metric names and values.
            metadata: Optional metadata about the metrics.

        Returns:
            List[int]: List of IDs of saved metric records.

        Example:
            >>> metrics = {"rmse": 175.2, "mae": 112.7, "r2": 0.725}
            >>> metric_ids = db_writer.save_performance_metrics(
            ...     "solar-forecasting-prod", "1", metrics
            ... )
        """
        logger.info(f"Saving {len(metrics)} performance metrics for {model_name}")

        insert_sql = """
        INSERT INTO monitoring.model_performance (
            model_name, model_version, metric_name, metric_value, metadata
        ) VALUES (
            %s, %s, %s, %s, %s
        ) RETURNING id;
        """

        try:
            metric_ids = []
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    for metric_name, metric_value in metrics.items():
                        cursor.execute(
                            insert_sql,
                            (
                                model_name,
                                model_version,
                                metric_name,
                                float(metric_value),
                                json.dumps(metadata or {}),
                            ),
                        )
                        metric_id = cursor.fetchone()[0]
                        metric_ids.append(metric_id)

                    conn.commit()

            logger.info(
                f"Performance metrics saved successfully: {len(metric_ids)} records"
            )
            return metric_ids

        except Exception as e:
            logger.error(f"Failed to save performance metrics: {str(e)}")
            raise RuntimeError(f"Performance metrics save failed: {str(e)}") from e

    def save_drift_results(
        self, dataset_name: str, drift_results: Dict[str, Dict[str, Any]]
    ) -> List[int]:
        """
        Save drift detection results to database.

        Args:
            dataset_name: Name of the dataset analyzed.
            drift_results: Dictionary of feature names to drift analysis results.

        Returns:
            List[int]: List of IDs of saved drift records.

        Example:
            >>> drift_results = {
            ...     "feature1": {"drift_score": 0.15, "drift_detected": False},
            ...     "feature2": {"drift_score": 0.85, "drift_detected": True}
            ... }
            >>> drift_ids = db_writer.save_drift_results("batch_predictions", drift_results)
        """
        logger.info(f"Saving drift results for {len(drift_results)} features")

        insert_sql = """
        INSERT INTO monitoring.data_drift (
            dataset_name, feature_name, drift_score, drift_detected
        ) VALUES (
            %s, %s, %s, %s
        ) RETURNING id;
        """

        try:
            drift_ids = []
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    for feature_name, drift_data in drift_results.items():
                        cursor.execute(
                            insert_sql,
                            (
                                dataset_name,
                                feature_name,
                                float(drift_data.get("drift_score", 0.0)),
                                bool(drift_data.get("drift_detected", False)),
                            ),
                        )
                        drift_id = cursor.fetchone()[0]
                        drift_ids.append(drift_id)

                    conn.commit()

            logger.info(f"Drift results saved successfully: {len(drift_ids)} records")
            return drift_ids

        except Exception as e:
            logger.error(f"Failed to save drift results: {str(e)}")
            raise RuntimeError(f"Drift results save failed: {str(e)}") from e

    def list_reference_data(self) -> List[Dict[str, Any]]:
        """
        List all reference data records with summary information.

        Returns:
            List[Dict[str, Any]]: List of reference data summaries.
        """
        logger.info("Listing all reference data records")

        query_sql = """
        SELECT
            id, model_name, model_version, training_samples, feature_count,
            target_horizons, collection_timestamp, timestamp
        FROM monitoring.reference_data
        ORDER BY timestamp DESC
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query_sql)
                    rows = cursor.fetchall()

            reference_list = [dict(row) for row in rows]
            logger.info(f"Retrieved {len(reference_list)} reference data records")
            return reference_list

        except Exception as e:
            logger.error(f"Failed to list reference data: {str(e)}")
            raise RuntimeError(f"Reference data listing failed: {str(e)}") from e

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the monitoring database.

        Returns:
            Dict[str, Any]: Health check results.
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Check table existence and basic stats
                    cursor.execute("SELECT COUNT(*) FROM monitoring.reference_data")
                    reference_count = cursor.fetchone()[0]

                    cursor.execute("SELECT COUNT(*) FROM monitoring.model_performance")
                    performance_count = cursor.fetchone()[0]

                    cursor.execute("SELECT COUNT(*) FROM monitoring.data_drift")
                    drift_count = cursor.fetchone()[0]

            return {
                "healthy": True,
                "reference_data_records": reference_count,
                "performance_records": performance_count,
                "drift_records": drift_count,
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


# Export main class
__all__ = ["MonitoringDBWriter"]
