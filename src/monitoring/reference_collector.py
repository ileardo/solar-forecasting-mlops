"""
Reference data collector for solar forecasting model monitoring.

This module collects statistical metadata from training datasets to establish
baseline reference statistics for drift detection and model monitoring.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReferenceDataCollector:
    """
    Collects and computes reference statistics from training datasets.

    This class generates comprehensive statistical metadata from training data
    that will be used as baseline references for monitoring model drift and
    data quality over time.

    Example:
        >>> collector = ReferenceDataCollector()
        >>> reference_stats = collector.collect_from_training_data(X_train, y_train)
        >>> print(f"Collected stats for {len(reference_stats['features'])} features")
    """

    def __init__(self) -> None:
        """Initialize the reference data collector."""
        logger.info("ReferenceDataCollector initialized")

    def _compute_feature_statistics(
        self, feature_data: pd.Series, feature_name: str
    ) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for a single feature, handling different dtypes.

        Args:
            feature_data: Series containing feature values.
            feature_name: Name of the feature.

        Returns:
            Dict[str, Any]: Statistical metadata for the feature.
        """
        # Remove NaN values for calculation
        clean_data = feature_data.dropna()

        if len(clean_data) == 0:
            logger.warning(f"Feature {feature_name} contains only NaN values")
            return self._get_empty_feature_stats(feature_name)

        # Base stats applicable to all types
        stats = {
            "feature_name": feature_name,
            "dtype": str(feature_data.dtype),
            "count": int(len(clean_data)),
            "missing_count": int(len(feature_data) - len(clean_data)),
            "missing_percentage": float(
                (len(feature_data) - len(clean_data)) / len(feature_data) * 100
            ),
            "unique_values": int(clean_data.nunique()),
            "is_constant": bool(clean_data.nunique() <= 1),
            "mode": (
                clean_data.mode().iloc[0] if not clean_data.mode().empty else None
            ),
        }

        # Only compute numerical stats for numeric columns
        if pd.api.types.is_numeric_dtype(feature_data):
            # Convert boolean to int (0/1) to avoid deprecated operations
            if pd.api.types.is_bool_dtype(clean_data):
                clean_data = clean_data.astype(int)

            # Update stats dict with numeric-only calculations
            stats.update(
                {
                    "mean": float(clean_data.mean()),
                    "median": float(clean_data.median()),
                    "std": float(clean_data.std()),
                    "variance": float(clean_data.var()),
                    "min": float(clean_data.min()),
                    "max": float(clean_data.max()),
                    "range": float(clean_data.max() - clean_data.min()),
                    "q1": float(clean_data.quantile(0.25)),
                    "q3": float(clean_data.quantile(0.75)),
                    "iqr": float(clean_data.quantile(0.75) - clean_data.quantile(0.25)),
                    "skewness": float(clean_data.skew()),
                    "kurtosis": float(clean_data.kurtosis()),
                }
            )

            # Outlier calculation using IQR method
            q1, q3 = stats["q1"], stats["q3"]
            iqr = stats["iqr"]
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = clean_data[
                (clean_data < lower_bound) | (clean_data > upper_bound)
            ]
            stats["outlier_count"] = int(len(outliers))
            stats["outlier_percentage"] = (
                float(len(outliers) / len(clean_data) * 100)
                if len(clean_data) > 0
                else 0.0
            )

        else:
            # For non-numeric types, fill numerical fields with None
            stats.update(
                {
                    "mean": None,
                    "median": None,
                    "std": None,
                    "variance": None,
                    "min": None,
                    "max": None,
                    "range": None,
                    "q1": None,
                    "q3": None,
                    "iqr": None,
                    "skewness": None,
                    "kurtosis": None,
                    "outlier_count": 0,
                    "outlier_percentage": 0.0,
                }
            )

        return stats

    def _get_empty_feature_stats(self, feature_name: str) -> Dict[str, Any]:
        """
        Return empty statistics template for features with no valid data.

        Args:
            feature_name: Name of the feature.

        Returns:
            Dict[str, Any]: Empty statistics template.
        """
        return {
            "feature_name": feature_name,
            "count": 0,
            "missing_count": 0,
            "missing_percentage": 100.0,
            "mean": None,
            "median": None,
            "mode": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "range": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "percentile_1": None,
            "percentile_5": None,
            "percentile_95": None,
            "percentile_99": None,
            "skewness": None,
            "kurtosis": None,
            "dtype": "unknown",
            "unique_values": 0,
            "is_constant": True,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
        }

    def _compute_target_statistics(self, y_train: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute statistics for target variables (multi-step forecasting).

        Args:
            y_train: Training targets dataframe.

        Returns:
            Dict[str, Any]: Target statistics metadata.
        """
        logger.info("Computing target statistics for multi-step forecasting")

        # Overall target statistics
        y_flat = y_train.values.flatten()
        y_clean = y_flat[~np.isnan(y_flat)]

        target_stats = {
            "total_samples": int(len(y_train)),
            "forecast_horizons": int(y_train.shape[1]),
            "total_predictions": int(len(y_flat)),
            "valid_predictions": int(len(y_clean)),
            "missing_predictions": int(len(y_flat) - len(y_clean)),
            # Overall statistics
            "overall_mean": float(np.mean(y_clean)),
            "overall_std": float(np.std(y_clean)),
            "overall_min": float(np.min(y_clean)),
            "overall_max": float(np.max(y_clean)),
            "overall_median": float(np.median(y_clean)),
        }

        # Per-horizon statistics
        horizon_stats = []
        for h in range(y_train.shape[1]):
            horizon_data = y_train.iloc[:, h].dropna()
            if len(horizon_data) > 0:
                horizon_stat = {
                    "horizon": h + 1,  # 1-indexed
                    "count": int(len(horizon_data)),
                    "mean": float(horizon_data.mean()),
                    "std": float(horizon_data.std()),
                    "min": float(horizon_data.min()),
                    "max": float(horizon_data.max()),
                    "median": float(horizon_data.median()),
                }
                horizon_stats.append(horizon_stat)

        target_stats["horizon_statistics"] = horizon_stats
        return target_stats

    def _compute_correlation_matrix(self, X_train: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute feature correlation matrix for monitoring feature relationships.

        Args:
            X_train: Training features dataframe.

        Returns:
            Dict[str, Any]: Correlation analysis results.
        """
        logger.info("Computing feature correlation matrix")

        # Select only numeric columns for correlation
        numeric_features = X_train.select_dtypes(include=[np.number])

        if len(numeric_features.columns) < 2:
            logger.warning("Insufficient numeric features for correlation analysis")
            return {
                "correlation_computed": False,
                "reason": "insufficient_numeric_features",
            }

        # Compute correlation matrix
        corr_matrix = numeric_features.corr()

        # Find highly correlated feature pairs (>0.8 or <-0.8)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    high_corr_pairs.append(
                        {
                            "feature_1": corr_matrix.columns[i],
                            "feature_2": corr_matrix.columns[j],
                            "correlation": float(corr_value),
                        }
                    )

        correlation_stats = {
            "correlation_computed": True,
            "numeric_features_count": len(numeric_features.columns),
            "high_correlation_pairs": high_corr_pairs,
            "high_correlation_count": len(high_corr_pairs),
            "average_absolute_correlation": (
                float(corr_matrix.abs().values.mean())
                if not pd.isna(corr_matrix.abs().values.mean())
                else 0.0
            ),
        }

        return correlation_stats

    def collect_from_training_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Collect comprehensive reference statistics from training data.

        Args:
            X_train: Training features dataframe.
            y_train: Training targets dataframe.
            model_metadata: Optional metadata about the model and training process.

        Returns:
            Dict[str, Any]: Complete reference data statistics.

        Example:
            >>> collector = ReferenceDataCollector()
            >>> metadata = {"model_name": "solar-forecasting-prod", "version": "1"}
            >>> reference_stats = collector.collect_from_training_data(X_train, y_train, metadata)
            >>> print(f"Collected stats for {reference_stats['summary']['feature_count']} features")
        """
        logger.info("Starting reference data collection from training dataset")
        logger.info(f"Training data shape: X{X_train.shape}, y{y_train.shape}")

        # Remove DATE_TIME column if present for statistics
        feature_columns = [col for col in X_train.columns if col != "DATE_TIME"]
        X_features = X_train[feature_columns]

        # Collect feature statistics
        logger.info("Computing feature statistics...")
        feature_stats = []
        for feature_name in X_features.columns:
            feature_stat = self._compute_feature_statistics(
                X_features[feature_name], feature_name
            )
            feature_stats.append(feature_stat)

        # Collect target statistics
        logger.info("Computing target statistics...")
        target_stats = self._compute_target_statistics(y_train)

        # Compute feature correlations
        logger.info("Computing correlation matrix...")
        correlation_stats = self._compute_correlation_matrix(X_features)

        # Compile complete reference data
        reference_data = {
            "collection_timestamp": datetime.now().isoformat(),
            "model_metadata": model_metadata or {},
            # Summary information
            "summary": {
                "training_samples": int(len(X_train)),
                "feature_count": len(feature_stats),
                "target_horizons": int(y_train.shape[1]),
                "total_features_analyzed": len(feature_columns),
                "date_column_excluded": "DATE_TIME" in X_train.columns,
            },
            # Detailed statistics
            "feature_statistics": feature_stats,
            "target_statistics": target_stats,
            "correlation_analysis": correlation_stats,
            # Data quality summary
            "data_quality": {
                "features_with_missing": sum(
                    1 for f in feature_stats if f["missing_count"] > 0
                ),
                "features_constant": sum(1 for f in feature_stats if f["is_constant"]),
                "features_with_outliers": sum(
                    1 for f in feature_stats if f["outlier_percentage"] > 5.0
                ),
                "average_missing_percentage": np.mean(
                    [f["missing_percentage"] for f in feature_stats]
                ),
            },
        }

        logger.info("Reference data collection completed successfully")
        logger.info(f"Collected statistics for {len(feature_stats)} features")
        logger.info(
            f"Data quality: {reference_data['data_quality']['features_with_missing']} features with missing values"
        )

        return reference_data

    def get_drift_reference_summary(
        self, reference_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract simplified reference statistics for drift detection.

        Args:
            reference_data: Complete reference data from collect_from_training_data.

        Returns:
            Dict[str, Dict[str, float]]: Simplified statistics per feature for drift detection.

        Example:
            >>> reference_summary = collector.get_drift_reference_summary(reference_data)
            >>> mean_power = reference_summary["ac_power_lag_1d"]["mean"]
        """
        logger.info("Extracting drift reference summary")

        drift_reference = {}

        for feature_stat in reference_data["feature_statistics"]:
            feature_name = feature_stat["feature_name"]

            # Only include numeric features with valid statistics
            if feature_stat["mean"] is not None:
                drift_reference[feature_name] = {
                    "mean": feature_stat["mean"],
                    "std": feature_stat["std"],
                    "min": feature_stat["min"],
                    "max": feature_stat["max"],
                    "q1": feature_stat["q1"],
                    "q3": feature_stat["q3"],
                    "median": feature_stat["median"],
                }

        logger.info(
            f"Drift reference summary created for {len(drift_reference)} features"
        )
        return drift_reference


# Export main class
__all__ = ["ReferenceDataCollector"]
