"""
Simple drift detector for solar forecasting model monitoring.

This module provides basic statistical drift detection by comparing
current data features against reference training statistics.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.monitoring.db_writer import MonitoringDBWriter


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleDriftDetector:
    """
    Basic drift detector using statistical distance metrics.

    Compares current feature statistics against reference data
    to detect data drift in batch predictions.

    Example:
        >>> detector = SimpleDriftDetector("solar-forecasting-prod")
        >>> drift_results = detector.detect_drift(current_features)
        >>> print(f"Drift detected: {drift_results['drift_detected']}")
    """

    def __init__(self, model_name: str, drift_threshold: float = 2.0) -> None:
        """
        Initialize drift detector.

        Args:
            model_name: Name of the model to get reference data for.
            drift_threshold: Threshold for drift detection (standard deviations).
        """
        self.model_name = model_name
        self.drift_threshold = drift_threshold
        self.db_writer = MonitoringDBWriter()
        self.reference_stats: Optional[Dict[str, Dict[str, float]]] = None

        logger.info(f"SimpleDriftDetector initialized for {model_name}")

    def load_reference_data(self, model_version: Optional[str] = None) -> bool:
        """
        Load reference statistics from database.

        Args:
            model_version: Optional model version. Uses latest if None.

        Returns:
            bool: True if reference data loaded successfully.
        """
        logger.info(f"Loading reference data for {self.model_name}")

        try:
            self.reference_stats = self.db_writer.get_drift_reference(
                self.model_name, model_version
            )

            if self.reference_stats:
                logger.info(
                    f"Loaded reference stats for {len(self.reference_stats)} features"
                )
                return True
            else:
                logger.warning(f"No reference data found for {self.model_name}")
                return False

        except Exception as e:
            logger.error(f"Failed to load reference data: {str(e)}")
            return False

    def _calculate_feature_drift_score(
        self,
        current_mean: float,
        current_std: float,
        reference_mean: float,
        reference_std: float,
    ) -> float:
        """
        Calculate drift score using standardized difference.

        Args:
            current_mean: Current feature mean.
            current_std: Current feature std.
            reference_mean: Reference feature mean.
            reference_std: Reference feature std.

        Returns:
            float: Drift score (higher = more drift).
        """
        # Avoid division by zero
        if reference_std == 0 or np.isnan(reference_std):
            reference_std = 1e-6

        # Standardized difference in means
        mean_diff = abs(current_mean - reference_mean) / reference_std

        # Add penalty for std deviation changes
        if current_std > 0 and not np.isnan(current_std):
            std_ratio = max(current_std / reference_std, reference_std / current_std)
            std_penalty = abs(std_ratio - 1.0)
        else:
            std_penalty = 0.0

        drift_score = mean_diff + (0.5 * std_penalty)
        return float(drift_score)

    def detect_drift(self, current_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift in current features against reference data.

        Args:
            current_features: Current feature data to check for drift.

        Returns:
            Dict[str, Any]: Drift detection results.

        Example:
            >>> results = detector.detect_drift(features)
            >>> if results['overall_drift_detected']:
            ...     print(f"Drift in {len(results['drifted_features'])} features")
        """
        logger.info("Starting drift detection...")

        if self.reference_stats is None:
            if not self.load_reference_data():
                return {
                    "drift_detection_failed": True,
                    "reason": "no_reference_data",
                    "overall_drift_detected": False,
                }

        # Remove DATE_TIME if present
        feature_columns = [
            col for col in current_features.columns if col != "DATE_TIME"
        ]
        current_data = current_features[feature_columns]

        # Calculate drift for each feature
        feature_drift_results = {}
        drifted_features = []
        drift_scores = []

        for feature_name in feature_columns:
            if feature_name in self.reference_stats:
                # Calculate current statistics
                feature_data = current_data[feature_name].dropna()

                if len(feature_data) == 0:
                    continue

                current_mean = float(feature_data.mean())
                current_std = float(feature_data.std())

                # Get reference statistics
                ref_stats = self.reference_stats[feature_name]
                ref_mean = ref_stats["mean"]
                ref_std = ref_stats["std"]

                # Calculate drift score
                drift_score = self._calculate_feature_drift_score(
                    current_mean, current_std, ref_mean, ref_std
                )

                # Determine if drift detected
                drift_detected = drift_score > self.drift_threshold

                feature_drift_results[feature_name] = {
                    "drift_score": drift_score,
                    "drift_detected": drift_detected,
                    "current_mean": current_mean,
                    "current_std": current_std,
                    "reference_mean": ref_mean,
                    "reference_std": ref_std,
                }

                if drift_detected:
                    drifted_features.append(feature_name)

                drift_scores.append(drift_score)

        # Overall drift assessment
        overall_drift_detected = len(drifted_features) > 0
        avg_drift_score = np.mean(drift_scores) if drift_scores else 0.0
        max_drift_score = np.max(drift_scores) if drift_scores else 0.0

        drift_results = {
            "overall_drift_detected": overall_drift_detected,
            "features_analyzed": len(feature_drift_results),
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "average_drift_score": float(avg_drift_score),
            "max_drift_score": float(max_drift_score),
            "drift_threshold": self.drift_threshold,
            "feature_drift_results": feature_drift_results,
        }

        logger.info(
            f"Drift detection complete: {len(drifted_features)} features drifted"
        )
        if overall_drift_detected:
            logger.warning(f"Drift detected in features: {drifted_features}")

        return drift_results

    def save_drift_results(
        self, drift_results: Dict[str, Any], dataset_name: str = "batch_predictions"
    ) -> Optional[int]:
        """
        Save drift detection results to database.

        Args:
            drift_results: Results from detect_drift method.
            dataset_name: Name of the dataset analyzed.

        Returns:
            Optional[int]: Number of records saved, None if failed.
        """
        if "feature_drift_results" not in drift_results:
            logger.warning("No feature drift results to save")
            return None

        try:
            # Format for database storage
            db_drift_results = {}
            for feature_name, feature_data in drift_results[
                "feature_drift_results"
            ].items():
                db_drift_results[feature_name] = {
                    "drift_score": feature_data["drift_score"],
                    "drift_detected": feature_data["drift_detected"],
                }

            # Save to database
            saved_ids = self.db_writer.save_drift_results(
                dataset_name, db_drift_results
            )

            logger.info(f"Drift results saved: {len(saved_ids)} records")
            return len(saved_ids)

        except Exception as e:
            logger.error(f"Failed to save drift results: {str(e)}")
            return None

    def quick_drift_check(
        self, current_features: pd.DataFrame, save_results: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Quick drift check with automatic saving.

        Args:
            current_features: Features to check for drift.
            save_results: Whether to save results to database.

        Returns:
            Tuple containing:
                - bool: True if drift detected.
                - Dict[str, Any]: Complete drift results.
        """
        drift_results = self.detect_drift(current_features)

        if save_results and not drift_results.get("drift_detection_failed", False):
            self.save_drift_results(drift_results)

        return drift_results["overall_drift_detected"], drift_results


# Export main class
__all__ = ["SimpleDriftDetector"]
