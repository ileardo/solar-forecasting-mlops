"""
Model evaluator for solar forecasting time series models.

This module provides comprehensive evaluation capabilities for multi-step
forecasting models with focus on numerical metrics and performance analysis.
"""

from src.data.preprocessor import SolarForecastingPreprocessor

import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.multioutput import MultiOutputRegressor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluator for solar forecasting time series models.

    This class provides detailed performance evaluation for multi-step forecasting
    models with separate analysis for overall performance and per-horizon metrics.

    Key Features:
    - Overall performance metrics (RMSE, MAE, R²)
    - Per-horizon analysis for each forecast step
    - Statistical analysis of forecast accuracy
    - Performance comparison capabilities
    - Structured numerical output for MLOps integration

    Example:
        >>> evaluator = ModelEvaluator()
        >>> metrics = evaluator.evaluate_model(model, X_test, y_test)
        >>> print(f"Overall RMSE: {metrics['overall']['rmse']:.2f}")
        >>> print(f"1h ahead RMSE: {metrics['per_horizon'][0]['rmse']:.2f}")
    """

    def __init__(self) -> None:
        """Initialize the model evaluator."""
        logger.info("ModelEvaluator initialized")

    def calculate_overall_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate overall performance metrics across all forecast horizons.

        Args:
            y_true: True values array of shape (samples, horizons).
            y_pred: Predicted values array of shape (samples, horizons).

        Returns:
            Dict[str, float]: Overall performance metrics.

        Example:
            >>> metrics = evaluator.calculate_overall_metrics(y_true, y_pred)
            >>> print(f"RMSE: {metrics['rmse']:.2f}")
        """
        logger.info("Calculating overall performance metrics...")

        # Flatten arrays for overall calculation
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # Calculate core metrics
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)

        # Calculate additional metrics
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100

        # Calculate residual statistics
        residuals = y_true_flat - y_pred_flat
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)

        # Calculate percentage of predictions within certain error bounds
        error_pct = np.abs(residuals) / (y_true_flat + 1e-8) * 100
        within_10pct = np.mean(error_pct <= 10) * 100
        within_20pct = np.mean(error_pct <= 20) * 100
        within_30pct = np.mean(error_pct <= 30) * 100

        overall_metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "residual_mean": float(residual_mean),
            "residual_std": float(residual_std),
            "predictions_within_10pct": float(within_10pct),
            "predictions_within_20pct": float(within_20pct),
            "predictions_within_30pct": float(within_30pct),
            "total_samples": int(len(y_true_flat)),
            "forecast_horizons": int(y_true.shape[1]),
        }

        logger.info(
            f"Overall metrics calculated: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}"
        )

        return overall_metrics

    def calculate_per_horizon_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> List[Dict[str, float]]:
        """
        Calculate performance metrics for each individual forecast horizon.

        Args:
            y_true: True values array of shape (samples, horizons).
            y_pred: Predicted values array of shape (samples, horizons).

        Returns:
            List[Dict[str, float]]: List of metrics for each forecast horizon.

        Example:
            >>> horizon_metrics = evaluator.calculate_per_horizon_metrics(y_true, y_pred)
            >>> for h, metrics in enumerate(horizon_metrics):
            ...     print(f"Hour {h+1}: RMSE={metrics['rmse']:.2f}")
        """
        logger.info("Calculating per-horizon performance metrics...")

        horizon_metrics = []

        for h in range(y_true.shape[1]):
            # Extract data for this horizon
            y_true_h = y_true[:, h]
            y_pred_h = y_pred[:, h]

            # Calculate core metrics for this horizon
            rmse_h = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
            mae_h = mean_absolute_error(y_true_h, y_pred_h)
            r2_h = r2_score(y_true_h, y_pred_h)

            # Calculate additional horizon-specific metrics
            mape_h = np.mean(np.abs((y_true_h - y_pred_h) / (y_true_h + 1e-8))) * 100

            # Residual analysis for this horizon
            residuals_h = y_true_h - y_pred_h
            residual_mean_h = np.mean(residuals_h)
            residual_std_h = np.std(residuals_h)

            # Prediction accuracy bounds
            error_pct_h = np.abs(residuals_h) / (y_true_h + 1e-8) * 100
            within_10pct_h = np.mean(error_pct_h <= 10) * 100
            within_20pct_h = np.mean(error_pct_h <= 20) * 100

            # Value statistics for this horizon
            true_mean_h = np.mean(y_true_h)
            true_std_h = np.std(y_true_h)
            pred_mean_h = np.mean(y_pred_h)
            pred_std_h = np.std(y_pred_h)

            horizon_metrics.append(
                {
                    "horizon": h + 1,  # 1-indexed for interpretability
                    "rmse": float(rmse_h),
                    "mae": float(mae_h),
                    "r2": float(r2_h),
                    "mape": float(mape_h),
                    "residual_mean": float(residual_mean_h),
                    "residual_std": float(residual_std_h),
                    "predictions_within_10pct": float(within_10pct_h),
                    "predictions_within_20pct": float(within_20pct_h),
                    "true_mean": float(true_mean_h),
                    "true_std": float(true_std_h),
                    "pred_mean": float(pred_mean_h),
                    "pred_std": float(pred_std_h),
                    "samples": int(len(y_true_h)),
                }
            )

        logger.info(
            f"Per-horizon metrics calculated for {len(horizon_metrics)} horizons"
        )

        return horizon_metrics

    def calculate_horizon_statistics(
        self, per_horizon_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate statistical summary across all forecast horizons.

        Args:
            per_horizon_metrics: List of per-horizon metrics from calculate_per_horizon_metrics.

        Returns:
            Dict[str, float]: Statistical summary of horizon performance.
        """
        logger.info("Calculating horizon statistics...")

        # Extract metrics across horizons
        rmse_values = [m["rmse"] for m in per_horizon_metrics]
        mae_values = [m["mae"] for m in per_horizon_metrics]
        r2_values = [m["r2"] for m in per_horizon_metrics]
        mape_values = [m["mape"] for m in per_horizon_metrics]

        horizon_stats = {
            # RMSE statistics
            "rmse_mean": float(np.mean(rmse_values)),
            "rmse_std": float(np.std(rmse_values)),
            "rmse_min": float(np.min(rmse_values)),
            "rmse_max": float(np.max(rmse_values)),
            "rmse_median": float(np.median(rmse_values)),
            # MAE statistics
            "mae_mean": float(np.mean(mae_values)),
            "mae_std": float(np.std(mae_values)),
            "mae_min": float(np.min(mae_values)),
            "mae_max": float(np.max(mae_values)),
            "mae_median": float(np.median(mae_values)),
            # R² statistics
            "r2_mean": float(np.mean(r2_values)),
            "r2_std": float(np.std(r2_values)),
            "r2_min": float(np.min(r2_values)),
            "r2_max": float(np.max(r2_values)),
            "r2_median": float(np.median(r2_values)),
            # MAPE statistics
            "mape_mean": float(np.mean(mape_values)),
            "mape_std": float(np.std(mape_values)),
            "mape_min": float(np.min(mape_values)),
            "mape_max": float(np.max(mape_values)),
            "mape_median": float(np.median(mape_values)),
            # Horizon analysis
            "best_horizon_rmse": int(np.argmin(rmse_values) + 1),
            "worst_horizon_rmse": int(np.argmax(rmse_values) + 1),
            "best_horizon_mae": int(np.argmin(mae_values) + 1),
            "worst_horizon_mae": int(np.argmax(mae_values) + 1),
            "best_horizon_r2": int(np.argmax(r2_values) + 1),
            "worst_horizon_r2": int(np.argmin(r2_values) + 1),
        }

        logger.info(
            f"Horizon statistics calculated: RMSE range [{horizon_stats['rmse_min']:.2f}, {horizon_stats['rmse_max']:.2f}]"
        )

        return horizon_stats

    def evaluate_model(
        self,
        model: MultiOutputRegressor,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        preprocessor: Optional[SolarForecastingPreprocessor] = None,
    ) -> Dict[str, Any]:
        """
        Complete model evaluation with overall and per-horizon analysis.

        Args:
            model: Trained multi-output regression model.
            X_test: Test features dataframe.
            y_test: Test targets dataframe.
            preprocessor: Optional preprocessor for additional context.

        Returns:
            Dict[str, Any]: Complete evaluation results with nested structure.

        Example:
            >>> results = evaluator.evaluate_model(model, X_test, y_test)
            >>> overall_rmse = results['overall']['rmse']
            >>> first_hour_rmse = results['per_horizon'][0]['rmse']
            >>> best_horizon = results['horizon_statistics']['best_horizon_rmse']
        """
        logger.info(f"Starting complete model evaluation on {len(X_test)} samples...")

        # Prepare features for prediction
        if "DATE_TIME" in X_test.columns:
            X_features = X_test.drop("DATE_TIME", axis=1)
        else:
            X_features = X_test

        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_features)
        y_true = y_test.values

        # Calculate all metrics
        overall_metrics = self.calculate_overall_metrics(y_true, y_pred)
        per_horizon_metrics = self.calculate_per_horizon_metrics(y_true, y_pred)
        horizon_statistics = self.calculate_horizon_statistics(per_horizon_metrics)

        # Extract key horizons for easy access
        key_horizons = {}
        for hour in [1, 6, 12, 24]:
            if hour <= len(per_horizon_metrics):
                key_horizons[f"{hour}h"] = per_horizon_metrics[hour - 1]

        # Compile complete evaluation results
        evaluation_results = {
            "overall": overall_metrics,
            "per_horizon": per_horizon_metrics,
            "horizon_statistics": horizon_statistics,
            "key_horizons": key_horizons,
            "evaluation_metadata": {
                "test_samples": len(X_test),
                "forecast_horizons": y_true.shape[1],
                "feature_count": X_features.shape[1],
                "model_type": type(model).__name__,
                "has_preprocessor": preprocessor is not None,
            },
        }

        # Add preprocessor info if available
        if preprocessor:
            evaluation_results["evaluation_metadata"][
                "preprocessor_info"
            ] = preprocessor.get_preprocessing_info()

        logger.info("Model evaluation completed successfully")
        logger.info(
            f"Overall performance: RMSE={overall_metrics['rmse']:.2f}, MAE={overall_metrics['mae']:.2f}, R²={overall_metrics['r2']:.3f}"
        )
        logger.info(
            f"Best horizon (RMSE): {horizon_statistics['best_horizon_rmse']} ({per_horizon_metrics[horizon_statistics['best_horizon_rmse']-1]['rmse']:.2f})"
        )
        logger.info(
            f"Worst horizon (RMSE): {horizon_statistics['worst_horizon_rmse']} ({per_horizon_metrics[horizon_statistics['worst_horizon_rmse']-1]['rmse']:.2f})"
        )

        return evaluation_results

    def compare_models(
        self, evaluation_results_list: List[Dict[str, Any]], model_names: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results.

        Args:
            evaluation_results_list: List of evaluation results from evaluate_model.
            model_names: List of model names corresponding to each evaluation result.

        Returns:
            Dict[str, Any]: Comparative analysis of models.

        Example:
            >>> results1 = evaluator.evaluate_model(model1, X_test, y_test)
            >>> results2 = evaluator.evaluate_model(model2, X_test, y_test)
            >>> comparison = evaluator.compare_models([results1, results2], ['XGBoost', 'RandomForest'])
        """
        logger.info(f"Comparing {len(evaluation_results_list)} models...")

        if len(evaluation_results_list) != len(model_names):
            raise ValueError(
                "Number of evaluation results must match number of model names"
            )

        # Extract overall metrics for comparison
        comparison_data = []
        for i, (results, name) in enumerate(zip(evaluation_results_list, model_names)):
            overall_metrics = results["overall"]
            comparison_data.append(
                {
                    "model_name": name,
                    "model_index": i,
                    "rmse": overall_metrics["rmse"],
                    "mae": overall_metrics["mae"],
                    "r2": overall_metrics["r2"],
                    "mape": overall_metrics["mape"],
                    "predictions_within_10pct": overall_metrics[
                        "predictions_within_10pct"
                    ],
                    "predictions_within_20pct": overall_metrics[
                        "predictions_within_20pct"
                    ],
                }
            )

        # Determine best model for each metric
        best_models = {
            "rmse": min(comparison_data, key=lambda x: x["rmse"]),
            "mae": min(comparison_data, key=lambda x: x["mae"]),
            "r2": max(comparison_data, key=lambda x: x["r2"]),
            "mape": min(comparison_data, key=lambda x: x["mape"]),
            "predictions_within_10pct": max(
                comparison_data, key=lambda x: x["predictions_within_10pct"]
            ),
        }

        # Calculate relative improvements
        rmse_values = [d["rmse"] for d in comparison_data]
        mae_values = [d["mae"] for d in comparison_data]

        comparison_results = {
            "model_comparison": comparison_data,
            "best_models": best_models,
            "performance_summary": {
                "rmse_range": [float(np.min(rmse_values)), float(np.max(rmse_values))],
                "mae_range": [float(np.min(mae_values)), float(np.max(mae_values))],
                "rmse_improvement": float(
                    (np.max(rmse_values) - np.min(rmse_values))
                    / np.max(rmse_values)
                    * 100
                ),
                "mae_improvement": float(
                    (np.max(mae_values) - np.min(mae_values)) / np.max(mae_values) * 100
                ),
            },
            "recommendation": best_models["rmse"][
                "model_name"
            ],  # Recommend based on RMSE
        }

        logger.info(
            f"Model comparison completed. Recommended model: {comparison_results['recommendation']}"
        )

        return comparison_results

    def get_evaluation_summary(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of evaluation results.

        Args:
            evaluation_results: Results from evaluate_model method.

        Returns:
            str: Formatted summary string.
        """
        overall = evaluation_results["overall"]
        horizon_stats = evaluation_results["horizon_statistics"]
        metadata = evaluation_results["evaluation_metadata"]

        summary = f"""
Model Evaluation Summary
========================
Test Samples: {metadata['test_samples']:,}
Forecast Horizons: {metadata['forecast_horizons']}
Feature Count: {metadata['feature_count']}

Overall Performance:
- RMSE: {overall['rmse']:.2f} kW
- MAE:  {overall['mae']:.2f} kW
- R²:   {overall['r2']:.3f}
- MAPE: {overall['mape']:.1f}%

Predictions within error bounds:
- ±10%: {overall['predictions_within_10pct']:.1f}%
- ±20%: {overall['predictions_within_20pct']:.1f}%
- ±30%: {overall['predictions_within_30pct']:.1f}%

Horizon Analysis:
- Best performing horizon (RMSE): {horizon_stats['best_horizon_rmse']}h
- Worst performing horizon (RMSE): {horizon_stats['worst_horizon_rmse']}h
- RMSE range: {horizon_stats['rmse_min']:.2f} - {horizon_stats['rmse_max']:.2f} kW
- Mean RMSE: {horizon_stats['rmse_mean']:.2f} ± {horizon_stats['rmse_std']:.2f} kW
"""

        return summary.strip()


# Export main class
__all__ = ["ModelEvaluator"]
