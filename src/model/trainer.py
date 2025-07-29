"""
XGBoost time series model trainer for solar forecasting.

This module implements a complete training pipeline with time series validation,
MLflow integration, and automatic model persistence for production deployment.
"""

import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn

from src.data.preprocessor import SolarForecastingPreprocessor
from src.model.model_config import TrainingConfig, get_training_config
from src.utils.mlflow_utils import (
    setup_mlflow_tracking,
    start_mlflow_run,
    log_model_metrics,
    log_model_params,
    get_or_create_experiment
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Complete XGBoost time series model trainer with MLflow integration.

    This class handles the complete training pipeline including:
    - Data preprocessing with SolarForecastingPreprocessor
    - Time series cross-validation
    - XGBoost multi-output model training
    - MLflow experiment tracking and model persistence
    - Performance evaluation and metrics logging

    Example:
        >>> config = get_training_config()
        >>> trainer = ModelTrainer(config)
        >>> model, metrics = trainer.train()
        >>> print(f"Model trained with RMSE: {metrics['test_rmse_overall']:.2f}")
    """

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        """
        Initialize the model trainer with configuration.

        Args:
            config: Training configuration. If None, uses default configuration.
        """
        self.config = config or get_training_config()

        # Initialize preprocessor
        self.preprocessor = SolarForecastingPreprocessor(
            forecast_horizon=self.config.preprocessor.forecast_horizon,
            lag_days=self.config.preprocessor.lag_days,
            rolling_windows=self.config.preprocessor.rolling_windows,
            scaling_method=self.config.preprocessor.scaling_method,
            target_frequency=self.config.preprocessor.target_frequency
        )

        # Training state
        self.trained_model: Optional[MultiOutputRegressor] = None
        self.training_metrics: Dict[str, float] = {}
        self.is_trained = False

        # Setup MLflow
        setup_mlflow_tracking()

        logger.info(f"ModelTrainer initialized with config: {self.config.experiment_name}")
        logger.info(f"Model params: {self.config.model.to_dict()}")

    def _create_xgboost_model(self) -> MultiOutputRegressor:
        """
        Create XGBoost multi-output regressor with configured parameters.

        Returns:
            MultiOutputRegressor: Configured XGBoost model for 24-hour forecasting.
        """
        # Base XGBoost regressor with config parameters
        xgb_base = xgb.XGBRegressor(**self.config.model.to_dict())

        # Multi-output wrapper for 24-hour forecasting
        model = MultiOutputRegressor(xgb_base, n_jobs=-1)

        logger.info("XGBoost multi-output model created successfully")
        return model

    def _perform_time_series_validation(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Perform time series cross-validation and return aggregated metrics.

        Args:
            X: Features dataframe with DATE_TIME column.
            y: Targets dataframe.

        Returns:
            Dict[str, float]: Aggregated validation metrics across all splits.
        """
        logger.info("Starting time series cross-validation...")

        # Setup time series cross-validation
        tscv = TimeSeriesSplit(
            n_splits=self.config.validation.n_splits,
            test_size=self.config.validation.test_size
        )

        # Remove DATE_TIME for training
        X_features = X.drop('DATE_TIME', axis=1)

        # Storage for fold metrics
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_features)):
            logger.info(f"Processing fold {fold_idx + 1}/{self.config.validation.n_splits}")

            # Split data
            X_train_fold = X_features.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_test_fold = X_features.iloc[test_idx]
            y_test_fold = y.iloc[test_idx]

            # Create and train model for this fold
            fold_model = self._create_xgboost_model()
            fold_model.fit(X_train_fold, y_train_fold)

            # Make predictions
            y_pred_fold = fold_model.predict(X_test_fold)

            # Calculate fold metrics
            fold_rmse = np.sqrt(mean_squared_error(y_test_fold.values, y_pred_fold))
            fold_mae = mean_absolute_error(y_test_fold.values, y_pred_fold)
            fold_r2 = r2_score(y_test_fold.values.flatten(), y_pred_fold.flatten())

            fold_metrics.append({
                'rmse': fold_rmse,
                'mae': fold_mae,
                'r2': fold_r2
            })

            logger.info(f"Fold {fold_idx + 1} - RMSE: {fold_rmse:.2f}, MAE: {fold_mae:.2f}, R²: {fold_r2:.3f}")

        # Aggregate metrics across folds
        aggregated_metrics = {
            'cv_rmse_mean': np.mean([m['rmse'] for m in fold_metrics]),
            'cv_rmse_std': np.std([m['rmse'] for m in fold_metrics]),
            'cv_mae_mean': np.mean([m['mae'] for m in fold_metrics]),
            'cv_mae_std': np.std([m['mae'] for m in fold_metrics]),
            'cv_r2_mean': np.mean([m['r2'] for m in fold_metrics]),
            'cv_r2_std': np.std([m['r2'] for m in fold_metrics])
        }

        logger.info(f"Cross-validation complete - Average RMSE: {aggregated_metrics['cv_rmse_mean']:.2f} ± {aggregated_metrics['cv_rmse_std']:.2f}")

        return aggregated_metrics

    def _calculate_detailed_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive forecasting metrics including per-horizon analysis.

        Args:
            y_true: True values (samples, horizon).
            y_pred: Predicted values (samples, horizon).

        Returns:
            Dict[str, float]: Comprehensive metrics dictionary.
        """
        metrics = {}

        # Overall metrics
        metrics['rmse_overall'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae_overall'] = mean_absolute_error(y_true, y_pred)
        metrics['r2_overall'] = r2_score(y_true.flatten(), y_pred.flatten())

        # Per-horizon metrics for key horizons
        horizon_rmse = []
        horizon_mae = []

        for h in range(y_true.shape[1]):
            h_rmse = np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))
            h_mae = mean_absolute_error(y_true[:, h], y_pred[:, h])
            horizon_rmse.append(h_rmse)
            horizon_mae.append(h_mae)

            # Log specific important horizons
            if h + 1 in [1, 6, 12, 24]:
                metrics[f'rmse_{h+1}h'] = h_rmse
                metrics[f'mae_{h+1}h'] = h_mae

        # Horizon statistics
        metrics['rmse_horizon_mean'] = np.mean(horizon_rmse)
        metrics['rmse_horizon_std'] = np.std(horizon_rmse)
        metrics['mae_horizon_mean'] = np.mean(horizon_mae)
        metrics['mae_horizon_std'] = np.std(horizon_mae)

        return metrics

    def train(self) -> Tuple[MultiOutputRegressor, Dict[str, float], str]:
        """
        Execute complete training pipeline with MLflow logging.

        This method:
        1. Loads and preprocesses data
        2. Performs time series cross-validation
        3. Trains final model on full dataset
        4. Logs everything to MLflow
        5. Returns trained model, metrics, and run_id

        Returns:
            Tuple containing:
                - MultiOutputRegressor: Trained XGBoost model.
                - Dict[str, float]: Complete training and validation metrics.
                - str: MLflow run ID for model registration.

        Raises:
            RuntimeError: When training fails.

        Example:
            >>> trainer = ModelTrainer()
            >>> model, metrics, run_id = trainer.train()
            >>> print(f"Training complete: {metrics['test_rmse_overall']:.2f} RMSE")
        """
        logger.info("Starting complete training pipeline...")

        try:
            # Create or get experiment
            experiment_id = get_or_create_experiment(self.config.experiment_name)

            # Start MLflow run
            run_name = f"{self.config.run_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with start_mlflow_run(
                experiment_name=self.config.experiment_name,
                run_name=run_name
            ):
                start_time = datetime.now()

                # Step 1: Load and preprocess data
                logger.info("Loading and preprocessing data...")
                X, y, metadata = self.preprocessor.fit_transform(
                    self.config.generation_data_path,
                    self.config.weather_data_path
                )

                logger.info(f"Data loaded: X {X.shape}, y {y.shape}")

                # Step 2: Time series cross-validation
                cv_metrics = self._perform_time_series_validation(X, y)

                # Step 3: Train final model on full dataset
                logger.info("Training final model on complete dataset...")

                # Create train/test split (80/20)
                split_point = int(0.8 * len(X))
                X_train = X.iloc[:split_point].reset_index(drop=True)
                y_train = y.iloc[:split_point].reset_index(drop=True)
                X_test = X.iloc[split_point:].reset_index(drop=True)
                y_test = y.iloc[split_point:].reset_index(drop=True)

                # Train final model
                self.trained_model = self._create_xgboost_model()
                X_train_features = X_train.drop('DATE_TIME', axis=1)
                X_test_features = X_test.drop('DATE_TIME', axis=1)

                self.trained_model.fit(X_train_features, y_train)

                # Calculate final metrics
                y_pred_train = self.trained_model.predict(X_train_features)
                y_pred_test = self.trained_model.predict(X_test_features)

                train_metrics = self._calculate_detailed_metrics(y_train.values, y_pred_train)
                test_metrics = self._calculate_detailed_metrics(y_test.values, y_pred_test)

                training_time = (datetime.now() - start_time).total_seconds()

                # Step 4: Log everything to MLflow

                # Log configuration parameters
                log_model_params(self.config.to_dict())

                # Log data info
                log_model_params({
                    'total_samples': len(X),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features_count': len(self.preprocessor.get_feature_names()),
                    'forecast_horizon': self.config.preprocessor.forecast_horizon
                })

                # Log cross-validation metrics
                cv_metrics_prefixed = {f'cv_{k}': v for k, v in cv_metrics.items()}
                log_model_metrics(cv_metrics_prefixed)

                # Log training and test metrics
                train_metrics_prefixed = {f'train_{k}': v for k, v in train_metrics.items()}
                test_metrics_prefixed = {f'test_{k}': v for k, v in test_metrics.items()}

                log_model_metrics(train_metrics_prefixed)
                log_model_metrics(test_metrics_prefixed)
                log_model_metrics({'training_time_seconds': training_time})

                # Step 5: Save model and preprocessor together

                # Save preprocessor to temporary file
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                    self.preprocessor.save_preprocessor(f.name)
                    preprocessor_path = f.name

                # Log preprocessor as artifact
                mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

                # Log model with input example
                sample_input = X_test_features.iloc[:1].values
                mlflow.sklearn.log_model(
                    self.trained_model,
                    "model",
                    input_example=sample_input
                )

                # Compile final metrics
                self.training_metrics = {
                    **cv_metrics,
                    **train_metrics_prefixed,
                    **test_metrics_prefixed,
                    'training_time_seconds': training_time
                }

                self.is_trained = True

                # Get current run ID for model registration
                current_run_id = mlflow.active_run().info.run_id

                logger.info(f"Training completed successfully in {training_time:.1f}s")
                logger.info(f"Final test RMSE: {test_metrics['rmse_overall']:.2f}")
                logger.info(f"Final test MAE: {test_metrics['mae_overall']:.2f}")
                logger.info(f"Final test R²: {test_metrics['r2_overall']:.3f}")
                logger.info(f"MLflow run ID: {current_run_id}")

                return self.trained_model, self.training_metrics, current_run_id

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Model training failed: {str(e)}") from e

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained model.

        Returns:
            Dict[str, float]: Feature names mapped to importance scores.

        Raises:
            RuntimeError: When model is not trained.
        """
        if not self.is_trained or self.trained_model is None:
            raise RuntimeError("Model must be trained before getting feature importance")

        # Get feature importance from first estimator (all estimators should be similar)
        if hasattr(self.trained_model.estimators_[0], 'feature_importances_'):
            importances = self.trained_model.estimators_[0].feature_importances_
            feature_names = self.preprocessor.get_feature_names()

            return dict(zip(feature_names, importances))
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return {}

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the trained model.

        Returns:
            Dict[str, Any]: Model information including configuration and metrics.

        Raises:
            RuntimeError: When model is not trained.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting model info")

        return {
            'model_type': 'XGBoost MultiOutputRegressor',
            'is_trained': self.is_trained,
            'configuration': self.config.to_dict(),
            'training_metrics': self.training_metrics,
            'feature_count': len(self.preprocessor.get_feature_names()),
            'forecast_horizon': self.config.preprocessor.forecast_horizon,
            'preprocessor_info': self.preprocessor.get_preprocessing_info()
        }


# Export main class
__all__ = ["ModelTrainer"]
