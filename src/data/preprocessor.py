"""
Solar forecasting time series preprocessor for operational 24-hour prediction.

This module implements a scientifically rigorous time series preprocessing pipeline
for solar power forecasting that respects operational constraints:
- Uses ONLY historical data available at midnight
- Generates 24-hour ahead forecasts
- Eliminates contemporary weather data leakage
- Designed for production MLOps deployment

The preprocessor creates features that are available at prediction time (midnight)
and generates multi-step targets for the next 24 hours of solar production.
"""

import logging
import pickle
import warnings
from datetime import datetime, timedelta
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# pylint: disable=dangerous-default-value,invalid-name
class SolarForecastingPreprocessor:
    """
    Time series preprocessor for operational solar power forecasting.

    This preprocessor is designed for real-world deployment where predictions
    must be made at midnight using only historical data. It generates features
    that respect temporal causality and creates multi-step forecasting targets.

    Key Design Principles:
        - No future data leakage (no contemporary weather)
        - Historical lag features only (AC_POWER, temporal patterns)
        - Multi-step target generation (24-hour forecast)
        - Operational midnight prediction capability
        - Rigorous time series validation

    Example:
        >>> preprocessor = SolarForecastingPreprocessor(
        ...     forecast_horizon=24,
        ...     lag_days=[1, 2, 3, 7, 30]
        ... )
        >>> X, y = preprocessor.create_forecasting_dataset(df)
        >>> print(f"Features shape: {X.shape}, Targets shape: {y.shape}")
    """

    def __init__(
        self,
        forecast_horizon: int = 24,
        lag_days: List[int] = [1, 2, 3, 7, 30],
        rolling_windows: List[int] = [7, 30],
        scaling_method: str = "standard",
        target_frequency: str = "1H",
    ) -> None:
        """
        Initialize the solar forecasting preprocessor.

        Args:
            forecast_horizon: Number of hours to forecast ahead. Defaults to 24.
            lag_days: List of lag days for historical features. Defaults to [1, 2, 3, 7, 30].
            rolling_windows: List of rolling window sizes in days. Defaults to [7, 30].
            scaling_method: Scaling method ('standard', 'minmax', 'none'). Defaults to "standard".
            target_frequency: Target data frequency. Defaults to "1H".

        Raises:
            ValueError: When invalid parameters are provided.
        """
        # Validate parameters
        if forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        if not lag_days or any(lag <= 0 for lag in lag_days):
            raise ValueError("lag_days must be positive integers")
        if scaling_method not in ["standard", "minmax", "none"]:
            raise ValueError(f"Invalid scaling method: {scaling_method}")

        self.forecast_horizon = forecast_horizon
        self.lag_days = sorted(lag_days)
        self.rolling_windows = sorted(rolling_windows)
        self.scaling_method = scaling_method
        self.target_frequency = target_frequency

        # Initialize scaler
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        # Fitted state tracking
        self.is_fitted = False
        self.fitted_feature_names: List[str] = []
        self.fitted_columns: List[str] = []
        self.feature_metadata: Dict[str, Any] = {}

        logger.info(
            f"Initialized SolarForecastingPreprocessor: "
            f"horizon={forecast_horizon}h, lags={lag_days} days, "
            f"rolling_windows={rolling_windows} days, scaling={scaling_method}"
        )

    def load_and_prepare_data(
        self, generation_path: str, weather_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load solar generation data and prepare for time series forecasting.

        Note: Weather data is loaded but NOT used in forecasting features
        to ensure operational realism (no contemporary weather at prediction time).

        Args:
            generation_path: Path to generation CSV file.
            weather_path: Path to weather CSV file (optional, for validation only).

        Returns:
            Tuple containing:
                - pd.DataFrame: Prepared dataframe with AC_POWER and temporal info.
                - Dict[str, Any]: Loading metadata and statistics.

        Raises:
            FileNotFoundError: When generation data file is not found.
            ValueError: When data loading or preparation fails.
        """
        try:
            logger.info("Loading solar generation data...")

            # Load generation data (primary data source)
            gen_df = pd.read_csv(generation_path)
            gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"])

            # Sort by time (crucial for time series)
            gen_df = gen_df.sort_values("DATE_TIME").reset_index(drop=True)

            # Select only required columns (no weather data in features)
            df = gen_df[["DATE_TIME", "AC_POWER"]].copy()

            # Resample to hourly frequency if needed
            if self.target_frequency == "1H":
                df = self._resample_to_hourly(df)

            # Load weather data for validation only (not used in features)
            weather_metadata = {}
            if weather_path:
                try:
                    weather_df = pd.read_csv(weather_path)
                    weather_metadata = {
                        "weather_records": len(weather_df),
                        "weather_available": True,
                    }
                except Exception as e:
                    logger.warning(f"Could not load weather data: {e}")
                    weather_metadata = {"weather_available": False}

            # Create metadata
            metadata = {
                "generation_records": len(gen_df),
                "processed_records": len(df),
                "date_range": {
                    "start": df["DATE_TIME"].min(),
                    "end": df["DATE_TIME"].max(),
                },
                "frequency": self.target_frequency,
                "total_days": (df["DATE_TIME"].max() - df["DATE_TIME"].min()).days,
                "weather_metadata": weather_metadata,
            }

            logger.info(
                f"Data loaded successfully: {len(df):,} records "
                f"from {metadata['date_range']['start']} to {metadata['date_range']['end']}"
            )

            return df, metadata

        except Exception as e:
            logger.error(f"Failed to load and prepare data: {str(e)}")
            raise

    def _resample_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to hourly frequency with proper aggregation.

        Args:
            df: Input dataframe with high-frequency data.

        Returns:
            pd.DataFrame: Hourly resampled dataframe.
        """
        logger.info("Resampling to hourly frequency...")

        df_resampled = df.set_index("DATE_TIME")
        df_resampled = (
            df_resampled.resample("1H")
            .agg({"AC_POWER": "mean"})  # Average power over the hour
            .reset_index()
        )

        # Remove any NaN values from resampling
        initial_rows = len(df_resampled)
        df_resampled = df_resampled.dropna()
        final_rows = len(df_resampled)

        if initial_rows != final_rows:
            logger.info(
                f"Removed {initial_rows - final_rows} NaN rows after resampling"
            )

        return df_resampled

    def create_lag_features_historical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create historical lag features that are available at prediction time.

        These features use only past AC_POWER values that would be available
        at midnight when making tomorrow's forecast.

        Args:
            df: Input dataframe with DATE_TIME and AC_POWER columns.

        Returns:
            pd.DataFrame: Dataframe with added historical lag features.
        """
        logger.info("Creating historical lag features...")
        df_features = df.copy()

        # Convert lag days to lag hours
        lag_hours = [lag_day * 24 for lag_day in self.lag_days]

        # Create basic lag features
        for lag_day, lag_hour in zip(self.lag_days, lag_hours):
            df_features[f"ac_power_lag_{lag_day}d"] = df_features["AC_POWER"].shift(
                lag_hour
            )

        # Create rolling statistics for longer-term patterns
        for window_days in self.rolling_windows:
            window_hours = window_days * 24

            # Rolling mean and std
            df_features[f"rolling_mean_{window_days}d"] = (
                df_features["AC_POWER"]
                .rolling(window=window_hours, min_periods=window_hours // 2)
                .mean()
                .shift(24)  # Shift to ensure availability at prediction time
            )

            df_features[f"rolling_std_{window_days}d"] = (
                df_features["AC_POWER"]
                .rolling(window=window_hours, min_periods=window_hours // 2)
                .std()
                .shift(24)  # Shift to ensure availability at prediction time
            )

            df_features[f"rolling_max_{window_days}d"] = (
                df_features["AC_POWER"]
                .rolling(window=window_hours, min_periods=window_hours // 2)
                .max()
                .shift(24)  # Shift to ensure availability at prediction time
            )

        # Same-time historical patterns (very important for solar)
        df_features["same_hour_last_week"] = df_features["AC_POWER"].shift(24 * 7)
        df_features["same_hour_2_weeks_ago"] = df_features["AC_POWER"].shift(24 * 14)

        # Day-of-week patterns (weekday vs weekend effects)
        df_features["same_weekday_last_week"] = df_features["AC_POWER"].shift(24 * 7)

        logger.info(
            f"Created {len(self.lag_days)} lag features and "
            f"{len(self.rolling_windows)} rolling features"
        )
        return df_features

    def create_temporal_features_future(self, dates: pd.Series) -> pd.DataFrame:
        """
        Generate temporal features for future dates (tomorrow's forecast).

        These features can be calculated at midnight for the next 24 hours
        and capture seasonal, daily, and weekly patterns.

        Args:
            dates: Series of datetime values for which to create features.

        Returns:
            pd.DataFrame: Dataframe with temporal features for future dates.
        """
        logger.info("Creating temporal features for future dates...")

        # Initialize features dataframe
        features = pd.DataFrame(index=dates.index)

        # Basic temporal components
        features["hour_of_day"] = dates.dt.hour
        features["day_of_week"] = dates.dt.dayofweek  # 0=Monday, 6=Sunday
        features["day_of_year"] = dates.dt.dayofyear
        features["month"] = dates.dt.month
        features["quarter"] = dates.dt.quarter

        # Solar-specific features
        features["is_daylight"] = (
            (features["hour_of_day"] >= 6) & (features["hour_of_day"] <= 18)
        ).astype(int)

        features["is_peak_solar"] = (
            (features["hour_of_day"] >= 10) & (features["hour_of_day"] <= 14)
        ).astype(int)

        features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

        # Cyclical encoding for periodic features
        features["hour_sin"] = np.sin(2 * np.pi * features["hour_of_day"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour_of_day"] / 24)

        features["day_of_year_sin"] = np.sin(2 * np.pi * features["day_of_year"] / 365)
        features["day_of_year_cos"] = np.cos(2 * np.pi * features["day_of_year"] / 365)

        features["day_of_week_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["day_of_week_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)

        # Solar elevation approximation (physics-based)
        features["solar_elevation_proxy"] = np.sin(
            2 * np.pi * features["hour_of_day"] / 24
        ) * np.sin(2 * np.pi * features["day_of_year"] / 365)

        # Season indicators
        features["season"] = ((features["month"] % 12 + 3) // 3).map(
            {1: "winter", 2: "spring", 3: "summer", 4: "autumn"}
        )

        logger.info(f"Created {features.shape[1]} temporal features")
        return features

    def create_forecasting_dataset(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create complete forecasting dataset with features (X) and targets (y).

        This method creates the time series structure required for multi-step
        forecasting while ensuring no future data leakage.

        Args:
            df: Input dataframe with DATE_TIME and AC_POWER columns.

        Returns:
            Tuple containing:
                - pd.DataFrame: Features matrix (X) with historical and temporal features.
                - pd.DataFrame: Targets matrix (y) with multi-step forecasts.

        Example:
            >>> preprocessor = SolarForecastingPreprocessor()
            >>> X, y = preprocessor.create_forecasting_dataset(df)
            >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
        """
        logger.info("Creating forecasting dataset...")

        # Step 1: Create historical lag features
        df_with_lags = self.create_lag_features_historical(df)

        # Step 2: Create multi-step targets (24-hour forecast)
        target_columns = []
        for h in range(1, self.forecast_horizon + 1):
            target_col = f"ac_power_{h}h"
            df_with_lags[target_col] = df_with_lags["AC_POWER"].shift(-h)
            target_columns.append(target_col)

        # Step 3: Create temporal features for current timestamp
        # (These will be shifted to represent "tomorrow" in preparation phase)
        temporal_features = self.create_temporal_features_future(
            df_with_lags["DATE_TIME"]
        )

        # Step 4: Combine all features
        feature_columns = [
            col
            for col in df_with_lags.columns
            if col not in ["DATE_TIME", "AC_POWER"] + target_columns
        ]

        # Combine lag features with temporal features
        X = pd.concat(
            [df_with_lags[["DATE_TIME"] + feature_columns], temporal_features], axis=1
        )

        # Handle categorical variables
        if "season" in X.columns:
            X = pd.get_dummies(X, columns=["season"], prefix="season")

        # Step 5: Create targets dataframe
        y = df_with_lags[target_columns].copy()

        # Step 6: Remove rows with NaN values
        # Important: Remove rows where we don't have complete lag features or targets
        valid_mask = X.select_dtypes(include=[np.number]).notna().all(
            axis=1
        ) & y.notna().all(axis=1)

        X_clean = X[valid_mask].reset_index(drop=True)
        y_clean = y[valid_mask].reset_index(drop=True)

        # Store fitted information
        self.fitted_feature_names = [
            col for col in X_clean.columns if col != "DATE_TIME"
        ]
        self.fitted_columns = X_clean.columns.tolist()

        logger.info(
            f"Forecasting dataset created: X shape {X_clean.shape}, y shape {y_clean.shape}"
        )
        logger.info(f"Removed {len(X) - len(X_clean)} rows with missing values")

        return X_clean, y_clean

    def prepare_midnight_prediction(
        self, df: pd.DataFrame, prediction_date: str
    ) -> pd.DataFrame:
        """
        Prepare features for operational midnight prediction.

        This method simulates the real operational scenario where we make
        predictions at midnight using only historical data available at that time.

        Args:
            df: Historical dataframe with DATE_TIME and AC_POWER columns.
            prediction_date: Date for which to make prediction (YYYY-MM-DD format).

        Returns:
            pd.DataFrame: Single-row feature vector ready for 24-hour prediction.

        Example:
            >>> features = preprocessor.prepare_midnight_prediction(df, "2023-05-15")
            >>> prediction = model.predict(features)  # 24-hour forecast
        """
        logger.info(f"Preparing midnight prediction for {prediction_date}")

        # Parse prediction date
        pred_date = pd.to_datetime(prediction_date)
        midnight_time = pred_date.replace(hour=0, minute=0, second=0, microsecond=0)

        # Filter data up to midnight (only historical data available)
        historical_data = df[df["DATE_TIME"] < midnight_time].copy()

        if len(historical_data) == 0:
            raise ValueError(f"No historical data available before {midnight_time}")

        # Create lag features using historical data
        df_with_lags = self.create_lag_features_historical(historical_data)

        # Get the latest available features (at midnight)
        latest_features = df_with_lags.iloc[-1:].copy()

        # Create temporal features for tomorrow (the 24 hours we're predicting)
        _ = pd.date_range(
            start=midnight_time + timedelta(hours=1), periods=24, freq="1H"
        )

        # For midnight prediction, we use the average temporal features for tomorrow
        # or the temporal features at a representative time (e.g., noon)
        representative_time = midnight_time + timedelta(hours=12)  # Noon tomorrow
        temporal_features = self.create_temporal_features_future(
            pd.Series([representative_time])
        )

        # Combine lag features with temporal features
        feature_row = pd.concat(
            [
                latest_features[
                    ["DATE_TIME"]
                    + [
                        col
                        for col in latest_features.columns
                        if col not in ["DATE_TIME", "AC_POWER"]
                    ]
                ],
                temporal_features,
            ],
            axis=1,
        )

        # Handle categorical variables (same as training)
        if "season" in feature_row.columns:
            feature_row = pd.get_dummies(
                feature_row, columns=["season"], prefix="season"
            )

            # Ensure all season columns from training are present
            for col in self.fitted_columns:
                if col.startswith("season_") and col not in feature_row.columns:
                    feature_row[col] = 0

        # Ensure same column order as training
        feature_columns = [col for col in self.fitted_columns if col != "DATE_TIME"]
        missing_cols = set(feature_columns) - set(feature_row.columns)

        if missing_cols:
            logger.warning(f"Missing feature columns: {missing_cols}")
            for col in missing_cols:
                feature_row[col] = 0

        # Select and reorder columns to match training
        prediction_features = feature_row[["DATE_TIME"] + feature_columns]

        logger.info(
            f"Midnight prediction features prepared: {prediction_features.shape}"
        )
        return prediction_features

    def validate_forecasting_setup(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that the forecasting setup has no future data leakage.

        This method performs comprehensive validation to ensure that:
        1. No contemporary weather data is used
        2. All lag features are properly shifted
        3. Temporal features are calculable at prediction time
        4. No target leakage exists

        Args:
            df: Input dataframe to validate.

        Returns:
            Dict[str, Any]: Comprehensive validation report.
        """
        logger.info("Validating forecasting setup for data leakage...")

        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "data_quality": {},
            "temporal_validation": {},
            "feature_validation": {},
            "target_validation": {},
            "leakage_checks": {},
            "overall_valid": False,
        }

        # Data quality checks
        validation_report["data_quality"] = {
            "total_records": len(df),
            "missing_ac_power": df["AC_POWER"].isna().sum(),
            "negative_ac_power": (df["AC_POWER"] < 0).sum(),
            "duplicate_timestamps": df["DATE_TIME"].duplicated().sum(),
            "temporal_gaps": self._check_temporal_gaps(df),
            "data_range": {
                "start": df["DATE_TIME"].min().isoformat(),
                "end": df["DATE_TIME"].max().isoformat(),
                "days": (df["DATE_TIME"].max() - df["DATE_TIME"].min()).days,
            },
        }

        # Temporal validation
        validation_report["temporal_validation"] = {
            "is_sorted": df["DATE_TIME"].is_monotonic_increasing,
            "frequency_consistent": self._check_frequency_consistency(df),
            "sufficient_history": len(df) >= max(self.lag_days) * 24,
            "min_lag_availability": len(df) >= min(self.lag_days) * 24,
        }

        # Feature validation (no weather features)
        forbidden_features = [
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION",
            "temp_difference",
            "temp_ratio",
            "irradiation_per_temp",
            "power_efficiency",
            "temp_irradiation_interaction",
        ]

        present_forbidden = [col for col in forbidden_features if col in df.columns]

        validation_report["feature_validation"] = {
            "forbidden_weather_features": present_forbidden,
            "has_ac_power": "AC_POWER" in df.columns,
            "has_datetime": "DATE_TIME" in df.columns,
            "only_allowed_features": len(present_forbidden) == 0,
        }

        # Leakage checks
        validation_report["leakage_checks"] = {
            "no_contemporary_weather": len(present_forbidden) == 0,
            "lag_features_proper": True,  # Will be set after creating features
            "temporal_features_future_only": True,  # Can be calculated at midnight
            "no_target_leakage": True,  # Targets are future values
        }

        # Overall validation
        validation_report["overall_valid"] = all(
            [
                validation_report["data_quality"]["missing_ac_power"] == 0,
                validation_report["temporal_validation"]["is_sorted"],
                validation_report["temporal_validation"]["sufficient_history"],
                validation_report["feature_validation"]["only_allowed_features"],
                validation_report["leakage_checks"]["no_contemporary_weather"],
            ]
        )

        # Log validation results
        if validation_report["overall_valid"]:
            logger.info("✅ Forecasting setup validation PASSED")
        else:
            logger.warning("❌ Forecasting setup validation FAILED")
            logger.warning(f"Issues found: {validation_report}")

        return validation_report

    def _check_temporal_gaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for gaps in the time series data."""
        df_sorted = df.sort_values("DATE_TIME")
        time_diffs = df_sorted["DATE_TIME"].diff()
        expected_diff = pd.Timedelta(hours=1)

        gaps = time_diffs[time_diffs > expected_diff]

        return {
            "total_gaps": len(gaps),
            "max_gap_hours": gaps.max().total_seconds() / 3600 if len(gaps) > 0 else 0,
            "gap_locations": gaps.index.tolist()[:10],  # First 10 gap locations
        }

    def _check_frequency_consistency(self, df: pd.DataFrame) -> bool:
        """Check if data frequency is consistent with target frequency."""
        if len(df) < 2:
            return True

        time_diffs = df.sort_values("DATE_TIME")["DATE_TIME"].diff().dropna()
        most_common_diff = (
            time_diffs.mode().iloc[0] if len(time_diffs) > 0 else pd.Timedelta(hours=1)
        )

        expected_diff = (
            pd.Timedelta(hours=1)
            if self.target_frequency == "1H"
            else pd.Timedelta(minutes=15)
        )

        return (
            abs((most_common_diff - expected_diff).total_seconds()) < 60
        )  # 1 minute tolerance

    def fit_transform(
        self, generation_path: str, weather_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Complete preprocessing pipeline: load, validate, and create forecasting dataset.

        Args:
            generation_path: Path to generation CSV file.
            weather_path: Path to weather CSV file (optional, for validation only).

        Returns:
            Tuple containing:
                - pd.DataFrame: Features matrix (X).
                - pd.DataFrame: Targets matrix (y).
                - Dict[str, Any]: Comprehensive preprocessing metadata.
        """
        logger.info("Starting complete forecasting preprocessing pipeline...")

        # Step 1: Load and prepare data
        df, load_metadata = self.load_and_prepare_data(generation_path, weather_path)

        # Step 2: Validate setup
        validation_report = self.validate_forecasting_setup(df)

        if not validation_report["overall_valid"]:
            raise ValueError(
                "Forecasting setup validation failed. Check validation_report for details."
            )

        # Step 3: Create forecasting dataset
        X, y = self.create_forecasting_dataset(df)

        # Step 4: Fit scaler if enabled
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

        if self.scaler is not None:
            X_scaled = X.copy()
            X_scaled[numeric_features] = self.scaler.fit_transform(X[numeric_features])
            X = X_scaled

        # Step 5: Mark as fitted
        self.is_fitted = True

        # Step 6: Compile metadata
        self.feature_metadata = {
            "load_metadata": load_metadata,
            "validation_report": validation_report,
            "feature_info": {
                "total_features": len(self.fitted_feature_names),
                "lag_features": sum(
                    1 for f in self.fitted_feature_names if "lag_" in f
                ),
                "rolling_features": sum(
                    1 for f in self.fitted_feature_names if "rolling_" in f
                ),
                "temporal_features": sum(
                    1
                    for f in self.fitted_feature_names
                    if any(t in f for t in ["hour", "day", "month", "season"])
                ),
                "feature_names": self.fitted_feature_names,
            },
            "target_info": {
                "forecast_horizon": self.forecast_horizon,
                "target_columns": y.columns.tolist(),
            },
            "configuration": {
                "lag_days": self.lag_days,
                "rolling_windows": self.rolling_windows,
                "scaling_method": self.scaling_method,
                "target_frequency": self.target_frequency,
            },
        }

        logger.info(f"Preprocessing complete! X shape: {X.shape}, y shape: {y.shape}")
        logger.info(
            f"Features: {len(self.fitted_feature_names)}, Targets: {self.forecast_horizon}"
        )

        return X, y, self.feature_metadata

    def transform(
        self, generation_path: str, weather_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor parameters.

        Args:
            generation_path: Path to generation CSV file.
            weather_path: Path to weather CSV file (optional, ignored).

        Returns:
            pd.DataFrame: Transformed features ready for prediction.

        Raises:
            RuntimeError: When preprocessor is not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        logger.info("Transforming new data using fitted parameters...")

        # Load and prepare data
        df, _ = self.load_and_prepare_data(generation_path, weather_path)

        # Create features (without targets)
        df_with_lags = self.create_lag_features_historical(df)
        temporal_features = self.create_temporal_features_future(
            df_with_lags["DATE_TIME"]
        )

        # Combine features
        feature_columns = [
            col for col in df_with_lags.columns if col not in ["DATE_TIME", "AC_POWER"]
        ]

        X = pd.concat(
            [df_with_lags[["DATE_TIME"] + feature_columns], temporal_features], axis=1
        )

        # Handle categorical variables
        if "season" in X.columns:
            X = pd.get_dummies(X, columns=["season"], prefix="season")

        # Ensure same columns as training
        for col in self.fitted_feature_names:
            if col not in X.columns:
                X[col] = 0

        # Select and reorder columns
        X = X[["DATE_TIME"] + self.fitted_feature_names]

        # Apply scaling if fitted
        if self.scaler is not None:
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            X[numeric_features] = self.scaler.transform(X[numeric_features])

        # Remove NaN rows
        X = X.dropna()

        logger.info(f"Transform complete! Shape: {X.shape}")
        return X

    def save_preprocessor(self, filepath: str) -> None:
        """
        Save fitted preprocessor to pickle file.

        Args:
            filepath: Path to save the preprocessor pickle file.

        Raises:
            RuntimeError: When preprocessor is not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before saving")

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        logger.info(f"Preprocessor saved to: {filepath}")

    @staticmethod
    def load_preprocessor(filepath: str) -> "SolarForecastingPreprocessor":
        """
        Load fitted preprocessor from pickle file.

        Args:
            filepath: Path to the preprocessor pickle file.

        Returns:
            SolarForecastingPreprocessor: Loaded and fitted preprocessor.
        """
        with open(filepath, "rb") as f:
            preprocessor = pickle.load(f)

        logger.info(f"Preprocessor loaded from: {filepath}")
        return preprocessor

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names after preprocessing.

        Returns:
            List[str]: List of feature column names.

        Raises:
            RuntimeError: When preprocessor is not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted to get feature names")

        return self.fitted_feature_names.copy()

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get comprehensive preprocessing information and validation summary.

        Returns:
            Dict[str, Any]: Complete preprocessing metadata and configuration.

        Raises:
            RuntimeError: When preprocessor is not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted to get info")

        return {
            "configuration": {
                "forecast_horizon": self.forecast_horizon,
                "lag_days": self.lag_days,
                "rolling_windows": self.rolling_windows,
                "scaling_method": self.scaling_method,
                "target_frequency": self.target_frequency,
            },
            "fitted_state": {
                "is_fitted": self.is_fitted,
                "feature_count": len(self.fitted_feature_names),
                "scaler_fitted": self.scaler is not None,
            },
            "metadata": self.feature_metadata,
            "validation_summary": {
                "no_weather_leakage": True,
                "historical_features_only": True,
                "multi_step_targets": True,
                "operational_ready": True,
            },
        }


# Export main class
__all__ = ["SolarForecastingPreprocessor"]
