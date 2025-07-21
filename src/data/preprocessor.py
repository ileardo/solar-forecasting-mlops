"""
Solar forecasting data preprocessor with comprehensive feature engineering.

This module implements scientific feature engineering for solar power prediction,
including temporal features, weather derivatives, and lag patterns.
"""

import logging
import warnings
from typing import (
    Any,
    Dict,
    List,
    Tuple
)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SolarDataPreprocessor:
    """
    Advanced preprocessor for solar forecasting data with feature engineering.

    This class handles the complete preprocessing pipeline including:
    - Data loading and merging
    - Temporal feature extraction
    - Weather-derived features
    - Lag feature creation
    - Data quality validation
    - Frequency conversion (15-min to hourly)
    """

    def __init__(
        self,
        target_frequency: str = "1H",
        lag_hours: List[int] = [24, 48, 72],
        scaling_method: str = "standard",
    ) -> None:
        """
        Initialize the solar data preprocessor.

        Args:
            target_frequency: Target resampling frequency. Defaults to "1H".
            lag_hours: List of lag hours for feature creation. Defaults to [24, 48, 72].
            scaling_method: Scaling method ('standard', 'minmax', 'none'). Defaults to "standard".

        Raises:
            ValueError: When invalid scaling method is provided.
        """
        self.target_frequency = target_frequency
        self.lag_hours = lag_hours
        self.scaling_method = scaling_method

        # Initialize scalers
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling_method == "none":
            self.scaler = None
        else:
            raise ValueError(f"Invalid scaling method: {scaling_method}")

        # Feature metadata
        self.feature_metadata: Dict[str, Any] = {}
        self.is_fitted = False

        logger.info(
            f"Initialized SolarDataPreprocessor with frequency={target_frequency}"
        )

    def load_and_merge_data(
        self, generation_path: str, weather_path: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load and merge generation and weather data.

        Args:
            generation_path: Path to generation CSV file.
            weather_path: Path to weather CSV file.

        Returns:
            Tuple containing:
                - pd.DataFrame: Merged dataframe with both generation and weather data.
                - Dict[str, Any]: Loading metadata and statistics.

        Raises:
            FileNotFoundError: When data files are not found.
            ValueError: When data merge fails.
        """
        try:
            # Load datasets
            logger.info("Loading generation and weather data...")
            gen_df = pd.read_csv(generation_path)
            weather_df = pd.read_csv(weather_path)

            # Convert datetime
            gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"])
            weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"])

            # Merge on timestamp
            merged_df = pd.merge(
                gen_df[["DATE_TIME", "AC_POWER", "DC_POWER", "DAILY_YIELD"]],
                weather_df[
                    [
                        "DATE_TIME",
                        "AMBIENT_TEMPERATURE",
                        "MODULE_TEMPERATURE",
                        "IRRADIATION",
                    ]
                ],
                on="DATE_TIME",
                how="inner",
            )

            # Create metadata
            metadata = {
                "generation_records": len(gen_df),
                "weather_records": len(weather_df),
                "merged_records": len(merged_df),
                "date_range": {
                    "start": merged_df["DATE_TIME"].min(),
                    "end": merged_df["DATE_TIME"].max(),
                },
                "merge_success_rate": len(merged_df)
                / max(len(gen_df), len(weather_df)),
            }

            logger.info(f"Successfully merged data: {len(merged_df):,} records")
            return merged_df, metadata

        except Exception as e:
            logger.error(f"Failed to load and merge data: {str(e)}")
            raise

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive temporal features for solar forecasting.

        Args:
            df: Input dataframe with DATE_TIME column.

        Returns:
            pd.DataFrame: Dataframe with added temporal features.
        """
        logger.info("Creating temporal features...")
        df_features = df.copy()

        # Basic time components
        df_features["hour"] = df_features["DATE_TIME"].dt.hour
        df_features["day_of_year"] = df_features["DATE_TIME"].dt.dayofyear
        df_features["month"] = df_features["DATE_TIME"].dt.month
        df_features["weekday"] = df_features["DATE_TIME"].dt.weekday

        # Solar-specific temporal features
        df_features["is_daylight"] = (
            (df_features["hour"] >= 6) & (df_features["hour"] <= 18)
        ).astype(int)
        df_features["solar_elevation_proxy"] = np.sin(
            2 * np.pi * df_features["hour"] / 24
        )

        # Cyclical encoding for periodic features
        df_features["hour_sin"] = np.sin(2 * np.pi * df_features["hour"] / 24)
        df_features["hour_cos"] = np.cos(2 * np.pi * df_features["hour"] / 24)
        df_features["day_sin"] = np.sin(2 * np.pi * df_features["day_of_year"] / 365)
        df_features["day_cos"] = np.cos(2 * np.pi * df_features["day_of_year"] / 365)

        # Season indicator
        df_features["season"] = ((df_features["month"] % 12 + 3) // 3).map(
            {1: "winter", 2: "spring", 3: "summer", 4: "autumn"}
        )

        logger.info("Temporal features created successfully")
        return df_features

    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-derived features for enhanced prediction accuracy.

        Args:
            df: Input dataframe with weather columns.

        Returns:
            pd.DataFrame: Dataframe with added weather-derived features.
        """
        logger.info("Creating weather-derived features...")
        df_features = df.copy()

        # Temperature differentials
        df_features["temp_difference"] = (
            df_features["MODULE_TEMPERATURE"] - df_features["AMBIENT_TEMPERATURE"]
        )
        df_features["temp_ratio"] = df_features["MODULE_TEMPERATURE"] / (
            df_features["AMBIENT_TEMPERATURE"] + 1e-8
        )

        # Irradiation efficiency features
        df_features["irradiation_per_temp"] = df_features["IRRADIATION"] / (
            df_features["MODULE_TEMPERATURE"] + 1e-8
        )
        df_features["power_efficiency"] = df_features["AC_POWER"] / (
            df_features["IRRADIATION"] + 1e-8
        )

        # Weather interaction features
        df_features["temp_irradiation_interaction"] = (
            df_features["AMBIENT_TEMPERATURE"] * df_features["IRRADIATION"]
        )

        # Binned weather features for non-linear relationships
        df_features["temp_category"] = pd.cut(
            df_features["AMBIENT_TEMPERATURE"],
            bins=[0, 22, 26, 30, 50],
            labels=["cool", "optimal", "warm", "hot"],
        )
        df_features["irradiation_category"] = pd.cut(
            df_features["IRRADIATION"],
            bins=[0, 0.2, 0.6, 1.0, 2.0],
            labels=["low", "medium", "high", "peak"],
        )

        logger.info("Weather features created successfully")
        return df_features

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag features for capturing temporal dependencies.

        Args:
            df: Input dataframe sorted by DATE_TIME.

        Returns:
            pd.DataFrame: Dataframe with added lag features.
        """
        logger.info("Creating lag features...")
        df_features = df.copy()
        df_features = df_features.sort_values("DATE_TIME")

        # Create lag features for AC_POWER
        for lag_hour in self.lag_hours:
            # Calculate lag periods based on frequency
            if self.target_frequency == "1H":
                lag_periods = lag_hour
            else:  # 15-min frequency
                lag_periods = lag_hour * 4

            df_features[f"ac_power_lag_{lag_hour}h"] = df_features["AC_POWER"].shift(
                lag_periods
            )
            df_features[f"irradiation_lag_{lag_hour}h"] = df_features[
                "IRRADIATION"
            ].shift(lag_periods)

        # Rolling statistics
        for window in [6, 12, 24]:  # 6h, 12h, 24h windows
            window_periods = window if self.target_frequency == "1H" else window * 4

            df_features[f"ac_power_rolling_mean_{window}h"] = (
                df_features["AC_POWER"]
                .rolling(window=window_periods, min_periods=1)
                .mean()
            )
            df_features[f"ac_power_rolling_std_{window}h"] = (
                df_features["AC_POWER"]
                .rolling(window=window_periods, min_periods=1)
                .std()
            )

        # Previous day same hour (very important for solar)
        daily_lag = 24 if self.target_frequency == "1H" else 96
        df_features["ac_power_same_hour_yesterday"] = df_features["AC_POWER"].shift(
            daily_lag
        )

        logger.info("Lag features created successfully")
        return df_features

    def resample_to_target_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to target frequency with appropriate aggregation.

        Args:
            df: Input dataframe with high-frequency data.

        Returns:
            pd.DataFrame: Resampled dataframe at target frequency.
        """
        if self.target_frequency == "15T":  # Already at 15-min frequency
            return df

        logger.info(f"Resampling to {self.target_frequency} frequency...")
        df_resampled = df.set_index("DATE_TIME")

        # Define aggregation rules
        agg_rules = {
            "AC_POWER": "mean",
            "DC_POWER": "mean",
            "DAILY_YIELD": "last",  # Cumulative value
            "AMBIENT_TEMPERATURE": "mean",
            "MODULE_TEMPERATURE": "mean",
            "IRRADIATION": "mean",
            "hour": "first",
            "day_of_year": "first",
            "month": "first",
            "weekday": "first",
            "is_daylight": "max",
            "solar_elevation_proxy": "mean",
            "hour_sin": "first",
            "hour_cos": "first",
            "day_sin": "first",
            "day_cos": "first",
            "season": "first",
            "temp_difference": "mean",
            "temp_ratio": "mean",
            "irradiation_per_temp": "mean",
            "power_efficiency": "mean",
            "temp_irradiation_interaction": "mean",
            "temp_category": "first",
            "irradiation_category": "first",
        }

        # Add lag features to aggregation rules
        for col in df.columns:
            if "lag_" in col or "rolling_" in col or "same_hour_" in col:
                agg_rules[col] = "mean"

        # Resample
        df_resampled = df_resampled.resample(self.target_frequency).agg(agg_rules)
        df_resampled = df_resampled.reset_index()

        logger.info(f"Resampled to {len(df_resampled):,} records")
        return df_resampled

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and identify potential issues.

        Args:
            df: Input dataframe to validate.

        Returns:
            Dict[str, Any]: Data quality report with issues and statistics.
        """
        logger.info("Validating data quality...")

        quality_report = {
            "total_records": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_timestamps": df["DATE_TIME"].duplicated().sum(),
            "negative_power": (df["AC_POWER"] < 0).sum(),
            "zero_irradiation_with_power": (
                (df["IRRADIATION"] == 0) & (df["AC_POWER"] > 0)
            ).sum(),
            "temperature_anomalies": {
                "extreme_ambient": (
                    (df["AMBIENT_TEMPERATURE"] < -10) | (df["AMBIENT_TEMPERATURE"] > 50)
                ).sum(),
                "extreme_module": (
                    (df["MODULE_TEMPERATURE"] < -10) | (df["MODULE_TEMPERATURE"] > 80)
                ).sum(),
            },
            "data_range": {
                "ac_power": {"min": df["AC_POWER"].min(), "max": df["AC_POWER"].max()},
                "irradiation": {
                    "min": df["IRRADIATION"].min(),
                    "max": df["IRRADIATION"].max(),
                },
                "ambient_temp": {
                    "min": df["AMBIENT_TEMPERATURE"].min(),
                    "max": df["AMBIENT_TEMPERATURE"].max(),
                },
            },
        }

        # Quality score calculation
        total_issues = (
            quality_report["duplicate_timestamps"]
            + quality_report["negative_power"]
            + quality_report["zero_irradiation_with_power"]
            + sum(quality_report["temperature_anomalies"].values())
        )
        quality_report["quality_score"] = max(0, 1 - (total_issues / len(df)))

        logger.info(f"Data quality score: {quality_report['quality_score']:.3f}")
        return quality_report

    def fit_transform(
        self, generation_path: str, weather_path: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete preprocessing pipeline: fit and transform data.

        Args:
            generation_path: Path to generation CSV file.
            weather_path: Path to weather CSV file.

        Returns:
            Tuple containing:
                - pd.DataFrame: Fully preprocessed dataset ready for ML.
                - Dict[str, Any]: Comprehensive preprocessing metadata.

        Example:
            >>> preprocessor = SolarDataPreprocessor()
            >>> features_df, metadata = preprocessor.fit_transform(
            ...     "data/raw/Plant_1_Generation_Data.csv",
            ...     "data/raw/Plant_1_Weather_Sensor_Data.csv"
            ... )
            >>> print(f"Features shape: {features_df.shape}")
        """
        logger.info("Starting complete preprocessing pipeline...")

        # Step 1: Load and merge data
        merged_df, load_metadata = self.load_and_merge_data(
            generation_path, weather_path
        )

        # Step 2: Create all features
        processed_df = self.create_temporal_features(merged_df)
        processed_df = self.create_weather_features(processed_df)
        processed_df = self.create_lag_features(processed_df)

        # Step 3: Resample to target frequency
        processed_df = self.resample_to_target_frequency(processed_df)

        # Step 4: Data quality validation
        quality_report = self.validate_data_quality(processed_df)

        # Step 5: Handle categorical variables
        processed_df = pd.get_dummies(
            processed_df, columns=["season", "temp_category", "irradiation_category"]
        )

        # Step 6: Scale features (if enabled)
        numeric_features = processed_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        numeric_features.remove("AC_POWER")  # Don't scale target

        if self.scaler is not None:
            processed_df[numeric_features] = self.scaler.fit_transform(
                processed_df[numeric_features]
            )

        # Step 7: Remove rows with NaN (from lag features)
        initial_rows = len(processed_df)
        processed_df = processed_df.dropna()
        final_rows = len(processed_df)

        # Compile metadata
        self.feature_metadata = {
            "load_metadata": load_metadata,
            "quality_report": quality_report,
            "processing_steps": {
                "initial_merged_records": load_metadata["merged_records"],
                "after_feature_engineering": initial_rows,
                "final_records_after_dropna": final_rows,
                "rows_dropped_due_to_nan": initial_rows - final_rows,
            },
            "feature_counts": {
                "total_features": processed_df.shape[1] - 1,  # Exclude target
                "temporal_features": 10,
                "weather_features": 7,
                "lag_features": len(self.lag_hours) * 2
                + 7,  # 2 vars per lag + rolling stats
                "categorical_features": processed_df.select_dtypes(
                    include=["uint8"]
                ).shape[1],
            },
            "target_frequency": self.target_frequency,
            "scaling_method": self.scaling_method,
        }

        self.is_fitted = True

        logger.info(f"Preprocessing complete! Final shape: {processed_df.shape}")
        logger.info(f"Total features: {processed_df.shape[1] - 1}")

        return processed_df, self.feature_metadata

    def transform(self, generation_path: str, weather_path: str) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.

        Args:
            generation_path: Path to generation CSV file.
            weather_path: Path to weather CSV file.

        Returns:
            pd.DataFrame: Transformed dataset using fitted parameters.

        Raises:
            RuntimeError: When preprocessor is not fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        # Apply same transformation pipeline
        processed_df, _ = self.fit_transform(generation_path, weather_path)
        return processed_df

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

        # This would be set during fit_transform
        return self.feature_metadata.get("feature_names", [])
