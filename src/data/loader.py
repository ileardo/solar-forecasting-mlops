"""
Data loading utilities for Solar Power Generation dataset.

This module provides functions to load and validate the solar power generation
and weather sensor data from Kaggle dataset.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Custom exception for data loading errors."""

    pass


def load_generation_data(data_path: Path = Path("data/raw")) -> pd.DataFrame:
    """
    Load Plant 1 generation data from CSV file.

    Args:
        data_path: Path to directory containing raw data files.

    Returns:
        DataFrame with generation data including columns:
        - DATE_TIME: Timestamp
        - PLANT_ID: Plant identifier (should be 1)
        - SOURCE_KEY: Inverter/sensor identifier
        - DC_POWER: DC power output (kW)
        - AC_POWER: AC power output (kW) - PRIMARY TARGET
        - DAILY_YIELD: Daily generation (kWh)
        - TOTAL_YIELD: Cumulative generation (kWh)

    Raises:
        DataLoadError: If file not found or data validation fails.

    Example:
        >>> df_gen = load_generation_data()
        >>> print(df_gen.shape)
        (68778, 7)
    """
    file_path = data_path / "Plant_1_Generation_Data.csv"

    if not file_path.exists():
        raise DataLoadError(f"Generation data file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(
            f"Loaded generation data: {df.shape[0]} rows, {df.shape[1]} columns"
        )

        # Basic validation
        expected_columns = [
            "DATE_TIME",
            "PLANT_ID",
            "SOURCE_KEY",
            "DC_POWER",
            "AC_POWER",
            "DAILY_YIELD",
            "TOTAL_YIELD",
        ]

        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            raise DataLoadError(f"Missing expected columns: {missing_columns}")

        # Convert DATE_TIME to datetime
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

        return df

    except Exception as e:
        raise DataLoadError(f"Error loading generation data: {str(e)}")


def load_weather_data(data_path: Path = Path("data/raw")) -> pd.DataFrame:
    """
    Load Plant 1 weather sensor data from CSV file.

    Args:
        data_path: Path to directory containing raw data files.

    Returns:
        DataFrame with weather data including columns:
        - DATE_TIME: Timestamp
        - PLANT_ID: Plant identifier (should be 1)
        - SOURCE_KEY: Sensor identifier
        - AMBIENT_TEMPERATURE: Ambient temperature (°C)
        - MODULE_TEMPERATURE: Module temperature (°C)
        - IRRADIATION: Solar irradiation (kW/m²)

    Raises:
        DataLoadError: If file not found or data validation fails.

    Example:
        >>> df_weather = load_weather_data()
        >>> print(df_weather.shape)
        (3182, 6)
    """
    file_path = data_path / "Plant_1_Weather_Sensor_Data.csv"

    if not file_path.exists():
        raise DataLoadError(f"Weather data file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded weather data: {df.shape[0]} rows, {df.shape[1]} columns")

        # Basic validation
        expected_columns = [
            "DATE_TIME",
            "PLANT_ID",
            "SOURCE_KEY",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION",
        ]

        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            raise DataLoadError(f"Missing expected columns: {missing_columns}")

        # Convert DATE_TIME to datetime
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

        return df

    except Exception as e:
        raise DataLoadError(f"Error loading weather data: {str(e)}")


def load_solar_data(
    data_path: Path = Path("data/raw"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both generation and weather data for Plant 1.

    Args:
        data_path: Path to directory containing raw data files.

    Returns:
        Tuple containing:
        - generation_df: DataFrame with generation data
        - weather_df: DataFrame with weather data

    Raises:
        DataLoadError: If any file loading fails.

    Example:
        >>> gen_df, weather_df = load_solar_data()
        >>> print(f"Generation: {gen_df.shape}, Weather: {weather_df.shape}")
        Generation: (68778, 7), Weather: (3182, 6)
    """
    generation_df = load_generation_data(data_path)
    weather_df = load_weather_data(data_path)

    return generation_df, weather_df


def validate_data_integrity(
    generation_df: pd.DataFrame, weather_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Perform comprehensive data integrity validation.

    Args:
        generation_df: Generation data DataFrame
        weather_df: Weather data DataFrame

    Returns:
        Dictionary with validation results including:
        - row_counts: Number of rows in each dataset
        - date_ranges: Start and end dates for each dataset
        - missing_values: Count of missing values per column
        - data_quality_issues: List of identified issues

    Example:
        >>> gen_df, weather_df = load_solar_data()
        >>> validation = validate_data_integrity(gen_df, weather_df)
        >>> print(validation['row_counts'])
        {'generation': 68778, 'weather': 3182}
    """
    validation_results = {
        "row_counts": {"generation": len(generation_df), "weather": len(weather_df)},
        "date_ranges": {
            "generation": {
                "start": generation_df["DATE_TIME"].min(),
                "end": generation_df["DATE_TIME"].max(),
            },
            "weather": {
                "start": weather_df["DATE_TIME"].min(),
                "end": weather_df["DATE_TIME"].max(),
            },
        },
        "missing_values": {
            "generation": generation_df.isnull().sum().to_dict(),
            "weather": weather_df.isnull().sum().to_dict(),
        },
        "data_quality_issues": [],
    }

    # Check for major data quality issues
    issues = []

    # Check if Plant ID is consistently 1
    if (
        generation_df["PLANT_ID"].nunique() != 1
        or generation_df["PLANT_ID"].iloc[0] != 1
    ):
        issues.append("Generation data contains multiple or incorrect plant IDs")

    if weather_df["PLANT_ID"].nunique() != 1 or weather_df["PLANT_ID"].iloc[0] != 1:
        issues.append("Weather data contains multiple or incorrect plant IDs")

    # Check for negative power values
    if (generation_df["AC_POWER"] < 0).any():
        issues.append("Negative AC_POWER values detected")

    if (generation_df["DC_POWER"] < 0).any():
        issues.append("Negative DC_POWER values detected")

    # Check for unrealistic temperature values
    if (weather_df["AMBIENT_TEMPERATURE"] < -50).any() or (
        weather_df["AMBIENT_TEMPERATURE"] > 60
    ).any():
        issues.append("Unrealistic ambient temperature values detected")

    # Check for negative irradiation
    if (weather_df["IRRADIATION"] < 0).any():
        issues.append("Negative irradiation values detected")

    validation_results["data_quality_issues"] = issues

    logger.info(f"Data validation completed. Found {len(issues)} issues.")
    for issue in issues:
        logger.warning(f"Data quality issue: {issue}")

    return validation_results


def get_dataset_summary() -> Dict[str, Any]:
    """
    Get a summary of the loaded dataset for quick reference.

    Returns:
        Dictionary with dataset summary including expected dimensions,
        column descriptions, and data types.

    Example:
        >>> summary = get_dataset_summary()
        >>> print(summary['expected_shapes'])
        {'generation': (68778, 7), 'weather': (3182, 6)}
    """
    return {
        "dataset_source": "Kaggle - Solar Power Generation Data",
        "focus": "Plant 1 only",
        "time_period": "34 days continuous data",
        "frequency": "Every 15 minutes",
        "expected_shapes": {"generation": (68778, 7), "weather": (3182, 6)},
        "primary_target": "AC_POWER (solar power generation in kW)",
        "generation_columns": [
            "DATE_TIME",
            "PLANT_ID",
            "SOURCE_KEY",
            "DC_POWER",
            "AC_POWER",
            "DAILY_YIELD",
            "TOTAL_YIELD",
        ],
        "weather_columns": [
            "DATE_TIME",
            "PLANT_ID",
            "SOURCE_KEY",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION",
        ],
    }
