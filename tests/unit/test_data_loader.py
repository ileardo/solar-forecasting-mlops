"""
Unit tests for data loading functionality.

These tests validate the data loading utilities without requiring
the actual dataset files to be present.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data.loader import (
    DataLoadError,
    get_dataset_summary,
    load_generation_data,
    load_solar_data,
    load_weather_data,
    validate_data_integrity,
)


class TestDataLoader(unittest.TestCase):
    """Test cases for data loading utilities."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Mock generation data
        self.mock_generation_data = pd.DataFrame(
            {
                "DATE_TIME": [
                    "2023-05-15 06:00:00",
                    "2023-05-15 06:15:00",
                    "2023-05-15 06:30:00",
                ],
                "PLANT_ID": [1, 1, 1],
                "SOURCE_KEY": ["1BY6WEcLGh8j5v7", "1BY6WEcLGh8j5v7", "1BY6WEcLGh8j5v7"],
                "DC_POWER": [10.5, 25.8, 45.2],
                "AC_POWER": [9.8, 24.1, 42.5],
                "DAILY_YIELD": [100.5, 150.2, 200.8],
                "TOTAL_YIELD": [50000.0, 50050.2, 50100.8],
            }
        )

        # Mock weather data
        self.mock_weather_data = pd.DataFrame(
            {
                "DATE_TIME": [
                    "2023-05-15 06:00:00",
                    "2023-05-15 06:15:00",
                    "2023-05-15 06:30:00",
                ],
                "PLANT_ID": [1, 1, 1],
                "SOURCE_KEY": ["HztaOTm8xT8CJI5", "HztaOTm8xT8CJI5", "HztaOTm8xT8CJI5"],
                "AMBIENT_TEMPERATURE": [25.5, 26.2, 27.1],
                "MODULE_TEMPERATURE": [35.8, 38.4, 42.1],
                "IRRADIATION": [0.1, 0.3, 0.5],
            }
        )

    def test_get_dataset_summary(self) -> None:
        """Test dataset summary function."""
        summary = get_dataset_summary()

        self.assertIsInstance(summary, dict)
        self.assertIn("dataset_source", summary)
        self.assertIn("expected_shapes", summary)
        self.assertIn("generation_columns", summary)
        self.assertIn("weather_columns", summary)
        self.assertEqual(summary["focus"], "Plant 1 only")
        self.assertEqual(
            summary["primary_target"], "AC_POWER (solar power generation in kW)"
        )

    @patch("pandas.read_csv")
    @patch("pathlib.Path.exists")
    def test_load_generation_data_success(
        self, mock_exists: MagicMock, mock_read_csv: MagicMock
    ) -> None:
        """Test successful loading of generation data."""
        # Setup mocks
        mock_exists.return_value = True
        mock_read_csv.return_value = self.mock_generation_data.copy()

        # Test loading
        df = load_generation_data()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn("AC_POWER", df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["DATE_TIME"]))

    @patch("pathlib.Path.exists")
    def test_load_generation_data_file_not_found(self, mock_exists: MagicMock) -> None:
        """Test error handling when generation data file is missing."""
        mock_exists.return_value = False

        with self.assertRaises(DataLoadError) as context:
            load_generation_data()

        self.assertIn("Generation data file not found", str(context.exception))

    @patch("pandas.read_csv")
    @patch("pathlib.Path.exists")
    def test_load_weather_data_success(
        self, mock_exists: MagicMock, mock_read_csv: MagicMock
    ) -> None:
        """Test successful loading of weather data."""
        # Setup mocks
        mock_exists.return_value = True
        mock_read_csv.return_value = self.mock_weather_data.copy()

        # Test loading
        df = load_weather_data()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn("IRRADIATION", df.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["DATE_TIME"]))

    @patch("pathlib.Path.exists")
    def test_load_weather_data_file_not_found(self, mock_exists: MagicMock) -> None:
        """Test error handling when weather data file is missing."""
        mock_exists.return_value = False

        with self.assertRaises(DataLoadError) as context:
            load_weather_data()

        self.assertIn("Weather data file not found", str(context.exception))

    @patch("pandas.read_csv")
    @patch("pathlib.Path.exists")
    def test_load_solar_data_success(
        self, mock_exists: MagicMock, mock_read_csv: MagicMock
    ) -> None:
        """Test successful loading of both datasets."""
        # Setup mocks
        mock_exists.return_value = True
        mock_read_csv.side_effect = [
            self.mock_generation_data.copy(),
            self.mock_weather_data.copy(),
        ]

        # Test loading
        gen_df, weather_df = load_solar_data()

        self.assertIsInstance(gen_df, pd.DataFrame)
        self.assertIsInstance(weather_df, pd.DataFrame)
        self.assertEqual(len(gen_df), 3)
        self.assertEqual(len(weather_df), 3)

    def test_validate_data_integrity_clean_data(self) -> None:
        """Test data validation with clean data."""
        # Convert DATE_TIME to datetime for realistic test
        gen_df = self.mock_generation_data.copy()
        weather_df = self.mock_weather_data.copy()
        gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"])
        weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"])

        validation_results = validate_data_integrity(gen_df, weather_df)

        self.assertIsInstance(validation_results, dict)
        self.assertIn("row_counts", validation_results)
        self.assertIn("date_ranges", validation_results)
        self.assertIn("missing_values", validation_results)
        self.assertIn("data_quality_issues", validation_results)

        # Check row counts
        self.assertEqual(validation_results["row_counts"]["generation"], 3)
        self.assertEqual(validation_results["row_counts"]["weather"], 3)

        # Should have no data quality issues with clean data
        self.assertEqual(len(validation_results["data_quality_issues"]), 0)

    def test_validate_data_integrity_with_issues(self) -> None:
        """Test data validation with problematic data."""
        # Create data with issues
        gen_df = self.mock_generation_data.copy()
        weather_df = self.mock_weather_data.copy()

        # Convert DATE_TIME to datetime
        gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"])
        weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"])

        # Introduce issues
        gen_df.loc[0, "AC_POWER"] = -10.0  # Negative power
        weather_df.loc[0, "AMBIENT_TEMPERATURE"] = -60.0  # Unrealistic temperature
        weather_df.loc[1, "IRRADIATION"] = -0.5  # Negative irradiation

        validation_results = validate_data_integrity(gen_df, weather_df)

        # Should detect issues
        issues = validation_results["data_quality_issues"]
        self.assertGreater(len(issues), 0)

        # Check specific issues are detected
        issue_text = " ".join(issues)
        self.assertIn("Negative AC_POWER", issue_text)
        self.assertIn("Unrealistic ambient temperature", issue_text)
        self.assertIn("Negative irradiation", issue_text)

    @patch("pandas.read_csv")
    @patch("pathlib.Path.exists")
    def test_load_generation_data_missing_columns(
        self, mock_exists: MagicMock, mock_read_csv: MagicMock
    ) -> None:
        """Test error handling when required columns are missing."""
        # Setup mocks with incomplete data
        mock_exists.return_value = True
        incomplete_data = pd.DataFrame(
            {
                "DATE_TIME": ["2023-05-15 06:00:00"],
                "PLANT_ID": [1],
                # Missing other required columns
            }
        )
        mock_read_csv.return_value = incomplete_data

        with self.assertRaises(DataLoadError) as context:
            load_generation_data()

        self.assertIn("Missing expected columns", str(context.exception))


class TestDataLoaderIntegration(unittest.TestCase):
    """Integration tests that require actual dataset files."""

    def setUp(self) -> None:
        """Set up for integration tests."""
        self.data_path = Path("data/raw")
        self.generation_file = self.data_path / "Plant_1_Generation_Data.csv"
        self.weather_file = self.data_path / "Plant_1_Weather_Sensor_Data.csv"

    @unittest.skipUnless(
        Path("data/raw/Plant_1_Generation_Data.csv").exists()
        and Path("data/raw/Plant_1_Weather_Sensor_Data.csv").exists(),
        "Dataset files not found - download dataset first",
    )
    def test_load_real_dataset(self) -> None:
        """Integration test with real dataset files."""
        try:
            # Load real data
            gen_df, weather_df = load_solar_data()

            # Basic validation
            self.assertIsInstance(gen_df, pd.DataFrame)
            self.assertIsInstance(weather_df, pd.DataFrame)

            # Check expected approximate sizes (from Kaggle description)
            self.assertGreater(len(gen_df), 60000)  # ~68K expected
            self.assertGreater(len(weather_df), 3000)  # ~3K expected

            # Validate data integrity
            validation_results = validate_data_integrity(gen_df, weather_df)

            # Print results for manual inspection
            print(f"\n📊 Real Dataset Test Results:")
            print(
                f"Generation records: {validation_results['row_counts']['generation']:,}"
            )
            print(f"Weather records: {validation_results['row_counts']['weather']:,}")

            if validation_results["data_quality_issues"]:
                print(
                    f"⚠️  Issues found: {len(validation_results['data_quality_issues'])}"
                )
                for issue in validation_results["data_quality_issues"]:
                    print(f"  - {issue}")
            else:
                print("✅ No data quality issues detected")

        except DataLoadError as e:
            self.fail(f"Failed to load real dataset: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
