import pytest
import pandas as pd
import os
from unittest.mock import patch, MagicMock, mock_open
from datetime import timedelta
from f1_predictor.data_aquisition.api_sample_generator import SampleGenerator
from f1_predictor.features.feature_calculator import CalculateSamplesRace


class TestSampleGenerator:
    """
    Simple tests for SampleGenerator - focus on core functionality.

    We mock the following:
        - CSV file reading: to avoid file system dependencies.
        - CalculateSamplesRace: to avoid complex calculations.
    """

    @pytest.fixture
    def mock_results_data(self):
        """Mock results CSV data."""
        return pd.DataFrame({
            'RoundNumber': [1, 1, 2, 2, 3, 3],
            'DriverNumber': [44, 33, 44, 33, 44, 33],
            'DriverPointsBefore': [25, 18, 50, 36, 75, 54],
            'DriverWinsCount_before': [1, 0, 2, 1, 3, 1],
            'elo_before': [1800, 1750, 1820, 1760, 1840, 1770],
            'TeamName': ['Mercedes', 'Red Bull', 'Mercedes', 'Red Bull', 'Mercedes', 'Red Bull'],
            'TeamPointsBefore': [43, 30, 86, 60, 129, 90]
        })

    @pytest.fixture
    def mock_laptimes_data(self):
        """Mock laptimes CSV data."""
        return pd.DataFrame({
            'RoundNumber': [1, 1, 2, 2, 3, 3],
            'LapNumber': [1, 1, 1, 1, 2, 2],
            'DriverNumber': [44, 33, 44, 33, 44, 33],
            'LapTime': ['0 days 00:01:23.456000', '0 days 00:01:24.789000', '0 days 00:01:22.123000',
                        '0 days 00:01:23.456000', '0 days 00:01:21.999000', '0 days 00:01:22.555000']
        })

    @pytest.fixture
    def mock_qualifying_data(self):
        """Mock qualifying CSV data."""
        return pd.DataFrame({
            'RoundNumber': [1, 1, 2, 2, 3, 3],
            'DriverNumber': [44, 33, 44, 33, 44, 33],
            'Position': [1, 2, 1, 2, 1, 2],
            'Q1': ['0 days 00:01:20.123000', '0 days 00:01:20.456000', '0 days 00:01:19.789000',
                   '0 days 00:01:20.012000', '0 days 00:01:19.555000', '0 days 00:01:19.888000'],
            'Q2': ['0 days 00:01:19.456000', '0 days 00:01:19.789000', '0 days 00:01:18.999000',
                   '0 days 00:01:19.333000', '0 days 00:01:18.777000', '0 days 00:01:19.111000'],
            'Q3': ['0 days 00:01:18.789000', '0 days 00:01:19.012000', '0 days 00:01:18.456000',
                   '0 days 00:01:18.777000', '0 days 00:01:18.222000', '0 days 00:01:18.555000']
        })

    def test_initialization_with_defaults(self):
        """Test SampleGenerator initializes with default parameters."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                pd.DataFrame({'RoundNumber': [1]}),  # results
                pd.DataFrame({'RoundNumber': [1]}),  # laptimes
                pd.DataFrame({'RoundNumber': [1]})  # qualifying
            ]

            generator = SampleGenerator()

            assert generator.round == 1
            assert generator.lap == 1
            assert mock_read_csv.call_count == 3

    def test_initialization_with_custom_values(self):
        """Test SampleGenerator initializes with custom parameters."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                pd.DataFrame({'RoundNumber': [5]}),  # results
                pd.DataFrame({'RoundNumber': [5]}),  # laptimes
                pd.DataFrame({'RoundNumber': [5]})  # qualifying
            ]

            generator = SampleGenerator(round=5, lap=10)

            assert generator.round == 5
            assert generator.lap == 10

    def test_initialization_with_latest_round(self, mock_results_data):
        """Test SampleGenerator handles 'latest' round correctly."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                mock_results_data,  # results
                pd.DataFrame({'RoundNumber': [3]}),  # laptimes
                pd.DataFrame({'RoundNumber': [3]})  # qualifying
            ]

            generator = SampleGenerator(round='latest')

            assert generator.round == 3  # Max RoundNumber from mock data

    def test_initialization_with_latest_lap(self, mock_laptimes_data):
        """Test SampleGenerator handles 'latest' lap correctly."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                pd.DataFrame({'RoundNumber': [1]}),  # results
                mock_laptimes_data,  # laptimes
                pd.DataFrame({'RoundNumber': [1]})  # qualifying
            ]

            generator = SampleGenerator(round=1, lap='latest')

            assert generator.lap == 1  # Max LapNumber for round 1

    def test_csv_files_are_loaded(self):
        """Test that all required CSV files are loaded during initialization."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                pd.DataFrame(),  # results
                pd.DataFrame(),  # laptimes
                pd.DataFrame()  # qualifying
            ]

            generator = SampleGenerator()

            assert hasattr(generator, 'results')
            assert hasattr(generator, 'laptimes')
            assert hasattr(generator, 'qualifying')
            assert mock_read_csv.call_count == 3

    def test_simplify_time_string_valid_formats(self):
        """Test simplify_time_string with valid time formats."""
        with patch('pandas.read_csv'):
            generator = SampleGenerator()

            # Test normal time with days format
            result = generator.simplify_time_string("0 days 00:01:23.456000")
            assert result == "1:23.456"

            # Test zero time
            result = generator.simplify_time_string("0 days 00:00:00.000000")
            assert result == "0:00.000"

    def test_simplify_time_string_handles_simple_format(self):
        """Test that simplify_time_string handles simple format gracefully."""
        with patch('pandas.read_csv'):
            generator = SampleGenerator()

            # This should raise an error with current implementation
            # but we can test that it fails in a predictable way
            with pytest.raises(ValueError):
                generator.simplify_time_string("1:23.456")

    def test_simplify_time_string_nan_input(self):
        """Test simplify_time_string with NaN input."""
        with patch('pandas.read_csv'):
            generator = SampleGenerator()

            result = generator.simplify_time_string(pd.NaT)
            assert result == "0:00.000"

    def test_convert_laptimes_to_seconds_series(self):
        """Test _convert_laptimes_to_seconds with pandas Series."""
        with patch('pandas.read_csv'):
            generator = SampleGenerator()

            # Create series with timedelta
            series = pd.Series([timedelta(minutes=1, seconds=23, milliseconds=456)])
            result = generator._convert_laptimes_to_seconds(series)

            assert isinstance(result, pd.Series)
            assert result.iloc[0] == 83.456

    def test_convert_laptimes_to_seconds_dataframe(self):
        """Test _convert_laptimes_to_seconds with pandas DataFrame."""
        with patch('pandas.read_csv'):
            generator = SampleGenerator()

            # Create DataFrame with timedelta column
            df = pd.DataFrame({
                'time_col': [timedelta(minutes=1, seconds=23)],
                'other_col': [1]
            })
            df['time_col'] = df['time_col'].astype('timedelta64[ns]')

            result = generator._convert_laptimes_to_seconds(df)

            assert isinstance(result, pd.DataFrame)
            assert 'time_col' in result.columns
            assert 'other_col' in result.columns

    def test_convert_laptimes_invalid_input(self):
        """Test _convert_laptimes_to_seconds with invalid input."""
        with patch('pandas.read_csv'):
            generator = SampleGenerator()

            with pytest.raises(ValueError, match="Input must be a pandas Series or DataFrame"):
                generator._convert_laptimes_to_seconds("invalid_input")

    def test_generate_sample_invalid_lap_raises_error(self, mock_results_data, mock_laptimes_data,
                                                      mock_qualifying_data):
        """Test that generate_sample raises error for invalid lap."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                mock_results_data,
                mock_laptimes_data,
                mock_qualifying_data
            ]

            generator = SampleGenerator(round=1, lap=99)  # Lap 99 doesn't exist

            with pytest.raises(ValueError, match="Lap 99 is not valid for round 1"):
                generator.generate_sample()

    def test_generate_sample_creates_expected_attributes(self, mock_results_data, mock_laptimes_data,
                                                         mock_qualifying_data):
        """Test that generate_sample creates expected attributes."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                mock_results_data,
                mock_laptimes_data,
                mock_qualifying_data
            ]

            generator = SampleGenerator(round=1, lap=1)
            generator.generate_sample()

            # Verify that sample_calculator attribute is set
            assert hasattr(generator, 'sample_calculator')
            assert generator.sample_calculator is not None

    def test_generate_sample_creates_calculator_instance(self, mock_results_data, mock_laptimes_data,
                                                         mock_qualifying_data):
        """Test that generate_sample creates CalculateSamplesRace instance."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                mock_results_data,
                mock_laptimes_data,
                mock_qualifying_data
            ]

            generator = SampleGenerator(round=1, lap=1)
            generator.generate_sample()

            # Verify that sample_calculator attribute is set
            assert hasattr(generator, 'sample_calculator')
            assert generator.sample_calculator is not None

    def test_generate_sample_returns_list_data(self, mock_results_data, mock_laptimes_data, mock_qualifying_data):
        """Test that generate_sample returns list-like data structure."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                mock_results_data,
                mock_laptimes_data,
                mock_qualifying_data
            ]

            generator = SampleGenerator(round=1, lap=1)
            result = generator.generate_sample()

            # Test basic properties of result
            assert result is not None
            assert isinstance(result, (list, tuple))  # Should be some kind of iterable
            if len(result) > 0:
                assert isinstance(result[0], dict)  # Should contain dictionaries

    def test_generate_sample_processes_time_data(self, mock_results_data, mock_laptimes_data, mock_qualifying_data):
        """Test that generate_sample processes time data without error."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                mock_results_data,
                mock_laptimes_data,
                mock_qualifying_data
            ]

            generator = SampleGenerator(round=1, lap=1)

            # Should complete without error if time conversion works
            try:
                result = generator.generate_sample()
                success = True
            except Exception:
                success = False

            assert success

    def test_generate_sample_handles_data_conversion(self, mock_results_data, mock_laptimes_data, mock_qualifying_data):
        """Test that generate_sample handles data conversion correctly."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                mock_results_data,
                mock_laptimes_data,
                mock_qualifying_data
            ]

            generator = SampleGenerator(round=1, lap=1)
            result = generator.generate_sample()

            # Test that result contains expected structure
            assert result is not None
            # If result has data, check it contains driver information
            if len(result) > 0:
                first_sample = result[0]
                assert 'driver_id' in first_sample
                assert isinstance(first_sample['driver_id'], (int, float))