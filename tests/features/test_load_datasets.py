import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import os

from f1_predictor.features.load_datasets import DatasetLoader


class TestDatasetLoader:
    """Simple tests for DatasetLoader"""

    @pytest.fixture
    def mock_csv_data(self):
        """Fixture for mock f1 csv data."""
        return {
            'races.csv': pd.DataFrame({
                'raceId': [1, 2, 3],
                'year': [2021, 2021, 2021],
                'round': [1, 2, 3]
            }),
            'lap_times.csv': pd.DataFrame({
                'raceId': [1, 1, 2],
                'driverId': [1, 2, 1],
                'lap': [1, 1, 1],
                'time': [90000, 91000, 89000]
            }),
            'results.csv': pd.DataFrame({
                'resultId': [1, 2, 3],
                'raceId': [1, 1, 2],
                'driverId': [1, 2, 1],
                'position': [1, 2, 1]
            }),
            'qualifying.csv': pd.DataFrame({
                'qualifyId': [1, 2, 3],
                'raceId': [1, 1, 2],
                'driverId': [1, 2, 1],
                'position': [1, 2, 1]
            }),
            'constructor_standings.csv': pd.DataFrame({
                'constructorStandingsId': [1, 2],
                'raceId': [1, 1],
                'constructorId': [1, 2],
                'points': [25, 18]
            }),
            'constructors.csv': pd.DataFrame({
                'constructorId': [1, 2],
                'constructorRef': ['mercedes', 'red_bull'],
                'name': ['Mercedes', 'Red Bull'],
                'nationality': ['German', 'Austrian']
            }),
            'driver_standings_with_elo.csv': pd.DataFrame({
                'driverStandingsId': [1, 2],
                'raceId': [1, 1],
                'driverId': [1, 2],
                'points': [25, 18],
                'elo': [2000, 1950]
            })
        }

    def test_all_datasets_loaded(self, mock_csv_data):
        """Test that all required datasets are loaded as DataFrames."""
        with patch('pandas.read_csv') as mock_read_csv:
            with patch('os.path.exists', return_value=True):
                # Setup mock to return appropriate DataFrames
                def side_effect(path):
                    filename = os.path.basename(path)
                    return mock_csv_data.get(filename, pd.DataFrame())

                mock_read_csv.side_effect = side_effect

                # Create loader
                loader = DatasetLoader()

                # Check all required attributes exist and are DataFrames
                assert isinstance(loader.races, pd.DataFrame)
                assert isinstance(loader.lap_times, pd.DataFrame)
                assert isinstance(loader.results, pd.DataFrame)
                assert isinstance(loader.qualifying, pd.DataFrame)
                assert isinstance(loader.constructor_standings, pd.DataFrame)
                assert isinstance(loader.driver_standings, pd.DataFrame)

                # Check they're not empty
                assert len(loader.races) > 0
                assert len(loader.lap_times) > 0
                assert len(loader.results) > 0

    def test_constructor_merge_adds_name_column(self, mock_csv_data):
        """Test that constructor names are properly merged into standings."""
        with patch('pandas.read_csv') as mock_read_csv:
            with patch('os.path.exists', return_value=True):
                mock_read_csv.side_effect = lambda path: mock_csv_data.get(
                    os.path.basename(path), pd.DataFrame()
                )

                loader = DatasetLoader()

                # Check that 'name' column was added to constructor_standings
                assert 'name' in loader.constructor_standings.columns

                # Check that names are correctly merged
                mercedes_row = loader.constructor_standings[
                    loader.constructor_standings['constructorId'] == 1
                    ].iloc[0]
                assert mercedes_row['name'] == 'Mercedes'

    def test_missing_elo_file_raises_error(self):
        """Test that missing driver_standings_with_elo.csv raises FileNotFoundError."""
        # Create minimal DataFrames with required columns for the merge
        constructor_standings_df = pd.DataFrame({
            'constructorStandingsId': [1],
            'raceId': [1],
            'constructorId': [1],
            'points': [25]
        })

        constructors_df = pd.DataFrame({
            'constructorId': [1],
            'name': ['Mercedes']
        })

        with patch('pandas.read_csv') as mock_read_csv:
            # Return appropriate DataFrames based on filename
            def side_effect(path):
                filename = os.path.basename(path)
                if filename == 'constructor_standings.csv':
                    return constructor_standings_df
                elif filename == 'constructors.csv':
                    return constructors_df
                else:
                    return pd.DataFrame()  # Empty for other files

            mock_read_csv.side_effect = side_effect

            # This patches os.path.exists to return False only for the ELO file
            with patch('os.path.exists') as mock_exists:
                def exists_side_effect(path):
                    # Return False only for driver_standings_with_elo.csv
                    return 'driver_standings_with_elo.csv' not in path

                mock_exists.side_effect = exists_side_effect

                with pytest.raises(FileNotFoundError) as exc_info:
                    DatasetLoader()

                assert "driver_standings_with_elo.csv" in str(exc_info.value)

    def test_data_shapes_preserved(self, mock_csv_data):
        """Test that data shapes are preserved after loading."""
        with patch('pandas.read_csv') as mock_read_csv:
            with patch('os.path.exists', return_value=True):
                mock_read_csv.side_effect = lambda path: mock_csv_data.get(
                    os.path.basename(path), pd.DataFrame()
                )

                loader = DatasetLoader()

                # Check that row counts match what is in the fixture data
                assert len(loader.races) == 3
                assert len(loader.lap_times) == 3
                assert len(loader.results) == 3
                assert len(loader.qualifying) == 3
                assert len(loader.driver_standings) == 2

    def test_required_columns_exist(self, mock_csv_data):
        """Test that critical columns exist in loaded data."""
        with patch('pandas.read_csv') as mock_read_csv:
            with patch('os.path.exists', return_value=True):
                mock_read_csv.side_effect = lambda path: mock_csv_data.get(
                    os.path.basename(path), pd.DataFrame()
                )

                loader = DatasetLoader()

                # Check critical columns for joining/filtering
                assert 'raceId' in loader.races.columns
                assert 'raceId' in loader.lap_times.columns
                assert 'driverId' in loader.lap_times.columns
                assert 'raceId' in loader.results.columns
                assert 'raceId' in loader.driver_standings.columns
                assert 'elo' in loader.driver_standings.columns
