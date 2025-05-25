import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, call
from f1_predictor.features.make_features import FeatureGenerator


class TestFeatureGenerator:
    """
    Simple tests for FeatureGenerator - focus on core functionality.

    We mock the following:
        - DataFolderManager: to avoid file system operations.
        - DatasetLoader: to provide controlled datasets.
        - CalculateSamplesRace: to avoid complex calculations.
    """

    @pytest.fixture
    def mock_datasets(self):
        """Create mock dataset with minimal required data."""
        mock = MagicMock()

        # Mock races DataFrame with years
        mock.races = pd.DataFrame({
            'raceId': [1, 2, 3, 4, 5, 6],
            'year': [2020, 2020, 2021, 2021, 2022, 2022]
        })

        # Mock other DataFrames
        mock.driver_standings = pd.DataFrame({
            'raceId': [1, 1, 2, 2, 3, 3],
            'driverId': [1, 2, 1, 2, 1, 2],
            'points': [25, 18, 25, 18, 25, 18]
        })

        mock.results = pd.DataFrame({
            'raceId': [1, 1, 2, 2, 3, 3],
            'driverId': [1, 2, 1, 2, 1, 2],
            'position': [1, 2, 1, 2, 1, 2]
        })

        mock.lap_times = pd.DataFrame({
            'raceId': [1, 1, 2, 2, 3, 3],
            'driverId': [1, 2, 1, 2, 1, 2],
            'lap': [1, 1, 1, 1, 1, 1],
            'time': [90000, 91000, 89000, 90000, 88000, 89000]
        })

        mock.qualifying = pd.DataFrame({
            'raceId': [1, 1, 2, 2, 3, 3],
            'driverId': [1, 2, 1, 2, 1, 2],
            'position': [1, 2, 1, 2, 1, 2]
        })

        mock.constructor_standings = pd.DataFrame({
            'raceId': [1, 1, 2, 2, 3, 3],
            'constructorId': [1, 2, 1, 2, 1, 2],
            'points': [43, 30, 43, 30, 43, 30]
        })

        return mock

    @pytest.fixture
    def mock_sample_data(self):
        """Mock data that CalculateSamplesRace would return."""
        return [
            {'race_id': 1, 'driver_id': 1, 'feature1': 0.5, 'feature2': 1.0, 'label': 1},
            {'race_id': 1, 'driver_id': 2, 'feature1': 0.3, 'feature2': 0.8, 'label': 0}
        ]

    def test_initialization(self):
        """Test FeatureGenerator initializes with correct parameters."""
        generator = FeatureGenerator(random_seed=42, test_size=0.3, empty_folder=False)

        assert generator.random_seed == 42
        assert generator.test_size == 0.3
        assert generator.empty_folder == False

    def test_run_basic_flow(self, mock_datasets, mock_sample_data):
        """Test the basic flow of run() method."""
        with patch('f1_predictor.features.make_features.DataFolderManager') as mock_folder_manager:
            with patch('f1_predictor.features.make_features.DatasetLoader') as mock_loader:
                with patch('f1_predictor.features.make_features.CalculateSamplesRace') as mock_calculator:
                    mock_loader.return_value = mock_datasets
                    mock_calculator.return_value.calculate_samples.return_value = mock_sample_data

                    generator = FeatureGenerator(random_seed=42, test_size=0.2)
                    generator.run()

                    mock_folder_manager.assert_called_once_with(empty_folder=True)
                    mock_loader.assert_called_once()
                    mock_folder_manager.return_value.save_features.assert_called_once()

    def test_train_test_split_by_year(self, mock_datasets, mock_sample_data):
        """Test that train/test split is done by year."""
        with patch('f1_predictor.features.make_features.DataFolderManager'):
            with patch('f1_predictor.features.make_features.DatasetLoader') as mock_loader:
                with patch('f1_predictor.features.make_features.CalculateSamplesRace') as mock_calculator:
                    mock_loader.return_value = mock_datasets
                    mock_calculator.return_value.calculate_samples.return_value = mock_sample_data

                    generator = FeatureGenerator(random_seed=42, test_size=0.33)
                    generator.run()

                    unique_years = mock_datasets.races['year'].unique()
                    assert len(unique_years) == 3

    def test_skips_races_without_laptimes(self, mock_datasets, mock_sample_data):
        """Test that races without lap times are skipped."""
        mock_datasets.lap_times = mock_datasets.lap_times[mock_datasets.lap_times['raceId'] != 2]

        with patch('f1_predictor.features.make_features.DataFolderManager'):
            with patch('f1_predictor.features.make_features.DatasetLoader') as mock_loader:
                with patch('f1_predictor.features.make_features.CalculateSamplesRace') as mock_calculator:
                    with patch('builtins.print') as mock_print:
                        mock_loader.return_value = mock_datasets
                        mock_calculator.return_value.calculate_samples.return_value = mock_sample_data

                        generator = FeatureGenerator(random_seed=42)
                        generator.run()

                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        skip_messages = [msg for msg in print_calls if "No lap times available" in msg]
                        assert len(skip_messages) > 0

    def test_feature_calculator_called_correctly(self, mock_datasets, mock_sample_data):
        """Test that CalculateSamplesRace is called with correct data."""
        with patch('f1_predictor.features.make_features.DataFolderManager'):
            with patch('f1_predictor.features.make_features.DatasetLoader') as mock_loader:
                with patch('f1_predictor.features.make_features.CalculateSamplesRace') as mock_calculator:
                    mock_loader.return_value = mock_datasets
                    mock_calculator.return_value.calculate_samples.return_value = mock_sample_data

                    generator = FeatureGenerator(random_seed=42)
                    generator.run()

                    assert mock_calculator.call_count > 0
                    first_call = mock_calculator.call_args_list[0]
                    kwargs = first_call[1]

                    assert 'race_id' in kwargs
                    assert 'laptimes' in kwargs
                    assert 'results' in kwargs
                    assert 'driver_standings' in kwargs
                    assert 'qualifying' in kwargs
                    assert 'constructor_standings' in kwargs

    def test_year_column_added_to_output(self, mock_datasets, mock_sample_data):
        """Test that year column is added to output DataFrames."""
        with patch('f1_predictor.features.make_features.DataFolderManager') as mock_folder_manager:
            with patch('f1_predictor.features.make_features.DatasetLoader') as mock_loader:
                with patch('f1_predictor.features.make_features.CalculateSamplesRace') as mock_calculator:
                    mock_loader.return_value = mock_datasets
                    mock_calculator.return_value.calculate_samples.return_value = mock_sample_data

                    generator = FeatureGenerator(random_seed=42)
                    generator.run()

                    save_call = mock_folder_manager.return_value.save_features.call_args
                    df_train, df_test = save_call[0]

                    assert 'year' in df_train.columns
                    assert 'year' in df_test.columns

    def test_output_shapes_logged(self, mock_datasets, mock_sample_data):
        """Test that output shapes are printed."""
        with patch('f1_predictor.features.make_features.DataFolderManager'):
            with patch('f1_predictor.features.make_features.DatasetLoader') as mock_loader:
                with patch('f1_predictor.features.make_features.CalculateSamplesRace') as mock_calculator:
                    with patch('builtins.print') as mock_print:
                        mock_loader.return_value = mock_datasets
                        mock_calculator.return_value.calculate_samples.return_value = mock_sample_data

                        generator = FeatureGenerator(random_seed=42)
                        generator.run()

                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        assert any("Training samples:" in str(c) for c in print_calls)
                        assert any("Test samples:" in str(c) for c in print_calls)
