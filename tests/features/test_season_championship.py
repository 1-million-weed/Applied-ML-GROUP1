import pytest
import pandas as pd
import os
from unittest.mock import patch, mock_open
from f1_predictor.data_aquisition.season_championship import ChampionshipCalculator


class TestChampionshipCalculator:
    """
    Simple tests for ChampionshipCalculator - focus on basic functionality.

    We mock CSV file reading/writing to avoid file system dependencies.
    """

    @pytest.fixture
    def mock_results_data(self):
        """Mock F1 results data for testing."""
        return pd.DataFrame({
            'RoundNumber': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'DriverId': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'BroadcastName': ['Hamilton', 'Verstappen', 'Leclerc', 'Hamilton', 'Verstappen', 'Leclerc', 'Hamilton',
                              'Verstappen', 'Leclerc'],
            'TeamName': ['Mercedes', 'Red Bull', 'Ferrari', 'Mercedes', 'Red Bull', 'Ferrari', 'Mercedes', 'Red Bull',
                         'Ferrari'],
            'ClassifiedPosition': [1, 2, 3, 2, 1, 3, 1, 3, 2],
            'Points': [25, 18, 15, 18, 25, 15, 25, 15, 18]
        })

    @pytest.fixture
    def simple_race_data(self):
        """Simple two-driver, two-race scenario."""
        return pd.DataFrame({
            'RoundNumber': [1, 1, 2, 2],
            'DriverId': [44, 33, 44, 33],
            'BroadcastName': ['Hamilton', 'Verstappen', 'Hamilton', 'Verstappen'],
            'TeamName': ['Mercedes', 'Red Bull', 'Mercedes', 'Red Bull'],
            'ClassifiedPosition': [1, 2, 2, 1],
            'Points': [25, 18, 18, 25]
        })

    def test_initialization(self):
        """Test ChampionshipCalculator initializes correctly."""
        calculator = ChampionshipCalculator()

        assert calculator.csv_file_path is not None
        assert calculator.output_file_path is not None
        assert calculator.df is None

    def test_initialization_with_custom_output_path(self):
        """Test ChampionshipCalculator with custom output path."""
        custom_path = '/custom/path/output.csv'
        calculator = ChampionshipCalculator(output_file_path=custom_path)

        assert calculator.output_file_path == custom_path

    def test_calculate_championship_stats_adds_required_columns(self, simple_race_data):
        """Test that championship calculation adds all required columns."""
        with patch('pandas.read_csv', return_value=simple_race_data):
            with patch('pandas.DataFrame.to_csv'):
                calculator = ChampionshipCalculator()
                result_df = calculator.calculate_championship_stats()

                # Check that all new columns are added
                expected_columns = [
                    'DriverChampionshipPosition',
                    'TeamChampionshipPoints',
                    'DriverWinsCount',
                    'DriverWinsCount_before',
                    'DriverPointsBefore',
                    'TeamPointsBefore'
                ]

                for col in expected_columns:
                    assert col in result_df.columns

    def test_driver_points_before_calculation(self, simple_race_data):
        """Test that DriverPointsBefore is calculated correctly."""
        with patch('pandas.read_csv', return_value=simple_race_data):
            with patch('pandas.DataFrame.to_csv'):
                calculator = ChampionshipCalculator()
                result_df = calculator.calculate_championship_stats()

                # Round 1: Both drivers should have 0 points before
                round1_data = result_df[result_df['RoundNumber'] == 1]
                assert all(round1_data['DriverPointsBefore'] == 0)

                # Round 2: Hamilton should have 25, Verstappen should have 18
                round2_data = result_df[result_df['RoundNumber'] == 2]
                hamilton_round2 = round2_data[round2_data['DriverId'] == 44]['DriverPointsBefore'].iloc[0]
                verstappen_round2 = round2_data[round2_data['DriverId'] == 33]['DriverPointsBefore'].iloc[0]

                assert hamilton_round2 == 25
                assert verstappen_round2 == 18

    def test_driver_wins_count_calculation(self, simple_race_data):
        """Test that driver wins are counted correctly."""
        with patch('pandas.read_csv', return_value=simple_race_data):
            with patch('pandas.DataFrame.to_csv'):
                calculator = ChampionshipCalculator()
                result_df = calculator.calculate_championship_stats()

                # After round 1: Hamilton should have 1 win, Verstappen 0
                round1_data = result_df[result_df['RoundNumber'] == 1]
                hamilton_wins_r1 = round1_data[round1_data['DriverId'] == 44]['DriverWinsCount'].iloc[0]
                verstappen_wins_r1 = round1_data[round1_data['DriverId'] == 33]['DriverWinsCount'].iloc[0]

                assert hamilton_wins_r1 == 1
                assert verstappen_wins_r1 == 0

                # After round 2: Both should have 1 win each
                round2_data = result_df[result_df['RoundNumber'] == 2]
                hamilton_wins_r2 = round2_data[round2_data['DriverId'] == 44]['DriverWinsCount'].iloc[0]
                verstappen_wins_r2 = round2_data[round2_data['DriverId'] == 33]['DriverWinsCount'].iloc[0]

                assert hamilton_wins_r2 == 1
                assert verstappen_wins_r2 == 1

    def test_wins_before_calculation(self, simple_race_data):
        """Test that DriverWinsCount_before is calculated correctly."""
        with patch('pandas.read_csv', return_value=simple_race_data):
            with patch('pandas.DataFrame.to_csv'):
                calculator = ChampionshipCalculator()
                result_df = calculator.calculate_championship_stats()

                # Round 1: Both drivers should have 0 wins before
                round1_data = result_df[result_df['RoundNumber'] == 1]
                assert all(round1_data['DriverWinsCount_before'] == 0)

                # Round 2: Hamilton should have 1 win before, Verstappen 0
                round2_data = result_df[result_df['RoundNumber'] == 2]
                hamilton_wins_before = round2_data[round2_data['DriverId'] == 44]['DriverWinsCount_before'].iloc[0]
                verstappen_wins_before = round2_data[round2_data['DriverId'] == 33]['DriverWinsCount_before'].iloc[0]

                assert hamilton_wins_before == 1
                assert verstappen_wins_before == 0

    def test_championship_position_calculation(self, simple_race_data):
        """Test that championship positions are calculated correctly."""
        with patch('pandas.read_csv', return_value=simple_race_data):
            with patch('pandas.DataFrame.to_csv'):
                calculator = ChampionshipCalculator()
                result_df = calculator.calculate_championship_stats()

                # Round 1: Hamilton should be P1, Verstappen P2
                round1_data = result_df[result_df['RoundNumber'] == 1]
                hamilton_pos_r1 = round1_data[round1_data['DriverId'] == 44]['DriverChampionshipPosition'].iloc[0]
                verstappen_pos_r1 = round1_data[round1_data['DriverId'] == 33]['DriverChampionshipPosition'].iloc[0]

                assert hamilton_pos_r1 == 1
                assert verstappen_pos_r1 == 2

                # Round 2: Both tied on points (43), but Hamilton has more wins so should be P1
                round2_data = result_df[result_df['RoundNumber'] == 2]
                hamilton_pos_r2 = round2_data[round2_data['DriverId'] == 44]['DriverChampionshipPosition'].iloc[0]
                verstappen_pos_r2 = round2_data[round2_data['DriverId'] == 33]['DriverChampionshipPosition'].iloc[0]

                assert hamilton_pos_r2 == 1
                assert verstappen_pos_r2 == 2

    def test_team_points_calculation(self, simple_race_data):
        """Test that team points are calculated correctly."""
        with patch('pandas.read_csv', return_value=simple_race_data):
            with patch('pandas.DataFrame.to_csv'):
                calculator = ChampionshipCalculator()
                result_df = calculator.calculate_championship_stats()

                # Round 1: Mercedes should have 25, Red Bull 18
                round1_data = result_df[result_df['RoundNumber'] == 1]
                mercedes_points_r1 = round1_data[round1_data['TeamName'] == 'Mercedes']['TeamChampionshipPoints'].iloc[
                    0]
                redbull_points_r1 = round1_data[round1_data['TeamName'] == 'Red Bull']['TeamChampionshipPoints'].iloc[0]

                assert mercedes_points_r1 == 25
                assert redbull_points_r1 == 18

                # Round 2: Both teams should have 43 points
                round2_data = result_df[result_df['RoundNumber'] == 2]
                mercedes_points_r2 = round2_data[round2_data['TeamName'] == 'Mercedes']['TeamChampionshipPoints'].iloc[
                    0]
                redbull_points_r2 = round2_data[round2_data['TeamName'] == 'Red Bull']['TeamChampionshipPoints'].iloc[0]

                assert mercedes_points_r2 == 43
                assert redbull_points_r2 == 43

    def test_csv_file_error_handling(self):
        """Test that calculator handles CSV read errors gracefully."""
        with patch('pandas.read_csv', side_effect=FileNotFoundError("File not found")):
            calculator = ChampionshipCalculator()
            result = calculator.calculate_championship_stats()

            assert result is None

    def test_run_method_with_existing_file(self):
        """Test run method when output file already exists."""
        with patch('os.path.exists', return_value=True):
            with patch('pandas.read_csv', return_value=pd.DataFrame()) as mock_read:
                with patch('builtins.print') as mock_print:
                    calculator = ChampionshipCalculator()
                    result = calculator.run()

                    # Should read existing file and print message
                    assert mock_read.called
                    mock_print.assert_called_once()
                    assert result is not None

    def test_run_method_creates_new_file(self, simple_race_data):
        """Test run method when output file doesn't exist."""
        with patch('os.path.exists', return_value=False):
            with patch('pandas.read_csv', return_value=simple_race_data):
                with patch('pandas.DataFrame.to_csv'):
                    calculator = ChampionshipCalculator()
                    result = calculator.run()

                    # Should process data and return DataFrame
                    assert result is not None
                    assert isinstance(result, pd.DataFrame)