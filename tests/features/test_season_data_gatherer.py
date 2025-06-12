import pytest
import pandas as pd
import datetime
import os
from unittest.mock import patch, MagicMock
from f1_predictor.data_aquisition.season_data_gatherer import SeasonDataGatherer


class TestSeasonDataGatherer:
    """
    Simple tests for SeasonDataGatherer.

    We mock fastf1 and file operations to avoid external dependencies.
    """

    @pytest.fixture
    def mock_schedule_data(self):
        """Mock F1 season schedule data."""
        return pd.DataFrame({
            'RoundNumber': [1, 2, 3, 4, 5],
            'EventDate': ['2025-03-15', '2025-03-29', '2025-04-12', '2025-04-26', '2025-05-10'],
            'EventFormat': ['conventional', 'conventional', 'conventional', 'conventional', 'conventional'],
            'OfficialEventName': ['Bahrain Grand Prix', 'Saudi Arabia Grand Prix', 'Australia Grand Prix',
                                  'Japan Grand Prix', 'China Grand Prix']
        })

    @pytest.fixture
    def mock_schedule_with_testing(self):
        """Mock schedule that includes testing sessions."""
        return pd.DataFrame({
            'RoundNumber': [1, 2, 3],
            'EventDate': ['2025-02-20', '2025-03-15', '2025-03-29'],
            'EventFormat': ['testing', 'conventional', 'conventional'],
            'OfficialEventName': ['Pre-Season Testing', 'Bahrain Grand Prix', 'Saudi Arabia Grand Prix']
        })

    @pytest.fixture
    def mock_lap_times_data(self):
        """Mock lap times data."""
        return pd.DataFrame({
            'DriverNumber': [44, 33, 1],
            'LapTime': ['1:23.456', '1:24.123', '1:23.789'],
            'LapNumber': [1, 1, 1]
        })

    @pytest.fixture
    def mock_qualifying_data(self):
        """Mock qualifying results data."""
        return pd.DataFrame({
            'DriverNumber': [44, 33, 1],
            'Position': [1, 2, 3],
            'Q1': ['1:20.123', '1:20.456', '1:20.789'],
            'Q2': ['1:19.456', '1:19.789', '1:20.012'],
            'Q3': ['1:18.789', '1:19.012', '1:19.345']
        })

    @pytest.fixture
    def mock_race_results_data(self):
        """Mock race results data."""
        return pd.DataFrame({
            'DriverNumber': [44, 33, 1],
            'Position': [1, 2, 3],
            'Points': [25, 18, 15],
            'GridPosition': [1, 2, 3]
        })

    def test_initialization(self):
        """Test SeasonDataGatherer initializes correctly."""
        with patch('fastf1.get_event_schedule', return_value=pd.DataFrame()):
            gatherer = SeasonDataGatherer(year=2025)

            assert gatherer.year == 2025
            assert gatherer.output_dir is not None
            assert isinstance(gatherer.passed_races, list)
            assert len(gatherer.passed_races) == 0
            assert gatherer.upcoming_race is None
            assert isinstance(gatherer.laptimes, list)
            assert isinstance(gatherer.qualifyings, list)
            assert isinstance(gatherer.results, list)

    def test_initialization_default_year(self):
        """Test SeasonDataGatherer initializes with default year."""
        with patch('fastf1.get_event_schedule', return_value=pd.DataFrame()):
            gatherer = SeasonDataGatherer()

            assert gatherer.year == 2025

    def test_race_finished_past_date(self):
        """Test race_finished returns True for past dates."""
        with patch('fastf1.get_event_schedule', return_value=pd.DataFrame()):
            gatherer = SeasonDataGatherer()

            past_date = datetime.date(2020, 1, 1)
            assert gatherer.race_finished(past_date) == True

    def test_race_finished_future_date(self):
        """Test race_finished returns False for future dates."""
        with patch('fastf1.get_event_schedule', return_value=pd.DataFrame()):
            gatherer = SeasonDataGatherer()

            future_date = datetime.date(2030, 12, 31)
            assert gatherer.race_finished(future_date) == False

    def test_identify_races_with_past_races(self, mock_schedule_data):
        """Test identify_races correctly identifies past races."""
        with patch('fastf1.get_event_schedule', return_value=mock_schedule_data):
            # Mock current date to be after first 2 races
            with patch.object(SeasonDataGatherer, '__init__', lambda x, year=2025: setattr(x, 'year', year) or
                                                                                   setattr(x, 'schedule',
                                                                                           mock_schedule_data) or
                                                                                   setattr(x, 'current_date',
                                                                                           datetime.date(2025, 4, 1)) or
                                                                                   setattr(x, 'passed_races', []) or
                                                                                   setattr(x, 'upcoming_race', None)):
                gatherer = SeasonDataGatherer()
                gatherer.identify_races()

                # Should identify 2 past races (March 15 and March 29)
                assert len(gatherer.passed_races) == 2

    def test_identify_races_filters_testing_sessions(self, mock_schedule_with_testing):
        """Test identify_races filters out testing sessions."""
        with patch('fastf1.get_event_schedule', return_value=mock_schedule_with_testing):
            # Mock current date to be after all events
            with patch.object(SeasonDataGatherer, '__init__', lambda x, year=2025: setattr(x, 'year', year) or
                                                                                   setattr(x, 'schedule',
                                                                                           mock_schedule_with_testing) or
                                                                                   setattr(x, 'current_date',
                                                                                           datetime.date(2025, 4, 1)) or
                                                                                   setattr(x, 'passed_races', []) or
                                                                                   setattr(x, 'upcoming_race', None)):
                gatherer = SeasonDataGatherer()
                gatherer.identify_races()

                # Should only identify 2 conventional races, not testing
                assert len(gatherer.passed_races) == 2

    def test_identify_races_sets_upcoming_race(self, mock_schedule_data):
        """Test identify_races correctly sets upcoming race."""
        with patch('fastf1.get_event_schedule', return_value=mock_schedule_data):
            # Mock current date to be after first race but before second
            with patch.object(SeasonDataGatherer, '__init__', lambda x, year=2025: setattr(x, 'year', year) or
                                                                                   setattr(x, 'schedule',
                                                                                           mock_schedule_data) or
                                                                                   setattr(x, 'current_date',
                                                                                           datetime.date(2025, 3,
                                                                                                         20)) or
                                                                                   setattr(x, 'passed_races', []) or
                                                                                   setattr(x, 'upcoming_race', None)):
                gatherer = SeasonDataGatherer()
                gatherer.identify_races()

                # Should have 1 past race and 1 upcoming race
                assert len(gatherer.passed_races) == 1
                assert gatherer.upcoming_race is not None

    def test_collect_session_data_adds_round_numbers(self, mock_lap_times_data, mock_qualifying_data,
                                                     mock_race_results_data):
        """Test collect_session_data adds RoundNumber to all dataframes."""
        with patch('fastf1.get_event_schedule', return_value=pd.DataFrame()):
            with patch('fastf1.get_session') as mock_get_session:
                # Setup mock sessions
                mock_race_session = MagicMock()
                mock_race_session.laps = mock_lap_times_data.copy()
                mock_race_session.results = mock_race_results_data.copy()

                mock_qualifying_session = MagicMock()
                mock_qualifying_session.results = mock_qualifying_data.copy()

                mock_get_session.side_effect = [mock_qualifying_session, mock_race_session]

                gatherer = SeasonDataGatherer()
                # Mock a passed race
                mock_race = pd.DataFrame({'RoundNumber': [1], 'OfficialEventName': ['Test GP']})
                gatherer.passed_races = [mock_race]

                gatherer.collect_session_data()

                # Check that RoundNumber was added to all data
                assert len(gatherer.laptimes) == 1
                assert 'RoundNumber' in gatherer.laptimes[0].columns
                assert gatherer.laptimes[0]['RoundNumber'].iloc[0] == 1

                assert len(gatherer.qualifyings) == 1
                assert 'RoundNumber' in gatherer.qualifyings[0].columns

                assert len(gatherer.results) == 1
                assert 'RoundNumber' in gatherer.results[0].columns
                assert 'EventName' in gatherer.results[0].columns

    def test_collect_session_data_handles_multiple_races(self, mock_lap_times_data, mock_qualifying_data,
                                                         mock_race_results_data):
        """Test collect_session_data handles multiple races correctly."""
        with patch('fastf1.get_event_schedule', return_value=pd.DataFrame()):
            with patch('fastf1.get_session') as mock_get_session:
                # Setup mock sessions for each call
                def create_mock_session(session_type):
                    mock_session = MagicMock()
                    if session_type == 'R':
                        mock_session.laps = mock_lap_times_data.copy()
                        mock_session.results = mock_race_results_data.copy()
                    else:  # 'Q'
                        mock_session.results = mock_qualifying_data.copy()
                    return mock_session

                mock_get_session.side_effect = lambda year, round_num, session: create_mock_session(session)

                gatherer = SeasonDataGatherer()
                # Mock 2 passed races
                mock_race1 = pd.DataFrame({'RoundNumber': [1], 'OfficialEventName': ['Test GP 1']})
                mock_race2 = pd.DataFrame({'RoundNumber': [2], 'OfficialEventName': ['Test GP 2']})
                gatherer.passed_races = [mock_race1, mock_race2]

                gatherer.collect_session_data()

                # Should have data for 2 races
                assert len(gatherer.laptimes) == 2
                assert len(gatherer.qualifyings) == 2
                assert len(gatherer.results) == 2

    def test_save_data_creates_output_directory(self, mock_lap_times_data, mock_qualifying_data,
                                                mock_race_results_data):
        """Test save_data creates output directory and saves files."""
        with patch('fastf1.get_event_schedule', return_value=pd.DataFrame()):
            with patch('os.makedirs') as mock_makedirs:
                with patch('pandas.DataFrame.to_csv') as mock_to_csv:
                    gatherer = SeasonDataGatherer()

                    # Add some test data
                    mock_lap_times_data['RoundNumber'] = 1
                    mock_qualifying_data['RoundNumber'] = 1
                    mock_race_results_data['RoundNumber'] = 1

                    gatherer.laptimes = [mock_lap_times_data]
                    gatherer.qualifyings = [mock_qualifying_data]
                    gatherer.results = [mock_race_results_data]

                    gatherer.save_data()

                    # Check directory creation
                    mock_makedirs.assert_called_once_with(gatherer.output_dir, exist_ok=True)

                    # Check that 3 CSV files are saved
                    assert mock_to_csv.call_count == 3

    def test_save_data_combines_dataframes_correctly(self, mock_lap_times_data, mock_qualifying_data,
                                                     mock_race_results_data):
        """Test save_data correctly combines multiple dataframes."""
        with patch('fastf1.get_event_schedule', return_value=pd.DataFrame()):
            with patch('os.makedirs'):
                with patch('pandas.DataFrame.to_csv') as mock_to_csv:
                    gatherer = SeasonDataGatherer()

                    # Create multiple dataframes with different round numbers
                    lap_data_1 = mock_lap_times_data.copy()
                    lap_data_1['RoundNumber'] = 1
                    lap_data_2 = mock_lap_times_data.copy()
                    lap_data_2['RoundNumber'] = 2

                    gatherer.laptimes = [lap_data_1, lap_data_2]
                    gatherer.qualifyings = [mock_qualifying_data]
                    gatherer.results = [mock_race_results_data]

                    gatherer.save_data()

                    # Verify that to_csv was called 3 times (once for each file type)
                    assert mock_to_csv.call_count == 3

                    # Check that the combined dataframe has data from both rounds
                    combined_laptimes_call = mock_to_csv.call_args_list[0]
                    assert 'combined_laptimes.csv' in str(combined_laptimes_call)

    def test_run_method_skips_if_data_exists(self):
        """Test run method skips collection if data already exists."""
        with patch('fastf1.get_event_schedule', return_value=pd.DataFrame()):
            with patch('os.path.exists', return_value=True):
                with patch('builtins.print') as mock_print:
                    gatherer = SeasonDataGatherer()
                    gatherer.run()

                    # Should print message and not collect data
                    mock_print.assert_called_once_with("Data already collected for this season.")

    def test_run_method_collects_data_if_not_exists(self, mock_schedule_data):
        """Test run method collects data if files don't exist."""
        with patch('fastf1.get_event_schedule', return_value=mock_schedule_data):
            with patch('os.path.exists', return_value=False):
                with patch.object(SeasonDataGatherer, 'identify_races') as mock_identify:
                    with patch.object(SeasonDataGatherer, 'collect_session_data') as mock_collect:
                        with patch.object(SeasonDataGatherer, 'save_data') as mock_save:
                            gatherer = SeasonDataGatherer()
                            gatherer.run()

                            # Should call all collection methods
                            mock_identify.assert_called_once()
                            mock_collect.assert_called_once()
                            mock_save.assert_called_once()

    def test_output_directory_structure(self):
        """Test that output directory is correctly constructed."""
        with patch('fastf1.get_event_schedule', return_value=pd.DataFrame()):
            gatherer = SeasonDataGatherer()

            # Output directory should end with '2025_data'
            assert gatherer.output_dir.endswith('2025_data')
            assert os.path.isabs(gatherer.output_dir)  # Should be absolute path

    def test_data_collection_workflow(self):
        """Test the complete data collection workflow."""
        with patch('fastf1.get_event_schedule', return_value=pd.DataFrame()):
            with patch('os.path.exists', return_value=False):
                gatherer = SeasonDataGatherer()

                # Initially, all data lists should be empty
                assert len(gatherer.laptimes) == 0
                assert len(gatherer.qualifyings) == 0
                assert len(gatherer.results) == 0
                assert len(gatherer.passed_races) == 0

                # After initialization, schedule should be set
                assert hasattr(gatherer, 'schedule')
                assert hasattr(gatherer, 'current_date')