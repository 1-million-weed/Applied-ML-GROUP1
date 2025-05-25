
import math
import pandas as pd
from pandas import testing as pd_testing
import numpy as np
import pytest

from f1_predictor.features.feature_calculator import CalculateSamplesRace

# --- FIXTURES --------------------------------------------------------
@pytest.fixture
def mock_calculator():
    """Minimal CalculateSamplesRace with all-empty tables."""
    empty = pd.DataFrame()
    return CalculateSamplesRace(
        race_id="R0",
        laptimes=empty,
        results=empty,
        driver_standings=empty,
        qualifying=empty,
        constructor_standings=empty,
    )

@pytest.fixture
def full_calculator():
    """
    Fixture to create a full CalculateSamplesRace object with all dataframes.
    """
    # 2 drivers, 2 laps each
    laptimes = pd.DataFrame([
        {"driverId": "D1", "lap": 1, "milliseconds": 90000},
        {"driverId": "D2", "lap": 1, "milliseconds": 92000},
        {"driverId": "D3", "lap": 1, "milliseconds": 94000},
        {"driverId": "D1", "lap": 2, "milliseconds": 89000},
        {"driverId": "D2", "lap": 2, "milliseconds": 91000},
        {"driverId": "D3", "lap": 2, "milliseconds": 93000},
    ])

    results = pd.DataFrame([
        {"driverId": "D1", "position": 1},
        {"driverId": "D2", "position": 2},
        {"driverId": "D3", "position": 3},
        {"driverId": "D4", "position": "invalid_pos"},  # For testing invalid position handling
    ])

    driver_standings = pd.DataFrame([
        {"driverId": "D1", "points": 100, "wins": 5, "elo": 1800, "constructor_name": "TeamA"},
        {"driverId": "D2", "points": 80, "wins": 2, "elo": 1600, "constructor_name": "TeamB"},
        {"driverId": "D3", "points": 60, "wins": 0, "elo": 1400, "constructor_name": "TeamC"},
        {"driverId": "D4", "points": 40, "wins": 1, "elo": 1200, "constructor_name": ""},  # Empty team name
        {"driverId": "D5", "points": 0, "wins": 0, "elo": 0, "constructor_name": "NonExistentTeam"},
    ])

    # Qualifying data with various time formats and missing values
    qualifying = pd.DataFrame([
        {"driverId": "D1", "q1": "1:20.500", "q2": "1:19.200", "q3": "1:18.800", "position": "1"},
        {"driverId": "D2", "q1": "1:21.000", "q2": "1:20.100", "q3": "", "position": "2"},
        {"driverId": "D3", "q1": "1:22.000", "q2": "\\N", "q3": "bad:format", "position": "3"},
        {"driverId": "D4", "q1": "", "q2": "", "q3": "", "position": "not_a_number"},
    ])

    constructor_standings = pd.DataFrame([
        {"name": "TeamA", "points": 150},
        {"name": "TeamB", "points": 120},
        {"name": "TeamC", "points": 90},
        # Note: "NonExistentTeam" intentionally missing for testing
    ])

    return CalculateSamplesRace(
        race_id="TEST_RACE",
        laptimes=laptimes,
        results=results,
        driver_standings=driver_standings,
        qualifying=qualifying,
        constructor_standings=constructor_standings,
    )


# --- TESTS FOR _convert_time_to_milliseconds ---------------------------------

@pytest.mark.parametrize("time_str,expected_ms", [
    ("1:27.236", int(1*60*1000 + 27.236*1000)),  # minutes and seconds
    ("87.236", int(87.236*1000)),                # just seconds
    ("", 4*60*1000),                             # blank inputs returns 4 minutes
    ("\\N", 4*60*1000),                          # NaN inputs return 4 minutes
    (None, 4*60*1000),                           # None inputs return 4 minutes
    ("bad:format", 4*60*1000),                   # incorrect inputs returns 4 minutes
    ("0:30.500", int(30.5*1000)),                # 30.5 seconds
    ("2:05.123", int(2*60*1000 + 5.123*1000)),   # 2 minutes 5.123 seconds
])
def test_convert_time(mock_calculator, time_str, expected_ms):
    """Test the _convert_time_to_milliseconds function."""
    ms = mock_calculator._convert_time_to_milliseconds(time_str)
    assert ms == expected_ms
    assert isinstance(ms, int)
    assert ms >= 0


# --- TESTS FOR _get_amount_of_wins -------------------------------------------

@pytest.mark.parametrize("driver_id,expected_wins", [
    ("D1", 5),  # Driver with wins
    ("D2", 2),  # Driver with some wins
    ("D3", 0),  # Driver with no wins
    ("D5", 0),  # Driver with zero wins
])
def test_get_amount_of_wins(full_calculator, driver_id, expected_wins):
    """Test _get_amount_of_wins for various drivers."""
    wins = full_calculator._get_amount_of_wins(driver_id)
    assert wins == expected_wins

def test_get_amount_of_wins_nonexistent_driver(full_calculator):
    """Test _get_amount_of_wins for non-existent driver raises IndexError."""
    with pytest.raises(IndexError):
        full_calculator._get_amount_of_wins("NONEXISTENT")


# --- TESTS FOR _min_max_normalize --------------------------------------------

# TODO: This is causing leakage since we normalize before splitting the data
@pytest.mark.parametrize("series,expected", [
    ([1, 2, 3, 4, 5],    [0.0, 0.25, 0.5, 0.75, 1.0]),
    ([10, 20, 30],       [0.0, 0.5, 1.0]),
    ([5, 5, 5],          [5, 5, 5]),      # Constant values
    ([42],               [42]),           # Single value
    ([],                 []),             # Empty series
    ([-10, -6, -2, 2, 6, 10], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), # Negative values
])
def test_min_max_normalize_normal_case(full_calculator, series, expected):
    """Test _min_max_normalize with dummy data."""
    # TODO: Add a test for normal data
    series = pd.Series(series)
    expected = pd.Series(expected)
    result = full_calculator._min_max_normalize(series)
    pd_testing.assert_series_equal(result, expected)

@pytest.mark.parametrize("invalid_input", [
    float('inf'),
    float('-inf'),
    float('nan'),
])
def test_min_max_normalize_with_invalid_values(full_calculator, invalid_input):
    """Test _min_max_normalize handles invalid float values."""
    series = pd.Series([1, 2, invalid_input, 4, 5])
    result = full_calculator._min_max_normalize(series)

    # Should handle invalid values without crashing
    assert len(result) == 5
    assert isinstance(result, pd.Series)

# --- TESTS FOR _get_team_points ----------------------------------------------

@pytest.mark.parametrize("driver_id,expected_points", [
    ("D1", 150),  # TeamA - exists in constructor_standings
    ("D2", 120),  # TeamB - exists in constructor_standings
    ("D3", 90),   # TeamC - exists in constructor_standings
    ("D4", 0),    # Empty constructor_name
    ("D5", 0),    # NonExistentTeam - not in constructor_standings
    ("NONEXISTENT", 0),  # Driver not in driver_standings
])
def test_get_team_points(full_calculator, driver_id, expected_points):
    """Test the _get_team_points function for various scenarios."""
    pts = full_calculator._get_team_points(driver_id)
    assert pts == expected_points

# --- TESTS FOR _process_qualifying -------------------------------------------

def test_process_qualifying(full_calculator):
    """Test _process_qualifying processes all qualifying scenarios."""
    # Store original qualifying data for comparison
    original_q1 = full_calculator.qualifying['q1'].copy()

    full_calculator._process_qualifying()

    # Check that time conversion columns were added
    expected_columns = ['q1_ms', 'q2_ms', 'q3_ms', 'fastest_time_ms', 'normalized_fastest_qualifying']
    for col in expected_columns:
        assert col in full_calculator.qualifying.columns

    # Check that original data wasn't modified
    pd.testing.assert_series_equal(full_calculator.qualifying['q1'], original_q1)

    # Check specific time conversions
    # D1's q1: "1:20.500" -> 80500ms
    assert full_calculator.qualifying.loc[0, 'q1_ms'] == 80500

    # Check that missing/invalid times were set to inf
    d2_row = full_calculator.qualifying[full_calculator.qualifying['driverId'] == 'D2'].iloc[0]
    assert d2_row['q3_ms'] == float('inf')  # Empty string

    d3_row = full_calculator.qualifying[full_calculator.qualifying['driverId'] == 'D3'].iloc[0]
    assert d3_row['q2_ms'] == 4 * 60 * 1000  # "\\N"
    assert d3_row['q3_ms'] == 4 * 60 * 1000  # "bad:format"

    # Check that fastest times were calculated
    assert full_calculator.qualifying['fastest_time_ms'].notna().all()

    # Check that normalization was applied
    assert 0 <= full_calculator.qualifying['normalized_fastest_qualifying'].min() <= 1
    assert 0 <= full_calculator.qualifying['normalized_fastest_qualifying'].max() <= 1

    # Check position normalization
    assert 0 <= full_calculator.qualifying['position'].min()
    assert full_calculator.qualifying['position'].max() <= 1


def test_process_qualifying_with_empty_data(mock_calculator):
    """Test _process_qualifying with empty qualifying data."""
    mock_calculator._process_qualifying()  # Should not crash
    assert len(mock_calculator.qualifying) == 0
