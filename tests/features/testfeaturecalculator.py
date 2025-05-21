
import math
import pandas as pd
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
def team_calculator():
    """Calculator preloaded with driver_standings and constructor_standings."""
    empty = pd.DataFrame()
    driver_standings = pd.DataFrame([
        {"driverId": "D1", "constructor_name": "Alpha"},
        {"driverId": "D2", "constructor_name": "Beta"},
    ])
    constructor_standings = pd.DataFrame([
        {"name": "Alpha", "points": 123},
    ])
    return CalculateSamplesRace(
        race_id="R1",
        laptimes=empty,
        results=empty,
        driver_standings=driver_standings,
        qualifying=empty,
        constructor_standings=constructor_standings,
    )

# --- TESTS FOR _convert_time_to_milliseconds ---------------------------------

@pytest.mark.parametrize("time_str,expected_ms", [
    ("1:27.236",    1*60*1000 + 27.236*1000), # Minutes and seconds
    ("87.236",      87.236*1000),   # just seconds
    ("",            4*60*1000),     # blank inputs returns 4 minutes
    ("\\N",         4*60*1000),     # NaN inputs return 4 minutes
    (None,          4*60*1000),     # None inputs return 4 minutes
    ("bad:format",  4*60*1000),     # incorrect inputs returns 4 minutes
])
def test_convert_time(mock_calculator, time_str, expected_ms):
    ms = mock_calculator._convert_time_to_milliseconds(time_str)
    assert ms == expected_ms


# --- TESTS FOR _get_team_points ----------------------------------------------

@pytest.mark.parametrize("driver_id,expected", [
    ("D1", 123),  # in driver_standings & constructor_standings
    ("D2",   0),  # in driver_standings but no constructor row
    ("D3",   0),  # not in driver_standings at all
])
def test_get_team_points(team_calculator, driver_id, expected):
    pts = team_calculator._get_team_points(driver_id)
    assert pts == expected

