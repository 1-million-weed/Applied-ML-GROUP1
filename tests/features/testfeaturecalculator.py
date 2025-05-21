
import math
import pandas as pd
import pytest
from f1_predictor.features.feature_calculator import CalculateSamplesRace

@pytest.fixture
def mock_calculator():
    # dataframe to satisfy __init__
    empty = pd.DataFrame()
    return CalculateSamplesRace(
        race_id="TEST_RACE",  # 1) race_id
        laptimes=empty,  # 2) laptimes
        results=empty,  # 3) results
        driver_standings=empty,  # 4) driver_standings
        qualifying=empty,  # 5) qualifying
        constructor_standings=empty  # 6) constructor_standings
    )

@pytest.mark.parametrize("time_str,expected_ms", [
    ("1:27.236",    1*60*1000 + 27.236*1000), # Minutes and seconds
    ("87.236",      87.236*1000),   # just seconds
    ("",            4*60*1000),     # blank inputs returns high values
    ("\\N",         4*60*1000),     # NaN inputs return high values
    (None,          4*60*1000),     # None inputs return high values
    ("bad:format",  4*60*1000),     # incorrect inputs returns high values
])
def test_convert_time(mock_calculator, time_str, expected_ms):
    ms = mock_calculator._convert_time_to_milliseconds(time_str)
    assert ms == expected_ms