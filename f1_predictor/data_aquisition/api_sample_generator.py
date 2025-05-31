import os
import pandas as pd
from datetime import timedelta
import os
import sys
from typing import Union

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the project root to sys.path
sys.path.append(project_root)

from f1_predictor.features.feature_calculator import CalculateSamplesRace


class SampleGenerator:
    def __init__(self, round: Union[int, str] = 1, lap: Union[int, str] = 1):
        self.round = round
        self.lap = lap
        self.results = pd.read_csv(os.path.join(os.path.dirname(__file__), '2025_data/results_with_elo_with_championships.csv'))
        self.laptimes = pd.read_csv(os.path.join(os.path.dirname(__file__), '2025_data/combined_laptimes.csv'))
        self.qualifying = pd.read_csv(os.path.join(os.path.dirname(__file__), '2025_data/combined_qualifyings.csv'))
        
        # Handle 'latest' round and lap
        if round == 'latest':
            self.round = self.results['RoundNumber'].max()
        else:
            self.round = round

        if lap == 'latest':
            self.lap = self.laptimes[self.laptimes['RoundNumber'] == self.round]['LapNumber'].max()
        else:
            self.lap = lap

    def simplify_time_string(self, time_str):
        #add a base case
        if pd.isna(time_str) or time_str == "0 days 00:00:00.000000":
            return "0:00.000"
        
        # Convert to timedelta
        days_part, time_part = time_str.split(" days ")
        days = int(days_part)
        hours, minutes, seconds = map(float, time_part.split(":"))
        
        # Compute total seconds
        total_seconds = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds).total_seconds()
        
        # Break into minutes and seconds
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        
        # Format: M:SS.mmm
        return f"{minutes}:{seconds:06.3f}"
    
    def _convert_laptimes_to_seconds(self, laptimes):
        """
        Convert lap times from timedelta to seconds.
        """
        if isinstance(laptimes, pd.Series):
            return laptimes.apply(lambda x: x.total_seconds() if isinstance(x, timedelta) else x)
        elif isinstance(laptimes, pd.DataFrame):
            for col in laptimes.select_dtypes(include=['timedelta64[ns]']).columns:
                laptimes[col] = laptimes[col].apply(lambda x: x.total_seconds() if isinstance(x, timedelta) else x)
            return laptimes
        else:
            raise ValueError("Input must be a pandas Series or DataFrame.")

    def generate_sample(self):
        race_result = self.results[self.results['RoundNumber'] == self.round].copy()
        race_qualifying = self.qualifying[self.qualifying['RoundNumber'] == self.round].copy()
        race_laptimes = self.laptimes[self.laptimes['RoundNumber'] == self.round].copy()

        # Check if the lap is valid
        if self.lap not in race_laptimes['LapNumber'].unique():
            raise ValueError(f"Lap {self.lap} is not valid for round {self.round}.")
        
        # Apply simplify_time_string to the qualifying times
        race_qualifying['Q1'] = race_qualifying['Q1'].apply(self.simplify_time_string)
        race_qualifying['Q2'] = race_qualifying['Q2'].apply(self.simplify_time_string)
        race_qualifying['Q3'] = race_qualifying['Q3'].apply(self.simplify_time_string)
        
        # First convert lap times to a simplified string format
        race_laptimes['LapTime'] = race_laptimes['LapTime'].apply(self.simplify_time_string)
        
        # Extract only the needed columns using proper DataFrame indexing
        laptimes = race_laptimes[['LapNumber', 'DriverNumber', 'LapTime']].copy()
        
        # Rename columns
        laptimes.rename(columns={
            'LapNumber': 'lap',
            'DriverNumber': 'driverId',
            'LapTime': 'milliseconds'
        }, inplace=True)
        
        # Convert the lap time strings to seconds
        laptimes['milliseconds'] = laptimes['milliseconds'].apply(
            lambda x: sum(float(part) * multiplier for part, multiplier in 
                        zip(x.replace(':', '.').split('.'), [60, 1, 0.001]))
            if isinstance(x, str) and x != "0:00.000" else 0
        )
        
        driver_standings = race_result[['DriverNumber', 'DriverPointsBefore', 'DriverWinsCount_before', 'elo_before', 'TeamName']].copy()
        driver_standings.rename(columns={
            'DriverNumber': 'driverId',
            'DriverPointsBefore': 'points',
            'DriverWinsCount_before': 'wins',
            'elo_before': 'elo',
            'TeamName': 'constructor_name'
        }, inplace=True)

        qualifying = race_qualifying[['DriverNumber', 'Position', 'Q1', 'Q2', 'Q3']].copy()
        qualifying.rename(columns={
            'DriverNumber': 'driverId',
            'Position': 'position',
            'Q1': 'q1',
            'Q2': 'q2',
            'Q3': 'q3'
        }, inplace=True)


        constructor_standings = race_result[['TeamName', 'TeamPointsBefore']].copy()
        constructor_standings.rename(columns={
            'TeamName': 'name',
            'TeamPointsBefore': 'points'
        }, inplace=True)
        self.sample_calculator = CalculateSamplesRace(
            race_id=self.round,
            laptimes=laptimes,
            results=None,
            driver_standings=driver_standings,
            qualifying=qualifying,
            constructor_standings=constructor_standings)
        return self.sample_calculator.get_samples_for_lap(self.lap)




