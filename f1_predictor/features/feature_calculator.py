import pandas as pd
from typing import Union

class CalculateSamplesRace:
    """
    A class to process and calculate per-lap race samples from F1 datasets.

    This includes normalization of lap times, qualifying data, and driver standings,
    along with generating samples for training.
    """
    def __init__(self, race_id: str,
                 laptimes: pd.DataFrame,
                 results,
                 driver_standings: pd.DataFrame,
                 qualifying: pd.DataFrame,
                 constructor_standings: pd.DataFrame) -> None:
        """
        Constructor method to initialize the CalculateSamplesRace object.

        :param race_id: Unique identifier of the race.
        :type race_id: str
        :param laptimes: DataFrame containing lap-by-lap time data.
        :type laptimes: pd.DataFrame
        :param results: DataFrame with race results. 
        :type results: pd.DataFrame
        :param driver_standings: DataFrame with driver standings.
        :type driver_standings: pd.DataFrame
        :param qualifying: DataFrame with qualifying session data.
        :type qualifying: pd.DataFrame
        :param constructor_standings: DataFrame with constructor team standings.
        :type constructor_standings: pd.DataFrame
        """    
        self.race_id = race_id
        self.laptimes = laptimes
        self.results = results
        self.driver_standings = driver_standings
        self.constructor_standings = constructor_standings
        self.qualifying = qualifying.copy()

    def _convert_time_to_milliseconds(self, time_str: str) -> int:
        """
        Convert a time like '1:27.236' to milliseconds.

        :param time_str: String representing a lap or qualifying time.
        :type time_str: str
        :return: Time in milliseconds.
        :rtype: int
        """
        if pd.isna(time_str) or time_str == '' or time_str == '\\N':
            return 4 * 60 * 1000   # Set to a very high value if the time is invalid
        try:
            if ':' in time_str:
                minutes, seconds = time_str.split(':')
                milliseconds = int(minutes) * 60 * 1000 + float(seconds) * 1000
            else:
                milliseconds = float(time_str) * 1000
            return int(milliseconds) # Ensure it's an integer
        except ValueError:
            return 4 * 60 * 1000  # Handle unexpected invalid formats
        
    def _get_amount_of_wins(self, driver_id: str) -> int:
        """
        Get the number of wins for a given driver.

        :param driver_id: Unique driver ID.
        :type driver_id: str
        :return: Number of wins.
        :rtype: int
        """
        return self.driver_standings[self.driver_standings['driverId'] == driver_id]['wins'].values[0]     

    def _min_max_normalize(self, series: pd.Series) -> pd.Series:
        """
        Apply min-max normalization to a pandas Series.

        :param series: The pandas Series to normalize.
        :type series: pd.Series
        :return: Normalized Series scaled between 0 and 1.
        :rtype: pd.Series
        """
        # Filter out NaN values
        series = series.dropna()
        if series.empty:  # Return a series of zeros if it's empty after dropping NaN
            return pd.Series([0] * len(series), index=series.index)
        min_val = series.min()
        max_val = series.max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val - min_val == 0:
            return pd.Series([0] * len(series), index=series.index)  # Return zeros if min or max is NaN or range is zero
        return (series - min_val) / (max_val - min_val)

    def _process_qualifying(self) -> None:
        """
        Process qualifying data by converting time strings to milliseconds,
        determining the fastest lap, and normalizing both time and position data.
        """
        # Convert each qualifying time column separately
        for col in ['q1', 'q2', 'q3']:
            self.qualifying[f'{col}_ms'] = self.qualifying[col].apply(
                lambda x: self._convert_time_to_milliseconds(x) if pd.notna(x) else float('inf')
            )

        # Find the fastest time for each driver across all qualifying sessions
        self.qualifying['fastest_time_ms'] = self.qualifying[['q1_ms', 'q2_ms', 'q3_ms']].min(axis=1)

        # Find the overall fastest qualifying time
        overall_fastest_time = self.qualifying['fastest_time_ms'].min()

        # Normalize each driver's fastest qualifying time by the overall fastest time
        self.qualifying['normalized_fastest_qualifying'] = (
            self.qualifying['fastest_time_ms'] / overall_fastest_time
        )
        self.qualifying['normalized_fastest_qualifying'] = self._min_max_normalize(self.qualifying['normalized_fastest_qualifying'])

        # Ensure 'position' column is numeric before normalizing
        self.qualifying['position'] = pd.to_numeric(self.qualifying['position'], errors='coerce')
        max_position = self.qualifying['position'].max()
        if max_position > 0:  # Avoid division by zero
            self.qualifying['position'] = self.qualifying['position'] / max_position
        else:
            self.qualifying['position'] = 0

    def _normalize_driver_standings(self) -> None:
        """
        Normalize driver standings and constructor standings using min-max scaling
        for points, ELO ratings, and win counts.
        """
        max_points_session = self.driver_standings['points'].max()
        if max_points_session == 0:
            max_points_session = 1
        self.driver_standings['points'] = self.driver_standings['points'] / max_points_session
        self.driver_standings['points'] = self._min_max_normalize(self.driver_standings['points'])

        self.constructor_standings['points'] = self._min_max_normalize(self.constructor_standings['points'])

        max_elo_session = self.driver_standings['elo'].max()
        if max_elo_session == 0:
            max_elo_session = 1
        self.driver_standings['normalized_elo'] = self.driver_standings['elo'] / max_elo_session
        self.driver_standings['normalized_elo'] = self._min_max_normalize(self.driver_standings['normalized_elo'])

        self.driver_standings['wins'] = self._min_max_normalize(self.driver_standings['wins'])

    def _get_finishing_positions(self):
        """
        Retrieve finishing positions for each driver.

        :return: A dictionary mapping driverId to finishing position.
        :rtype: dict[str, int]
        """
        if self.results is None:
            return {}
        return self.results[['driverId', 'position']].set_index('driverId').to_dict()['position']

    def _process_lap_data(self, lap, current_shortest, amount_of_laps, finishing_positions, driver_history) -> tuple[pd.DataFrame, float]:
        """
        Process data for a single lap by normalizing lap times and computing lap progress.

        :param lap: Current lap number.
        :type lap: int
        :param current_shortest: Current shortest lap time encountered so far.
        :type current_shortest: float
        :param amount_of_laps: Total number of laps in the race.
        :type amount_of_laps: int
        :param finishing_positions: Dictionary mapping driverId to final positions.
        :type finishing_positions: dict[str, int]
        :param driver_history: Historical normalized lap times per driver.
        :type driver_history: dict[str, list[float]]
        :return: Tuple of processed lap DataFrame and updated shortest time.
        :rtype: tuple[pd.DataFrame, float]
        """
        driver_laptimes = self.laptimes[self.laptimes['lap'] == lap].copy()
        
        # Only merge with results if they are available
        if self.results is not None:
            driver_laptimes = driver_laptimes.merge(self.results[['driverId', 'position']], on='driverId', how='left')

        lap_min = driver_laptimes['milliseconds'].min()
        if lap_min < current_shortest:
            current_shortest = lap_min

        driver_laptimes['milliseconds'] = driver_laptimes['milliseconds'] / current_shortest
        driver_laptimes['milliseconds'] = self._min_max_normalize(driver_laptimes['milliseconds'])
        driver_laptimes['lap_progress'] = lap / amount_of_laps

        driver_laptimes = driver_laptimes.sort_values('milliseconds').reset_index(drop=True)
        return driver_laptimes, current_shortest

    def _get_team_points(self, driver_id: str) -> int:
        """
        Get the points for the team of a given driver.

        :param driver_id: identifier for the driver
        :type driver_id: str
        :return: Points scored by the  driver's team, or 0 if not found.
        :rtype: int
        """
        driver_standings_row = self.driver_standings[self.driver_standings['driverId'] == driver_id]
        if not driver_standings_row.empty:
            team_name_driver = driver_standings_row['constructor_name'].values[0]
        else:
            team_name_driver = ''

        if team_name_driver == '':
            return 0

        points_team = self.constructor_standings[self.constructor_standings['name'] == team_name_driver]['points']
        if points_team.empty:
            return 0
        return points_team.values[0]

    def _create_sample(self, row, lap, driver_history, finishing_positions, total_drivers) -> Union[dict[str, float], None]:
        """
        Create a single training sample for a driver based on their lap performance, 
        historical performance, and various race features.

        :param row: A row from the lap time dataframe containing driver performance metrics.
        :type row: pd.Series
        :param lap: The current lap number.
        :type lap: int
        :param driver_history: Dictionary storing each driver's history of normalized lap times.
        :type driver_history: dict[str, list[float]]
        :param finishing_positions: Mapping from driver ID to their final race position.
        :type finishing_positions: dict[str, int]
        :param total_drivers: The total number of drivers in the current lap.
        :type total_drivers: int
        :return: A dictionary representing a training sample with engineered features for a single driver, or None if the position is invalid.
        :rtype: dict[str, float] | None
        """
        driver_id = row.driverId
        norm_lap = row.milliseconds

        if driver_id not in driver_history:
            driver_history[driver_id] = []
        driver_history[driver_id].append(norm_lap)
        avg_norm = sum(driver_history[driver_id]) / len(driver_history[driver_id])
        current_rank_norm = row.Index / total_drivers

        points_team = self._get_team_points(driver_id)

        pos = finishing_positions.get(driver_id, 20)
        try:
            pos = int(pos)
            if pos > 20:
                pos = 20
        except:
            return None

        driver_standings_row = self.driver_standings[self.driver_standings['driverId'] == driver_id]
        normalized_driver_standing = driver_standings_row['points'].values[0] if not driver_standings_row.empty else 0

        qualifying_rows = self.qualifying[self.qualifying['driverId'] == driver_id]
        if qualifying_rows.empty:
            normalized_fastest_qualifying = 1.0
            position_quali = 1.0
        else:
            normalized_fastest_qualifying = qualifying_rows['normalized_fastest_qualifying'].values[0]
            position_quali = qualifying_rows['position'].values[0]

        normalized_driver_elo = self.driver_standings[self.driver_standings['driverId'] == driver_id]['normalized_elo']
        normalized_driver_elo = normalized_driver_elo.values[0] if not normalized_driver_elo.empty else 0

        amount_of_wins = self._get_amount_of_wins(driver_id)


        return {
            "race_id": self.race_id,
            "driver_id": driver_id,
            "lap": lap,
            "normalized_lap": norm_lap,
            "average_normalized_lap": avg_norm,
            "lap_progress": row.lap_progress,
            "current_position_norm": current_rank_norm,
            "finishing_position": pos,
            "normalized_driver_standing": normalized_driver_standing,
            "normalized_fastest_qualifying": normalized_fastest_qualifying,
            "position_quali": position_quali,
            "normalized_driver_elo": normalized_driver_elo,
            "amount_of_wins": amount_of_wins,
            'points_team': points_team,
        }
    
    def _create_sample_without_finishing_position(self, row, lap, driver_history, total_drivers) -> Union[dict[str, float], None]:
        """
        Create a single training sample for a driver without including the finishing position.
        Used for making predictions for a specific lap.

        :param row: A row from the lap time dataframe containing driver performance metrics.
        :type row: pd.Series
        :param lap: The current lap number.
        :type lap: int
        :param driver_history: Dictionary storing each driver's history of normalized lap times.
        :type driver_history: dict[str, list[float]]
        :param total_drivers: The total number of drivers in the current lap.
        :type total_drivers: int
        :return: A dictionary representing a training sample with engineered features for a single driver.
        :rtype: dict[str, float] | None
        """
        driver_id = row.driverId
        norm_lap = row.milliseconds

        if driver_id not in driver_history:
            driver_history[driver_id] = []
        driver_history[driver_id].append(norm_lap)
        avg_norm = sum(driver_history[driver_id]) / len(driver_history[driver_id])
        current_rank_norm = row.Index / total_drivers

        points_team = self._get_team_points(driver_id)

        driver_standings_row = self.driver_standings[self.driver_standings['driverId'] == driver_id]
        normalized_driver_standing = driver_standings_row['points'].values[0] if not driver_standings_row.empty else 0

        qualifying_rows = self.qualifying[self.qualifying['driverId'] == driver_id]
        if qualifying_rows.empty:
            normalized_fastest_qualifying = 1.0
            position_quali = 1.0
        else:
            normalized_fastest_qualifying = qualifying_rows['normalized_fastest_qualifying'].values[0]
            position_quali = qualifying_rows['position'].values[0]

        normalized_driver_elo = self.driver_standings[self.driver_standings['driverId'] == driver_id]['normalized_elo']
        normalized_driver_elo = normalized_driver_elo.values[0] if not normalized_driver_elo.empty else 0

        amount_of_wins = self._get_amount_of_wins(driver_id)

        return {
            "race_id": self.race_id,
            "driver_id": driver_id,
            "lap": lap,
            "normalized_lap": norm_lap,
            "average_normalized_lap": avg_norm,
            "lap_progress": row.lap_progress,
            "current_position_norm": current_rank_norm,
            "normalized_driver_standing": normalized_driver_standing,
            "normalized_fastest_qualifying": normalized_fastest_qualifying,
            "position_quali": position_quali,
            "normalized_driver_elo": normalized_driver_elo,
            "amount_of_wins": amount_of_wins,
            'points_team': points_team,
        }
    
    def get_samples_for_lap(self, target_lap: int) -> list[dict[str, float]]:
        """
        Calculate samples for a specific lap number for all drivers without including
        the finishing position (for making predictions).
        
        :param target_lap: The lap number to get samples for
        :type target_lap: int
        :return: A list of dictionaries representing samples for all drivers at the specified lap
        :rtype: list[dict[str, float]]
        """
        # Check if the target lap exists in the data
        if target_lap not in self.laptimes['lap'].unique():
            return []
            
        self._process_qualifying()
        self._normalize_driver_standings()
        samples = []
        driver_history = {}
        
        # Get finishing positions if results are available
        finishing_positions = self._get_finishing_positions()  # Will return empty dict if self.results is None

        current_shortest = float('inf')
        amount_of_laps = self.laptimes['lap'].unique().max()
        
        # Process laps up to and including the target lap to build driver history
        for lap in sorted(self.laptimes['lap'].unique()):
            if lap > target_lap:
                break
                
            driver_laptimes, current_shortest = self._process_lap_data(
                lap, current_shortest, amount_of_laps, finishing_positions, driver_history
            )
            total_drivers = len(driver_laptimes)
            
            # Only add samples for the target lap, using the method that excludes finishing position
            if lap == target_lap:
                for row in driver_laptimes.itertuples():
                    sample = self._create_sample_without_finishing_position(row, lap, driver_history, total_drivers)
                    if sample:
                        samples.append(sample)
                        
        return samples

    def calculate_samples(self) -> list[dict[str, float]]:
        """
        Calculate a list of normalized samples for a given race.

        :return: A list of dictionaries representing samples with race features.
        :rtype: list[dict[str, float]]
        """
        self._process_qualifying()
        self._normalize_driver_standings()
        samples = []
        driver_history = {}
        finishing_positions = self._get_finishing_positions()

        current_shortest = float('inf')
        amount_of_laps = self.laptimes['lap'].unique().max()

        for lap in sorted(self.laptimes['lap'].unique()):
            driver_laptimes, current_shortest = self._process_lap_data(
                lap, current_shortest, amount_of_laps, finishing_positions, driver_history
            )
            total_drivers = len(driver_laptimes)

            for row in driver_laptimes.itertuples():
                sample = self._create_sample(row, lap, driver_history, finishing_positions, total_drivers)
                if sample:
                    samples.append(sample)

        return samples

