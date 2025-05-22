import pandas as pd

class CalculateSamplesRace:
    def __init__(self, race_id: str,
                 laptimes: pd.DataFrame,
                 results: pd.DataFrame,
                 driver_standings: pd.DataFrame,
                 qualifying: pd.DataFrame,
                 constructor_standings: pd.DataFrame):
        self.race_id = race_id
        self.laptimes = laptimes
        self.results = results
        self.driver_standings = driver_standings
        self.constructor_standings = constructor_standings
        self.qualifying = qualifying.copy()

    def _convert_time_to_milliseconds(self, time_str: str) -> int:
        """Convert a time like '1:27.236' to milliseconds."""
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
        """Get the number of wins for a driver."""
        return self.driver_standings[self.driver_standings['driverId'] == driver_id]['wins'].values[0]

    def _min_max_normalize(self, series: pd.Series) -> pd.Series:
        """Apply min-max normalization to a pandas Series."""
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val == 0:
            return series
        return (series - min_val) / (max_val - min_val)

    def _process_qualifying(self):
        """Process qualifying data, handling time conversions and normalization properly."""
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

    def _normalize_driver_standings(self):
        """Normalize driver standings data."""
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
        """Retrieve finishing positions from results."""
        return self.results[['driverId', 'position']].set_index('driverId').to_dict()['position']

    def _process_lap_data(self, lap, current_shortest, amount_of_laps, finishing_positions, driver_history):
        """Process data for a single lap."""
        driver_laptimes = self.laptimes[self.laptimes['lap'] == lap].copy()
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
        """Get the points for the team of a given driver."""
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

    def _create_sample(self, row, lap, driver_history, finishing_positions, total_drivers):
        """Create a sample for a single driver."""
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

    def calculate_samples(self) -> list:
        """Calculate samples for race prediction."""
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