import pandas as pd
import numpy as np

class CalculateSamplesRace:
    def __init__(self, race_id: str,
                  laptimes: pd.DataFrame,
                  results: pd.DataFrame,
                  driver_standings: pd.DataFrame,
                  qualifying: pd.DataFrame):
        
        self.race_id = race_id
        self.laptimes = laptimes
        self.results = results
        self.driver_standings = driver_standings
        # Create a copy to avoid SettingWithCopyWarning
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
            return milliseconds
        except ValueError:
            return 4 * 60 * 1000  # Handle unexpected invalid formats
        
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
        
        # Ensure 'position' column is numeric before normalizing
        self.qualifying['position'] = pd.to_numeric(self.qualifying['position'], errors='coerce')
        max_position = self.qualifying['position'].max()
        if max_position > 0:  # Avoid division by zero
            self.qualifying['position'] = self.qualifying['position'] / max_position
        else:
            self.qualifying['position'] = 0
        

    def calculate_samples(self) -> list:
        """Calculate samples for race prediction, with proper data handling."""
        self._process_qualifying()
        samples = []
        driver_history = {}
        
        # Handle case when max_points_session is 0
        max_points_session = self.driver_standings['points'].max()
        if max_points_session == 0:
            max_points_session = 1
        
        finishing_positions = self.results[['driverId', 'position']].set_index('driverId').to_dict()['position']
        current_shortest = float('inf')
        amount_of_laps = self.laptimes['lap'].unique().max()    
        
        for lap in sorted(self.laptimes['lap'].unique()):
            driver_laptimes = self.laptimes[self.laptimes['lap'] == lap].copy()
            driver_laptimes = driver_laptimes.merge(self.results[['driverId', 'position']], on='driverId', how='left')
            
            # Update the current shortest lap time
            lap_min = driver_laptimes['milliseconds'].min()
            if lap_min < current_shortest:
                current_shortest = lap_min
                
            # Normalize lap times and normalize lap progress (current lap / max laps)
            driver_laptimes['milliseconds'] = driver_laptimes['milliseconds'] / current_shortest
            driver_laptimes['lap_progress'] = lap / amount_of_laps
            
            # To get the ranking for this lap based on lap performance, sort by normalized lap time.
            driver_laptimes = driver_laptimes.sort_values('milliseconds').reset_index(drop=True)
            total_drivers = len(driver_laptimes)
            
            # For each driver in this lap, compute features and store an observation.
            for rank, row in enumerate(driver_laptimes.itertuples(), start=1):
                driver_id = row.driverId
                norm_lap = row.milliseconds  # normalized lap time for current lap
                
                # Update running history of normalized lap times per driver
                if driver_id not in driver_history:
                    driver_history[driver_id] = []
                driver_history[driver_id].append(norm_lap)
                avg_norm = sum(driver_history[driver_id]) / len(driver_history[driver_id])
                # Current lap ranking normalized: lower is better.
                current_rank_norm = rank / total_drivers
                
                # Use finishing position as ground truth; if not found, default to 20.
                pos = finishing_positions.get(driver_id, 20)
                try:
                    pos = int(pos)
                except:
                    pos = 20

                # Get driver standings data safely
                driver_standings_rows = self.driver_standings[self.driver_standings['driverId'] == driver_id]
                if driver_standings_rows.empty:
                    print(f"Driver {driver_id} not found in driver standings")
                    normalized_driver_standing = 0
                else:
                    normalized_driver_standing = float(driver_standings_rows['points'].values[0] / max_points_session)

                # Get qualifying data safely
                qualifying_rows = self.qualifying[self.qualifying['driverId'] == driver_id]
                if qualifying_rows.empty:
                    normalized_fastest_qualifying = 1.0  # Default value
                    position_quali = 1.0  # Default value
                else:
                    normalized_fastest_qualifying = qualifying_rows['normalized_fastest_qualifying'].values[0]
                    position_quali = qualifying_rows['position'].values[0]
                
                sample = {
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
                }
                samples.append(sample)
        return samples