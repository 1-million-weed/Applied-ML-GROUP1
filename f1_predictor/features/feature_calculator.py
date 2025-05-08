import pandas as pd

class CalculateSamplesRace:
    def __init__(self, race_id: str,
                  laptimes: pd.DataFrame,
                  results: pd.DataFrame,
                  driver_standings: pd.DataFrame):
        
        self.race_id = race_id
        self.laptimes = laptimes
        self.results = results
        self.driver_standings = driver_standings

    def calculate_samples(self) -> list:
        samples = []
        driver_history = {}
        max_points_session = self.driver_standings['points'].max()
        finishing_positions = self.results[['driverId', 'position']].set_index('driverId').to_dict()['position']
        current_shortest = float('inf')
        if max_points_session == 0:
            max_points_session = 1
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

                if self.driver_standings[self.driver_standings['driverId'] == driver_id].empty:
                    print(f"Driver {driver_id} not found in driver")
                    normslized_driver_standing = 0
                else:
                    normslized_driver_standing = float(self.driver_standings[self.driver_standings['driverId'] == driver_id]['points'].values[0] / max_points_session)
                sample = {
                    "race_id": self.race_id,
                    "driver_id": driver_id,
                    "lap": lap,
                    "normalized_lap": norm_lap,
                    "average_normalized_lap": avg_norm,
                    "lap_progress": row.lap_progress,
                    "current_position_norm": current_rank_norm,
                    "finishing_position": pos,
                    "normalized_driver_standing": normslized_driver_standing,
                }
                samples.append(sample)
        return samples

