import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import sys

currentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, currentdir)

races = pd.read_csv(os.path.join(currentdir, "data", "races.csv"))
lap_times = pd.read_csv(os.path.join(currentdir, "data", "lap_times.csv"))    
results = pd.read_csv(os.path.join(currentdir, "data", "results.csv"))
driver_standings = pd.read_csv(os.path.join(currentdir, "data", "driver_standings.csv"))

# Prepare a list for all observations; each observation is a dict of features and corresponding ground truth
all_samples = []

# Process each race in the dataset
for race_id in races['raceId']:
    # For each race, reset history for computing per-driver average lap time
    driver_history = {}
    driver_standings_race = driver_standings[driver_standings['raceId'] == race_id]
    # Since finishing positions come from the results file, get those now.
    results_race = results[results['raceId'] == race_id]
    finishing_positions = results_race[['driverId', 'position']].set_index('driverId').to_dict()['position']
    # In case a finishing position is missing, we default to 20.
    
    max_points_session = driver_standings_race['points'].max()
    if max_points_session == 0:
        max_points_session = 1

    # Get the race name and year (can be used later for plotting, if desired)
    race_info = races[races['raceId'] == race_id].iloc[0]
    
    # Select lap times for this race. (Make sure race_id matches exactly)
    laptimes_race = lap_times[lap_times['raceId'] == race_id]
    if laptimes_race.empty:
        print(f"No lap times available for race_id {race_id}. Skipping.")
        continue
    # Determine maximum number of laps in this race.
    amount_of_laps = laptimes_race['lap'].unique().max()
    
    # current_shortest is used to normalize lap times.
    current_shortest = float('inf')
    
    # Loop over laps in order
    for lap in sorted(laptimes_race['lap'].unique()):
        # Get all driver lap times for the current lap.
        driver_laptimes = laptimes_race[laptimes_race['lap'] == lap].copy()
        # Merge finishing positions to access current ranking if needed later.
        driver_laptimes = driver_laptimes.merge(results_race[['driverId', 'position']], on='driverId', how='left')
        
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


            if driver_standings_race[driver_standings_race['driverId'] == driver_id].empty:
                print(f"Driver {driver_id} not found in driver")
                normslized_driver_standing = 0
            else:
                normslized_driver_standing = float(driver_standings_race[driver_standings_race['driverId'] == driver_id]['points'].values[0] / max_points_session)
            sample = {
                "race_id": race_id,
                "driver_id": driver_id,
                "lap": lap,
                "normalized_lap": norm_lap,
                "average_normalized_lap": avg_norm,
                "lap_progress": row.lap_progress,
                "current_position_norm": current_rank_norm,
                "finishing_position": pos,
                "normalized_driver_standing": normslized_driver_standing,
            }
            all_samples.append(sample)
    
    # Optional: display race info (and delay if you want to mimic previous plot pauses)
    # plt.figure(figsize=(10, 6))
    # plt.title(f"Race: {race_info['name']}, Year: {race_info['year']}")
    # plt.show()
    # time.sleep(2)

# Convert to DataFrame
df_samples = pd.DataFrame(all_samples)

# Optionally, split data by race: use a race-level split (80% training, 20% test)
unique_races = df_samples['race_id'].unique()
train_races, test_races = train_test_split(unique_races, test_size=0.2, random_state=42)
train_data = df_samples[df_samples['race_id'].isin(train_races)]
test_data = df_samples[df_samples['race_id'].isin(test_races)]

print("Training samples:", len(train_data))
print("Test samples:", len(test_data))
print("\nA sample training observation:")
print(train_data.iloc[0])
train_data.to_csv("data/train_data.csv", index=False)
test_data.to_csv("data/test_data.csv", index=False)
print("CSV files saved: data/train_data.csv and data/test_data.csv")