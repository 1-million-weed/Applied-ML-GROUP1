from data_folder_manager import DataFolderManager
from load_datasets import DatasetLoader
from feature_calculator import CalculateSamplesRace

import pandas as pd
from sklearn.model_selection import train_test_split



if __name__ == "__main__":
    #make sure data_folder is empty
    data_folder_manager = DataFolderManager(empty_folder=True)
    
    #load datasets
    datasets = DatasetLoader()
    all_samples = []
    for race_id in datasets.races['raceId']:
        driver_standings_race = datasets.driver_standings[datasets.driver_standings['raceId'] == race_id]
        results_race = datasets.results[datasets.results['raceId'] == race_id]
        laptimes_race = datasets.lap_times[datasets.lap_times['raceId'] == race_id]
        qualifying_race = datasets.qualifying[datasets.qualifying['raceId'] == race_id]
        if laptimes_race.empty:
            print(f"No lap times available for race_id {race_id}. Skipping.")
            continue
        feature_calculator = CalculateSamplesRace(race_id=race_id,
                                                laptimes=laptimes_race,
                                                results=results_race,
                                                driver_standings=driver_standings_race,
                                                qualifying=qualifying_race)
        all_samples += feature_calculator.calculate_samples()

    df_samples = pd.DataFrame(all_samples)
    unique_races = df_samples['race_id'].unique()
    train_races, test_races = train_test_split(unique_races, test_size=0.2, random_state=42)
    train_data = df_samples[df_samples['race_id'].isin(train_races)]
    test_data = df_samples[df_samples['race_id'].isin(test_races)]
    data_folder_manager.save_features(train_data, test_data)
    print("Training samples:", len(train_data))
    print("Test samples:", len(test_data))
    print("\nA sample training observation:")
    print(train_data.iloc[0])
    