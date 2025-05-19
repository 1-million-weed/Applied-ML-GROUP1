from .data_folder_manager import DataFolderManager
from .load_datasets import DatasetLoader
from .feature_calculator import CalculateSamplesRace

import pandas as pd
from sklearn.model_selection import train_test_split



class FeatureGenerator:
    def __init__(self,
                random_seed: int,
                test_size: float = 0.2,
                empty_folder: bool = True):
        
        self.test_size = test_size
        self.random_seed = random_seed
        self.empty_folder = empty_folder

    def run(self):
        #make sure data_folder is empty
        data_folder_manager = DataFolderManager(empty_folder=self.empty_folder)
        
        #load datasets
        datasets = DatasetLoader()
        all_samples = []
        for race_id in datasets.races['raceId']:
        #for race_id in range(1031, 1144):
            driver_standings_race = datasets.driver_standings[datasets.driver_standings['raceId'] == race_id]
            results_race = datasets.results[datasets.results['raceId'] == race_id]
            laptimes_race = datasets.lap_times[datasets.lap_times['raceId'] == race_id]
            qualifying_race = datasets.qualifying[datasets.qualifying['raceId'] == race_id]
            constructor_standings_race = datasets.constructor_standings[datasets.constructor_standings['raceId'] == race_id]
            if laptimes_race.empty:
                print(f"No lap times available for race_id {race_id}. Skipping.")
                continue
            feature_calculator = CalculateSamplesRace(race_id=race_id,
                                                    laptimes=laptimes_race,
                                                    results=results_race,
                                                    driver_standings=driver_standings_race,
                                                    qualifying=qualifying_race,
                                                    constructor_standings=constructor_standings_race)
            all_samples += feature_calculator.calculate_samples()

        df_samples = pd.DataFrame(all_samples)
        unique_races = df_samples['race_id'].unique()
        train_races, test_races = train_test_split(unique_races, test_size=self.test_size, random_state=self.random_seed)
        train_data = df_samples[df_samples['race_id'].isin(train_races)]
        test_data = df_samples[df_samples['race_id'].isin(test_races)]
        data_folder_manager.save_features(train_data, test_data)
        print("Training samples:", len(train_data))
        print("Test samples:", len(test_data))
        print("\nA sample training observation:")
        print(train_data.iloc[0])
    