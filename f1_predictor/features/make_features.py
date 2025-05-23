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
        unique_years = datasets.races['year'].unique()
        train_years, test_years = train_test_split(unique_years, test_size=self.test_size, random_state=self.random_seed)
        train_race_ids = datasets.races[datasets.races['year'].isin(train_years)]['raceId']
        test_race_ids = datasets.races[datasets.races['year'].isin(test_years)]['raceId']
        
        train_samples = []
        for race_id in train_race_ids:
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
            train_samples += feature_calculator.calculate_samples()
        
        test_samples = []
        for race_id in test_race_ids:
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
            test_samples += feature_calculator.calculate_samples()
        
        df_train = pd.DataFrame(train_samples)
        df_test = pd.DataFrame(test_samples)
        # Optional: add year column if needed
        year_map = dict(zip(datasets.races['raceId'], datasets.races['year']))
        df_train['year'] = df_train['race_id'].map(year_map)
        df_test['year'] = df_test['race_id'].map(year_map)
        
        data_folder_manager.save_features(df_train, df_test)
        print("Training samples:", len(df_train))
        print("Test samples:", len(df_test))
        print("\nA sample training observation:")
        print(df_train.iloc[0])
