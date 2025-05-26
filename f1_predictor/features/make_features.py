from .data_folder_manager import DataFolderManager
from .load_datasets import DatasetLoader
from .feature_calculator import CalculateSamplesRace

import pandas as pd
from sklearn.model_selection import train_test_split



class FeatureGenerator:
    """
    Generates training and testing features for Formula 1 race prediction.

    This class handles feature creation from the data set for each race,
    and splits data into train/test sets before saving.
    """
    def __init__(self,
                random_seed: int,
                test_size: float = 0.2,
                empty_folder: bool = True) -> None:
        
        self.test_size = test_size
        self.random_seed = random_seed
        self.empty_folder = empty_folder
        """
        Initialize the feature generator.

        :param random_seed: Random seed for reproducibility of train-test split.
        :type random_seed: int
        :param test_size: Fraction of races to be used as test data. Defaults to 0.2.
        :type test_size: float, optional
        :param empty_folder: Whether to clear the data folder before saving new features.
        :type empty_folder: bool, optional
        """
    def run(self) -> None:
        """
        Run the feature generation process.

        Loads datasets, calculates features for each race,
        splits data into training and testing sets, and saves them for each race.
        """    
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
    