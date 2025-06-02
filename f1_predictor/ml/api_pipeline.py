import os 
import sys
import numpy as np
import pandas as pd
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from f1_predictor.data_aquisition.api_sample_generator import SampleGenerator
from f1_predictor.features.data_folder_manager import DataFolderManager


class APIpipeline:
    def __init__(self, model, round: int = 1, lap: int = 1):

        self.logger = logging.getLogger(__name__)

        self.round = round
        self.lap = lap
        self.model = model
        self.sample_generator = SampleGenerator(round=self.round, lap=self.lap)
        data_folder_manager = DataFolderManager(empty_folder=False, data_folder_path='data_aquisition/2025_data')

        self.logger.info(f"Initializing API pipeline with round: {self.round}, lap: {self.lap}, model: {self.model}")

        if not data_folder_manager.available_2025_data():
            raise FileNotFoundError("2025 data is not available. Please run the data acquisition pipeline first.")

    def _generate_sample(self):
        return self.sample_generator.generate_sample()
        
    def run(self):
        samples = self._generate_sample()
        return self._predict_multiple(samples)
        
    def _predict_multiple(self, observations: np.ndarray, return_zero_indexed = False) -> dict:
        predictions = {}
        for sample in observations:
            sample = {k: (v if not isinstance(v, float) or not np.isnan(v) else 0) for k, v in sample.items()}
            input_data = {
                'normalized_lap': sample['normalized_lap'],
                'average_normalized_lap': sample['average_normalized_lap'],
                'lap_progress': sample['lap_progress'],
                'current_position_norm': sample['current_position_norm'],
                'normalized_driver_standing': sample['normalized_driver_standing'],
                'normalized_fastest_qualifying': sample['normalized_fastest_qualifying'],
                'position_quali': sample['position_quali'],
                'normalized_driver_elo': sample['normalized_driver_elo'],
                'amount_of_wins': sample['amount_of_wins'],
                'points_team': sample['points_team']
            }
            # Ensure no NaN values in the dictionary
            input_data = {k: (v if not isinstance(v, float) or not np.isnan(v) else 0) for k, v in input_data.items()}
            # Wrap each value in a list to properly create DataFrame with scalar values
            input_data = {k: [v] for k, v in input_data.items()}
            input_data = pd.DataFrame.from_dict(input_data, orient='columns')
            self.logger.info(f"Input data for prediction", extra={"input_data": input_data.to_dict(orient='records')})
            prediction = self.model.predict(input_data, round=False)
            self.logger.info(f"Raw prediction: {prediction}, type: {type(prediction)}")
            
            # Extract the prediction value regardless of structure
            if isinstance(prediction, (list, tuple, np.ndarray)):
                if len(prediction) > 0:
                    if isinstance(prediction[0], (list, tuple, np.ndarray)):
                        pred_value = prediction[0][0]
                    else:
                        pred_value = prediction[0]
                else:
                    pred_value = 0
            else:
                pred_value = prediction  # It's already a scalar
                
            predictions[sample['driver_id']] = pred_value

        self.logger.info("Predictions before sorting", extra={"unsorted_predictions": predictions})
        # Sort by prediction values and create a new dict
        sorted_predictions = sorted(predictions.items(), key=lambda item: item[1])
        self.logger.info("Sorted predictions", extra={"sorted_predictions": sorted_predictions})

        if not return_zero_indexed:
            predictions = {k: v + 1 for k, v in sorted_predictions}
        else:
            predictions = {k: v for k, v in sorted_predictions}
            
        return predictions
    


class CurrentYearInfo:
    """
    Class to hold information about the current year, round, and lap.
    This is useful for API endpoints that need
    to provide information about the
    current

    season.
    """
    def __init__(self, year :int = 2025):
        data_folder_manager = DataFolderManager(empty_folder=False, data_folder_path='data_aquisition/2025_data')
        if not data_folder_manager.available_2025_data():
            raise FileNotFoundError("2025 data is not available. Please run the data acquisition pipeline first.")
        self.year = year
        datafolder_path = os.path.join(project_root, 'f1_predictor', 'data_aquisition', '2025_data')
        self.results = pd.read_csv(os.path.join(datafolder_path, 'results.csv'))
        self.laptimes = pd.read_csv(os.path.join(datafolder_path, 'combined_laptimes.csv'))

    def list_rounds(self) -> dict:
        """
        List all rounds for the current year.
        
        Returns:
            A dictionary mapping event names to round numbers.
        """
        round_numbers = self.results['RoundNumber'].unique().tolist()
        races = {}
        for round_number in round_numbers:
            filtered = self.results[self.results['RoundNumber'] == round_number]
            if not filtered.empty:
                event_name = filtered.iloc[0]['EventName']
                races[event_name] = round_number
        return races
        
    def get_max_laps_round(self, round_number: int) -> int:
        """
        Get the maximum number of laps for a given round.
        
        Args:
            round_number (int): The round number to check.
        
        Returns:
            int: The maximum number of laps for the specified round.
        """
        filtered = self.laptimes[self.laptimes['RoundNumber'] == round_number]
        if not filtered.empty:
            return filtered['LapNumber'].max()
        else:
            raise ValueError(f"No data found for round {round_number}.")

