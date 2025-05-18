import os
import pandas as pd

class DatasetManager:
    def __init__(self):
        currentdir = os.path.dirname(os.path.abspath(__file__))
        parentdir = os.path.dirname(currentdir)
        self.data_folder = os.path.join(parentdir, 'data')
        self.required_columns = []

    def _check_required_columns(self, df):
        for column in self.required_columns:
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")
            
    def get_training_data(self):
        train_data_path = os.path.join(self.data_folder, 'train_data.csv')
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"Training data file not found: {train_data_path}")
        
        train_data = pd.read_csv(train_data_path)
        self._check_required_columns(train_data)
        return train_data

    def get_validation_data(self):
        val_data_path = os.path.join(self.data_folder, 'test_data.csv')
        if not os.path.exists(val_data_path):
            raise FileNotFoundError(f"Validation data file not found: {val_data_path}")
        
        val_data = pd.read_csv(val_data_path)
        self._check_required_columns(val_data)
        return val_data