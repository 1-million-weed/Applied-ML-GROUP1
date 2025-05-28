import os
import pandas as pd

class DatasetManager:
    """Manages loading and validation of training and validation datasets.

    Handles file paths, column presence checks, and basic format validation.
    """
    def __init__(self, training_features: list = []) -> None:
        """Constructor Method to tnitialize the dataset manager.

        Sets up the data folder path and expected required columns for validation.
        """ 
        currentdir = os.path.dirname(os.path.abspath(__file__))
        parentdir = os.path.dirname(currentdir)
        self.data_folder = os.path.join(parentdir, 'data')
        self.required_columns = training_features

    def _check_required_columns(self, df) -> None:
        """Check whether required columns are present in the provided DataFrame.


        :param df: DataFrame to validate.
        :type df: pd.DataFrame
        :raises ValueError: If any required column is missing.
        """    
        for column in self.required_columns:
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")
            
    def get_training_data(self) -> pd.DataFrame:
        """Load and return the training dataset.

        :raises FileNotFoundError: If the training CSV file does not exist.
        :return: Training data as a pandas DataFrame.
        :rtype: pd.DataFrame
        """    
        train_data_path = os.path.join(self.data_folder, 'train_data.csv')
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"Training data file not found: {train_data_path}")
        
        train_data = pd.read_csv(train_data_path)
        self._check_required_columns(train_data)
        return train_data

    def get_validation_data(self) -> pd.DataFrame:
        """Load and return the validation (test) dataset.

        :raises FileNotFoundError: If the test CSV file does not exist.
        :return: Validation data as a pandas DataFrame.
        :rtype: pd.DataFrame
        """    
        val_data_path = os.path.join(self.data_folder, 'test_data.csv')
        if not os.path.exists(val_data_path):
            raise FileNotFoundError(f"Validation data file not found: {val_data_path}")
        
        val_data = pd.read_csv(val_data_path)
        self._check_required_columns(val_data)
        return val_data
    
    def validate_data(self, data) -> bool:
        """Validate that the input data is a pandas DataFrames.

        :param data: Data to validate.
        :type data: pd.DataFrame
        :raises ValueError: If data is not a DataFrame or if required columns are missing.
        :return: True if data is valid.
        :rtype: bool
        """    
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        self._check_required_columns(data)
        return True