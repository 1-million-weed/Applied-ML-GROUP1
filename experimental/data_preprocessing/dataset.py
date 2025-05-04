import os

import pandas as pd


class Dataset:
    def __init__(self, name: str):
        current_dir = os.path.dirname(__file__)
        data_path = os.path.join(current_dir, "data", name)
        self.path = data_path
        self.data = self._load_dataset()

    def _load_dataset(self) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset not found at {self.path}")
        return pd.read_csv(self.path)

    def print_columns(self) -> None:
        if not hasattr(self, 'data'):
            raise AttributeError("Data not loaded. Call load() method first.")
        print(self.data.columns)

    def get_data(self, column_name):
        if not hasattr(self, 'data'):
            raise AttributeError("Data not loaded. Call load() method first.")
        if column_name not in self.data.columns:
            raise ValueError(f"Column '{column_name}' not found in dataset.")
        return self.data[column_name]

    def print_amount_of_rows(self) -> None:
        if not hasattr(self, 'data'):
            raise AttributeError("Data not loaded. Call load() method first.")
        print(f"Number of rows: {len(self.data)}")

    def get_amount_of_rows(self) -> int:
        if not hasattr(self, 'data'):
            raise AttributeError("Data not loaded. Call load() method first.")
        return len(self.data)

    def get_data_by_id(self, id: int) -> pd.Series:
        if not hasattr(self, 'data'):
            raise AttributeError("Data not loaded. Call load() method first.")
        if id < 1 or id > len(self.data):
            raise IndexError(f"ID {id} is out of bounds for dataset with {len(self.data)} rows.")
        return self.data.iloc[id - 1]

    def get_row(self, column_name: str, value) -> pd.DataFrame:
        if not hasattr(self, 'data'):
            raise AttributeError("Data not loaded. Call load() method first.")
        if column_name not in self.data.columns:
            raise ValueError(f"Column '{column_name}' not found in dataset.")
        return self.data[self.data[column_name] == value]
