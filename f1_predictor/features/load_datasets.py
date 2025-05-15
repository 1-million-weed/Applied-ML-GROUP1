import os
import pandas as pd

class DatasetLoader:
    def __init__(self):
        currentdir = os.path.dirname(os.path.abspath(__file__))
        f1_dir = os.path.dirname(currentdir)
        data_dir = os.path.join(os.path.dirname(f1_dir), 'data')
        self.races = pd.read_csv(os.path.join(data_dir, "races.csv"))
        self.lap_times = pd.read_csv(os.path.join(data_dir, "lap_times.csv"))
        self.results = pd.read_csv(os.path.join(data_dir, "results.csv"))
        self.driver_standings = pd.read_csv(os.path.join(data_dir, "driver_standings.csv"))
        self.qualifying = pd.read_csv(os.path.join(data_dir, "qualifying.csv"))

    