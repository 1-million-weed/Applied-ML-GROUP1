import os
import pandas as pd

class DatasetLoader:
    """
    Class for loading Formula 1 datasets.

    Attributes:
        races (pd.DataFrame): Data of F1 races.
        lap_times (pd.DataFrame): Lap-by-lap time data.
        results (pd.DataFrame): Final race results.
        qualifying (pd.DataFrame): Qualifying session times and positions.
        constructor_standings (pd.DataFrame): Team standings merged with constructor names.
        driver_standings (pd.DataFrame): Driver standings including ELO rating.
    """
    def __init__(self) -> None:
        """
        Constructor method to initialize the DatasetLoader and load all necessary CSV files.

        :raises FileNotFoundError: If `driver_standings_with_elo.csv` is not found.
        """
        currentdir = os.path.dirname(os.path.abspath(__file__))
        f1_dir = os.path.dirname(currentdir)
        data_dir = os.path.join(os.path.dirname(f1_dir), 'data')
        self.races = pd.read_csv(os.path.join(data_dir, "races.csv"))
        self.lap_times = pd.read_csv(os.path.join(data_dir, "lap_times.csv"))
        self.results = pd.read_csv(os.path.join(data_dir, "results.csv"))
        self.qualifying = pd.read_csv(os.path.join(data_dir, "qualifying.csv"))
        self.constructor_standings = pd.read_csv(os.path.join(data_dir, "constructor_standings.csv"))
        constructors = pd.read_csv(os.path.join(data_dir, "constructors.csv"))
        self.constructor_standings = pd.merge(self.constructor_standings, constructors[['constructorId','name']], on='constructorId', how='left')
        if not os.path.exists(os.path.join(data_dir, "driver_standings_with_elo.csv")):
            raise FileNotFoundError("driver_standings_with_elo.csv not found in data directory. Please run the script to generate it.")
        else:
            self.driver_standings = pd.read_csv(os.path.join(data_dir, "driver_standings_with_elo.csv"))
    