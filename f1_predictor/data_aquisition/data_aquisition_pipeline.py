from .season_championship import ChampionshipCalculator
from .season_data_gatherer import SeasonDataGatherer
from .season_data_processor import SeasonEloCalculator

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from f1_predictor.features.data_folder_manager import DataFolderManager


class DataAquisitionPipeline:
    def __init__(self, current_year = 2025):
        self.current_year = current_year
        data_folder_manager = DataFolderManager(empty_folder=True, data_folder_path='data_aquisition/2025_data')

    def run(self):
        data_gatherer = SeasonDataGatherer(year=self.current_year)
        data_gatherer.run()
        data_processor = SeasonEloCalculator()
        data_processor.run()
        champtionship_calculator = ChampionshipCalculator()
        champtionship_calculator.run()