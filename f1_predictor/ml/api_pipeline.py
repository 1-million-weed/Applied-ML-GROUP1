import os 
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the project root to sys.path
sys.path.append(project_root)

from f1_predictor.data_aquisition.api_sample_generator import SampleGenerator
from f1_predictor.data_aquisition.season_championship import ChampionshipCalculator
from f1_predictor.data_aquisition.season_data_gatherer import SeasonDataGatherer
from f1_predictor.data_aquisition.season_data_processor import SeasonEloCalculator


class APIpipeline:
    def __init__(self, model, round: int = 1, lap: int = 1):
        self.round = round
        self.lap = lap
        self.model = model
        self.sample_generator = SampleGenerator(round=self.round, lap=self.lap)
        self.championship_calculator = ChampionshipCalculator()
        self.season_data_gatherer = SeasonDataGatherer()
        self.season_elo_calculator = SeasonEloCalculator()

    def generate_sample(self):
        return self.sample_generator.generate_sample()

    def calculate_championships(self):
        return self.championship_calculator.run()

    def gather_season_data(self):
        return self.season_data_gatherer.run()

    def calculate_season_elo(self):
        return self.season_elo_calculator.run(debug_mode=False)
        
    def run(self):
        self.gather_season_data()
        self.calculate_season_elo()
        self.calculate_championships()
        samples = self.generate_sample()
        return self.model.predict_multiple(samples)
        

