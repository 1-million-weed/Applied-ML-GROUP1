from ..data_aquisition.api_sample_generator import SampleGenerator
from ..data_aquisition.season_championship import ChampionshipCalculator
from ..data_aquisition.season_data_gatherer import SeasonDataGatherer
from ..data_aquisition.season_data_processor import SeasonEloCalculator


class APIpipeline:
    def __init__(self, round: int = 1, lap: int = 1):
        self.round = round
        self.lap = lap
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
        print(samples)


