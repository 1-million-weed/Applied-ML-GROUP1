import fastf1
import datetime
import pandas as pd
import os

class SeasonDataGatherer:
    def __init__(self, year=2025):
        self.year = year
        self.schedule = fastf1.get_event_schedule(self.year)
        self.current_date = datetime.datetime.now().date()
        data_aquisition_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(data_aquisition_dir, '2025_data')
        self.passed_races = []
        self.upcoming_race = None
        self.laptimes = []
        self.qualifyings = []
        self.results = []
        
    def race_finished(self, date):
        return date < self.current_date
        
    def identify_races(self):
        for event_date in self.schedule['EventDate']:
            event_date = pd.Timestamp(event_date).date()
            if self.race_finished(event_date):
                event_date_str = event_date.strftime('%Y-%m-%d')
                event_format = self.schedule[self.schedule['EventDate'] == event_date_str]['EventFormat'].values[0]
                if event_format != 'testing':
                    self.passed_races.append(self.schedule[self.schedule['EventDate'] == event_date_str])
            else:
                event_date_str = event_date.strftime('%Y-%m-%d')
                self.upcoming_race = self.schedule[self.schedule['EventDate'] == event_date_str]
                break
    
    def collect_session_data(self):
        for race in self.passed_races:
            round_number = race['RoundNumber'].values[0]
            
            qualifying_session = fastf1.get_session(self.year, round_number, 'Q')
            qualifying_session.load()
            
            race_session = fastf1.get_session(self.year, round_number, 'R')
            race_session.load()
            
            laptimes_df = race_session.laps
            laptimes_df['RoundNumber'] = round_number
            self.laptimes.append(laptimes_df)
            
            qualifying_df = qualifying_session.results
            qualifying_df['RoundNumber'] = round_number
            self.qualifyings.append(qualifying_df)
            
            race_results = race_session.results
            race_results['RoundNumber'] = round_number
            self.results.append(race_results)
    
    def save_data(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
        combined_laptimes = pd.concat(self.laptimes, ignore_index=True)
        combined_laptimes.to_csv(os.path.join(self.output_dir, "combined_laptimes.csv"), index=False)
        
        combined_qualifyings = pd.concat(self.qualifyings, ignore_index=True)
        combined_qualifyings.to_csv(os.path.join(self.output_dir, "combined_qualifyings.csv"), index=False)
        
        combined_results = pd.concat(self.results, ignore_index=True)
        combined_results.to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
    
    def run(self):
        self.identify_races()
        self.collect_session_data()
        self.save_data()

if __name__ == "__main__":
    gatherer = SeasonDataGatherer()
    gatherer.run()