import os
import pandas as pd
from datetime import timedelta

class SampleGenerator:
    def __init__(self, round:int = 1, lap:int = 1):
        self.round = round
        self.lap = lap
        self.results = pd.read_csv(os.path.join(os.path.dirname(__file__), '2025_data/results_with_elo_with_championships.csv'))
        #DriverNumber,BroadcastName,Abbreviation,DriverId,TeamName,TeamColor,TeamId,FirstName,LastName,FullName,HeadshotUrl,CountryCode,Position,ClassifiedPosition,GridPosition,Q1,Q2,Q3,Time,Status,Points,RoundNumber,elo_before,elo_after,DriverChampionshipPosition,TeamChampionshipPoints,DriverWinsCount,DriverWinsCount_before
        #4,L NORRIS,NOR,norris,McLaren,FF8000,mclaren,Lando,Norris,Lando Norris,https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LANNOR01_Lando_Norris/lannor01.png.transform/1col/image.png,,1.0,1,1.0,,,,0 days 01:42:06.304000,Finished,25.0,1,1130.0,1144.0,1,27,1,0
        self.laptimes = pd.read_csv(os.path.join(os.path.dirname(__file__), '2025_data/combined_laptimes.csv'))
        #Time,Driver,DriverNumber,LapTime,LapNumber,Stint,PitOutTime,PitInTime,Sector1Time,Sector2Time,Sector3Time,Sector1SessionTime,Sector2SessionTime,Sector3SessionTime,SpeedI1,SpeedI2,SpeedFL,SpeedST,IsPersonalBest,Compound,TyreLife,FreshTyre,Team,LapStartTime,LapStartDate,TrackStatus,Position,Deleted,DeletedReason,FastF1Generated,IsAccurate,RoundNumber
        #0 days 01:13:00.002000,VER,1,0 days 00:01:59.392000,1.0,1.0,,,,0 days 00:00:20.705000,0 days 00:00:55.250000,,0 days 01:12:04.853000,0 days 01:13:00.058000,249.0,292.0,247.0,215.0,False,INTERMEDIATE,1.0,True,Red Bull Racing,0 days 01:11:00.355000,2025-03-16 04:18:22.974,124,5.0,False,,False,False,1
        self.qualifying = pd.read_csv(os.path.join(os.path.dirname(__file__), '2025_data/combined_qualifyings.csv'))
        #DriverNumber,BroadcastName,Abbreviation,DriverId,TeamName,TeamColor,TeamId,FirstName,LastName,FullName,HeadshotUrl,CountryCode,Position,ClassifiedPosition,GridPosition,Q1,Q2,Q3,Time,Status,Points,RoundNumber
        #4,L NORRIS,NOR,norris,McLaren,FF8000,mclaren,Lando,Norris,Lando Norris,https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LANNOR01_Lando_Norris/lannor01.png.transform/1col/image.png,,1.0,,,0 days 00:01:15.912000,0 days 00:01:15.415000,0 days 00:01:15.096000,,,,1

    def simplify_time_string(self, time_str):
        #add a base case
        if pd.isna(time_str) or time_str == "0 days 00:00:00.000000":
            return "0:00.000"
        
        # Convert to timedelta
        days_part, time_part = time_str.split(" days ")
        days = int(days_part)
        hours, minutes, seconds = map(float, time_part.split(":"))
        
        # Compute total seconds
        total_seconds = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds).total_seconds()
        
        # Break into minutes and seconds
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        
        # Format: M:SS.mmm
        return f"{minutes}:{seconds:06.3f}"

    def generate_sample(self):
        race_result = self.results[self.results['RoundNumber'] == self.round].copy()
        race_qualifying = self.qualifying[self.qualifying['RoundNumber'] == self.round].copy()
        race_laptimes = self.laptimes[self.laptimes['RoundNumber'] == self.round].copy()

        # Apply simplify_time_string to the 'Time' column in race_result and race_qualifying
        race_qualifying['Q1'] = race_qualifying['Q1'].apply(self.simplify_time_string)
        race_qualifying['Q2'] = race_qualifying['Q2'].apply(self.simplify_time_string)
        race_qualifying['Q3'] = race_qualifying['Q3'].apply(self.simplify_time_string)
        race_laptimes['LapTime'] = race_laptimes['LapTime'].apply(self.simplify_time_string)

        
            



if __name__ == "__main__":
    # Example usage
    sample_gen = SampleGenerator(round=1, lap=2)
    sample_gen.generate_sample()