import pandas as pd
import numpy as np
from collections import defaultdict

class ChampionshipCalculator:
    def __init__(self, output_file_path=None):
        self.csv_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '2025_data/results_with_elo.csv')
        self.output_file_path = output_file_path
        if self.output_file_path is None:
            self.output_file_path = self.csv_file_path.replace('.csv', '_with_championships.csv')
        self.df = None
    
    def calculate_championship_stats(self):
        """
        Parse F1 CSV and add championship statistics:
        - Driver championship position
        - Team championship points
        - Driver wins count (up to current round)
        - Driver points before current race
        - Team points before current race
        """
        
        # Read the CSV file
        try:
            self.df = pd.read_csv(self.csv_file_path)
        except Exception as e:
            return None
        
        # Sort by RoundNumber and ClassifiedPosition to ensure proper chronological order
        self.df = self.df.sort_values(['RoundNumber', 'ClassifiedPosition']).reset_index(drop=True)
        
        # Initialize new columns
        self.df['DriverChampionshipPosition'] = 0
        self.df['TeamChampionshipPoints'] = 0
        self.df['DriverWinsCount'] = 0
        self.df['DriverWinsCount_before'] = 0  # New column for wins before current race
        self.df['DriverPointsBefore'] = 0  # New column for points before current race
        self.df['TeamPointsBefore'] = 0  # New column for team points before current race
        
        # Track cumulative stats
        driver_points = defaultdict(int)
        driver_wins = defaultdict(int)
        team_points = defaultdict(int)
        
        # Process each round
        for round_num in sorted(self.df['RoundNumber'].unique()):
            round_data = self.df[self.df['RoundNumber'] == round_num].copy()
            
            # Update win count and points before the current race for all drivers in this round
            for idx, row in round_data.iterrows():
                driver_id = row['DriverId']
                team_name = row['TeamName']
                self.df.at[idx, 'DriverWinsCount_before'] = driver_wins[driver_id]
                self.df.at[idx, 'DriverPointsBefore'] = driver_points[driver_id]
                self.df.at[idx, 'TeamPointsBefore'] = team_points[team_name]
            
            # First identify winners for this round
            for idx, row in round_data.iterrows():
                driver_id = row['DriverId']
                classified_pos = row['ClassifiedPosition']
                
                # Count win if driver finished first in this round
                if pd.notna(classified_pos) and classified_pos == 1:
                    driver_wins[driver_id] += 1
                
                if pd.notna(classified_pos) and isinstance(classified_pos, str) and classified_pos.strip() == "1":
                    driver_wins[driver_id] += 1
            
            # Now update points and championship data
            for idx, row in round_data.iterrows():
                driver_id = row['DriverId']
                team_name = row['TeamName']
                points = row['Points'] if pd.notna(row['Points']) else 0
                
                # Add points to driver and team totals
                driver_points[driver_id] += points
                team_points[team_name] += points
            
            # Calculate championship positions for this round
            # Sort drivers by points (descending), then by wins (descending)
            driver_standings = sorted(
                driver_points.items(), 
                key=lambda x: (x[1], driver_wins[x[0]]), 
                reverse=True
            )
            
            # Create position lookup
            driver_positions = {driver_id: pos + 1 for pos, (driver_id, _) in enumerate(driver_standings)}
            
            # Update dataframe for this round
            for idx, row in round_data.iterrows():
                driver_id = row['DriverId']
                team_name = row['TeamName']
                
                self.df.at[idx, 'DriverChampionshipPosition'] = driver_positions[driver_id]
                self.df.at[idx, 'TeamChampionshipPoints'] = team_points[team_name]
                self.df.at[idx, 'DriverWinsCount'] = driver_wins[driver_id]
        
        # Save updated CSV
        try:
            self.df.to_csv(self.output_file_path, index=False)
        except Exception as e:
            pass
        
        return self.df

    def analyze_championship_progression(self):
        """
        Additional analysis of championship progression throughout the season
        """
        if self.df is None:
            return
            
        # Track championship leader changes
        leaders_by_round = []
        for round_num in sorted(self.df['RoundNumber'].unique()):
            round_data = self.df[self.df['RoundNumber'] == round_num]
            leader = round_data[round_data['DriverChampionshipPosition'] == 1].iloc[0]
            leaders_by_round.append({
                'Round': round_num,
                'Leader': leader['BroadcastName'],
                'Team': leader['TeamName'],
                'Wins': leader['DriverWinsCount']
            })

    def run(self):
        """
        Run the complete championship analysis process
        """
        # Process the CSV
        updated_df = self.calculate_championship_stats()
        
        if updated_df is not None:
            # Run additional analysis
            self.analyze_championship_progression()
            
            return updated_df
        return None

