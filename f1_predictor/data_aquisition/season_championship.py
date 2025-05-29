import pandas as pd
import numpy as np
from collections import defaultdict

class ChampionshipCalculator:
    def __init__(self, csv_file_path, output_file_path=None):
        self.csv_file_path = csv_file_path
        self.output_file_path = output_file_path
        if self.output_file_path is None:
            self.output_file_path = csv_file_path.replace('.csv', '_with_championships.csv')
        self.df = None
    
    def calculate_championship_stats(self):
        """
        Parse F1 CSV and add championship statistics:
        - Driver championship position
        - Team championship points
        - Driver wins count (up to current round)
        """
        
        # Read the CSV file
        try:
            self.df = pd.read_csv(self.csv_file_path)
            print(f"Loaded CSV with {len(self.df)} rows")
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None
        
        # Sort by RoundNumber and ClassifiedPosition to ensure proper chronological order
        self.df = self.df.sort_values(['RoundNumber', 'ClassifiedPosition']).reset_index(drop=True)
        
        # Initialize new columns
        self.df['DriverChampionshipPosition'] = 0
        self.df['TeamChampionshipPoints'] = 0
        self.df['DriverWinsCount'] = 0
        
        # Track cumulative stats
        driver_points = defaultdict(int)
        driver_wins = defaultdict(int)
        team_points = defaultdict(int)
        
        # Process each round
        for round_num in sorted(self.df['RoundNumber'].unique()):
            round_data = self.df[self.df['RoundNumber'] == round_num].copy()
            
            # Update cumulative points and wins for this round
            for idx, row in round_data.iterrows():
                driver_id = row['DriverId']
                team_name = row['TeamName']
                points = row['Points'] if pd.notna(row['Points']) else 0
                classified_pos = row['ClassifiedPosition']
                
                # Add points to driver and team totals
                driver_points[driver_id] += points
                team_points[team_name] += points
                
                # Count wins (ClassifiedPosition = 1 means they won the race)
                if pd.notna(classified_pos) and classified_pos == 1:
                    driver_wins[driver_id] += 1
            
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
        
        # Display sample results
        print("\nSample of updated data:")
        sample_cols = ['RoundNumber', 'BroadcastName', 'TeamName', 'ClassifiedPosition', 'Points', 
                    'DriverChampionshipPosition', 'TeamChampionshipPoints', 'DriverWinsCount']
        print(self.df[sample_cols].head(10).to_string(index=False))
        
        # Show championship standings after final round
        final_round = self.df['RoundNumber'].max()
        final_standings = self.df[self.df['RoundNumber'] == final_round].copy()
        final_standings = final_standings.sort_values('DriverChampionshipPosition')
        
        print(f"\nDriver Championship after Round {final_round}:")
        standings_cols = ['DriverChampionshipPosition', 'BroadcastName', 'TeamName', 'DriverWinsCount']
        print(final_standings[standings_cols].drop_duplicates('BroadcastName').head(10).to_string(index=False))
        
        # Show team standings
        team_standings = final_standings.groupby('TeamName')['TeamChampionshipPoints'].first().sort_values(ascending=False)
        print(f"\nTeam Championship after Round {final_round}:")
        for i, (team, points) in enumerate(team_standings.head(10).items(), 1):
            print(f"{i:2d}. {team:<25} {points:3.0f} points")
        
        # Save updated CSV
        try:
            self.df.to_csv(self.output_file_path, index=False)
            print(f"\nUpdated CSV saved to: {self.output_file_path}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
        
        return self.df

    def analyze_championship_progression(self):
        """
        Additional analysis of championship progression throughout the season
        """
        if self.df is None:
            print("No data available for analysis. Please run calculate_championship_stats first.")
            return
            
        print("\n" + "="*60)
        print("CHAMPIONSHIP PROGRESSION ANALYSIS")
        print("="*60)
        
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
        
        print("\nChampionship Leaders by Round:")
        for leader_info in leaders_by_round:
            print(f"Round {leader_info['Round']:2d}: {leader_info['Leader']:<20} ({leader_info['Team']}) - {leader_info['Wins']} wins")

    def run(self):
        """
        Run the complete championship analysis process
        """
        print("F1 Championship Statistics Calculator")
        print("="*50)
        
        # Process the CSV
        updated_df = self.calculate_championship_stats()
        
        if updated_df is not None:
            # Run additional analysis
            self.analyze_championship_progression()
            
            print(f"\nProcessing complete! New columns added:")
            print("- DriverChampionshipPosition: Current position in driver's championship")
            print("- TeamChampionshipPoints: Current team's total points")  
            print("- DriverWinsCount: Number of wins for the driver so far")
            
            return updated_df
        return None

# Example usage
if __name__ == "__main__":
    # Use the path to the CSV file
    import os
    csv_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '2025_data/results_with_elo.csv')
    
    # Create calculator instance and run
    calculator = ChampionshipCalculator(csv_file_path)
    calculator.run()