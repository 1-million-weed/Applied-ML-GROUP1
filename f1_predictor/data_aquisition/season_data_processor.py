import pandas as pd
import os
import numpy as np
import logging
from typing import Dict, Optional

class SeasonEloCalculator:
    def __init__(self):
        data_aquisition_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(data_aquisition_dir, '2025_data')
        self.laptimes = os.path.join(self.output_dir, 'laptimes.csv')
        self.qualifyings = os.path.join(self.output_dir, 'qualifyings.csv')
        self.results = os.path.join(self.output_dir, 'results.csv')
        
        f1_dir = os.path.dirname(data_aquisition_dir)
        data_folder = os.path.join(os.path.dirname(f1_dir), 'data')
        self.last_year_standings = pd.read_csv(os.path.join(data_folder, 'driver_standings_with_elo.csv'))
        self.historic_races = pd.read_csv(os.path.join(data_folder, 'races.csv'))
        
        # ELO calculation parameters
        self.k_factor = 32  # Standard K-factor for Elo calculations
        self.c_factor = 400  # Standard C-factor for Elo calculations
        self.regression_factor = 0.85  # Regression factor for between seasons
        
        # ELO tracking
        self.driver_elos = {}
        self.last_year_elo = None
        
        # Driver code mapping for handling code changes between seasons
        self.driver_code_mapping = {
            # Add mappings if a driver's code changed from last season to this one
            # 'NEW_CODE': 'OLD_CODE'
        }
        
        # Setup logging
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the class"""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _get_last_race_elo_prev_season(self):
        """Get the ELO ratings from the last race of the previous season"""
        max_year = self.historic_races['year'].astype(int).max()
        last_race_prev_season = self.historic_races[self.historic_races['year'].astype(int) == max_year]
        max_round = last_race_prev_season['round'].astype(int).max()
        last_race = last_race_prev_season[last_race_prev_season['round'].astype(int) == max_round]
        last_race_id = last_race['raceId'].values[0]
        
        self.last_year_elo = self.last_year_standings[self.last_year_standings['raceId'] == last_race_id]
        self.logger.info(f"Found {len(self.last_year_elo)} drivers from last race of {max_year}")
        
        # Initialize driver ELOs with regression
        self._initialize_driver_elos_with_regression()
        
        return self.last_year_elo

    def _initialize_driver_elos_with_regression(self):
        """Initialize driver ELO ratings with regression factor applied"""
        self.driver_elos = {}
        
        for _, row in self.last_year_elo.iterrows():
            driver_code = row['code']
            driver_elo = row['elo']
            
            if pd.notna(driver_code) and pd.notna(driver_elo):
                # Apply regression towards mean (1000)
                regressed_elo = 1000 + (driver_elo - 1000) * self.regression_factor
                self.driver_elos[driver_code] = round(regressed_elo)
        
        self.logger.info(f"Initialized ELO ratings for {len(self.driver_elos)} drivers with regression factor {self.regression_factor}")
        self.logger.debug(f"Driver ELOs initialized: {self.driver_elos}")

    def get_driver_elo(self, driver_code: str, default_elo: float = 1000.0) -> float:
        """Get driver's current ELO rating, return default if not found"""
        # Check if the code needs to be mapped
        mapped_code = self.driver_code_mapping.get(driver_code, driver_code)
        
        if mapped_code in self.driver_elos:
            return self.driver_elos[mapped_code]
        elif driver_code in self.driver_elos:  # Try original code as fallback
            return self.driver_elos[driver_code]
        else:
            # New driver - initialize with default ELO
            self.driver_elos[driver_code] = default_elo
            self.logger.info(f"New driver {driver_code} initialized with ELO {default_elo}")
            return default_elo

    def compute_expected_outcome(self, elo_a: float, elo_b: float) -> float:
        """Compute expected outcome given ELO ratings of driver A and B"""
        q_a = 10 ** (elo_a / self.c_factor)
        q_b = 10 ** (elo_b / self.c_factor)
        return q_a / (q_a + q_b)

    def compute_actual_outcome(self, position_a: int, position_b: int) -> int:
        """Compute actual outcome given finishing positions (lower position = better)"""
        return 1 if position_a < position_b else 0

    def compute_elo_gain(self, expected_outcome: float, actual_outcome: int) -> float:
        """Calculate ELO gain/loss"""
        return self.k_factor * (actual_outcome - expected_outcome)

    def process_race_results(self, race_results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single race's results and update ELO ratings
        
        Expected columns in race_results_df:
        - Abbreviation: Driver code
        - ClassifiedPosition: Final classified position
        - TeamName: Team name
        - Status: Race completion status
        """
        # Create a copy to avoid modifying original
        results_with_elo = race_results_df.copy()
        
        # Verify required columns exist
        required_columns = ['Abbreviation', 'ClassifiedPosition', 'TeamName']
        for col in required_columns:
            if col not in results_with_elo.columns:
                self.logger.error(f"Required column '{col}' not found in results data")
                return results_with_elo
        
        # Check for and report empty driver codes
        if results_with_elo['Abbreviation'].isna().any() or (results_with_elo['Abbreviation'] == '').any():
            missing_codes = results_with_elo[results_with_elo['Abbreviation'].isna() | (results_with_elo['Abbreviation'] == '')]
            self.logger.warning(f"Found {len(missing_codes)} entries with missing driver codes")
            if not missing_codes.empty:
                self.logger.warning(f"Entries with missing codes: \n{missing_codes}")
        
        # Add ELO before race
        results_with_elo['elo_before'] = results_with_elo['Abbreviation'].apply(self.get_driver_elo)
        
        # Filter to only classified finishers for ELO calculation
        # Adjust the status condition based on your data format
        classified_results = results_with_elo[
            results_with_elo['ClassifiedPosition'].notna() & 
            (results_with_elo['ClassifiedPosition'] != '') &
            (results_with_elo['ClassifiedPosition'] != 0)  # Assuming 0 or empty means DNF
        ].copy()
        
        # Handle different status formats - adjust as needed for your data
        if 'Status' in classified_results.columns:
            classified_results = classified_results[
                classified_results['Status'].str.contains('Finished|Lapped|\+', case=False, na=False) |
                classified_results['Status'].str.match(r'^\d+$', na=False)  # Just position numbers
            ]
        
        if len(classified_results) < 2:
            self.logger.warning("Less than 2 classified finishers - no ELO updates")
            results_with_elo['elo_after'] = results_with_elo['elo_before']
            return results_with_elo
        
        # Convert ClassifiedPosition to numeric
        classified_results['ClassifiedPosition'] = pd.to_numeric(classified_results['ClassifiedPosition'], errors='coerce')
        classified_results = classified_results.dropna(subset=['ClassifiedPosition'])
        
        # Group by team to identify teammates
        team_groups = classified_results.groupby('TeamName')
        
        # Calculate ELO changes
        elo_changes = {}
        
        for driver_idx, driver_row in classified_results.iterrows():
            driver_code = driver_row['Abbreviation']
            driver_position = driver_row['ClassifiedPosition']
            driver_team = driver_row['TeamName']
            driver_elo = self.get_driver_elo(driver_code)
            
            total_elo_change = 0
            comparisons = 0
            
            # Compare against teammates first (more weight)
            if driver_team in team_groups.groups:
                teammates = team_groups.get_group(driver_team)
                for teammate_idx, teammate_row in teammates.iterrows():
                    if teammate_idx != driver_idx:  # Don't compare with self
                        teammate_code = teammate_row['Abbreviation']
                        teammate_position = teammate_row['ClassifiedPosition']
                        teammate_elo = self.get_driver_elo(teammate_code)
                        
                        expected = self.compute_expected_outcome(driver_elo, teammate_elo)
                        actual = self.compute_actual_outcome(driver_position, teammate_position)
                        elo_change = self.compute_elo_gain(expected, actual)
                        
                        # Give teammate comparisons more weight
                        total_elo_change += elo_change * 2
                        comparisons += 2
            
            # Compare against all other drivers (less weight)
            for other_idx, other_row in classified_results.iterrows():
                if other_idx != driver_idx and other_row['TeamName'] != driver_team:
                    other_code = other_row['Abbreviation']
                    other_position = other_row['ClassifiedPosition']
                    other_elo = self.get_driver_elo(other_code)
                    
                    expected = self.compute_expected_outcome(driver_elo, other_elo)
                    actual = self.compute_actual_outcome(driver_position, other_position)
                    elo_change = self.compute_elo_gain(expected, actual)
                    
                    total_elo_change += elo_change * 0.5  # Reduced weight for non-teammates
                    comparisons += 0.5
            
            # Average the ELO change and store
            if comparisons > 0:
                avg_elo_change = total_elo_change / comparisons
                elo_changes[driver_code] = avg_elo_change
            else:
                elo_changes[driver_code] = 0
        
        # Update driver ELOs
        for driver_code, elo_change in elo_changes.items():
            old_elo = self.driver_elos[driver_code]
            new_elo = round(old_elo + elo_change)
            self.driver_elos[driver_code] = new_elo
        
        # Add final ELO column
        results_with_elo['elo_after'] = results_with_elo['Abbreviation'].apply(
            lambda code: self.driver_elos.get(code, self.get_driver_elo(code))
        )
        
        return results_with_elo

    def process_season_results(self, season_results_df: pd.DataFrame, 
                             round_column: str = 'RoundNumber') -> pd.DataFrame:
        """
        Process an entire season's results and calculate ELO progression
        """
        self.logger.info("Processing season results...")
        
        all_results = []
        
        # Process each round in order
        rounds = sorted(season_results_df[round_column].unique())
        
        for round_num in rounds:
            self.logger.info(f"Processing Round {round_num}...")
            
            round_results = season_results_df[season_results_df[round_column] == round_num]
            
            # Process this race
            round_with_elo = self.process_race_results(round_results)
            
            all_results.append(round_with_elo)
        
        # Combine all results
        complete_results = pd.concat(all_results, ignore_index=True)
        
        self.logger.info("Season processing completed!")
        
        return complete_results

    def get_current_standings(self) -> pd.DataFrame:
        """Get current ELO standings as a DataFrame"""
        standings_data = [
            {'DriverCode': code, 'CurrentELO': elo} 
            for code, elo in self.driver_elos.items()
        ]
        
        standings_df = pd.DataFrame(standings_data)
        standings_df = standings_df.sort_values('CurrentELO', ascending=False).reset_index(drop=True)
        standings_df['Position'] = standings_df.index + 1
        
        return standings_df

    def calculate_elo_for_new_season(self) -> Optional[pd.DataFrame]:
        """
        Main method to calculate ELO for the new season
        Returns the results with ELO if successful
        """
        try:
            # Initialize with previous season data
            self._get_last_race_elo_prev_season()
            
            # Check if results file exists
            if not os.path.exists(self.results):
                self.logger.warning(f"Results file not found: {self.results}")
                return None
            
            # Load new season results
            new_season_df = pd.read_csv(self.results)
            self.logger.info(f"Loaded {len(new_season_df)} race results")
            
            # Print unique driver codes for debugging
            unique_drivers = new_season_df['Abbreviation'].unique()
            self.logger.info(f"Found {len(unique_drivers)} unique drivers in new season: {sorted(unique_drivers)}")
            self.logger.info(f"Drivers from last season: {sorted(self.driver_elos.keys())}")
            
            # Identify new drivers not in previous season
            new_drivers = [code for code in unique_drivers if code not in self.driver_elos 
                          and self.driver_code_mapping.get(code) not in self.driver_elos]
            if new_drivers:
                self.logger.info(f"New drivers detected: {new_drivers}")
            
            # Process the results
            results_with_elo = self.process_season_results(new_season_df)
            
            # Save results with ELO
            output_file = os.path.join(self.output_dir, 'results_with_elo.csv')
            results_with_elo.to_csv(output_file, index=False)
            self.logger.info(f"Results with ELO saved to: {output_file}")
            
            # Save current standings
            standings = self.get_current_standings()
            standings_file = os.path.join(self.output_dir, 'current_elo_standings.csv')
            standings.to_csv(standings_file, index=False)
            self.logger.info(f"Current standings saved to: {standings_file}")
            
            return results_with_elo
            
        except Exception as e:
            self.logger.error(f"Error calculating ELO: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def run(self, debug_mode=False):
        """
        Run the ELO calculation process for the new season
        
        Args:
            debug_mode (bool): Enable debug logging if True
        
        Returns:
            pd.DataFrame or None: Results with ELO if successful, None otherwise
        """
        if os.path.exists(os.path.join(self.output_dir, 'results_with_elo.csv')):
            self.logger.info("ELO results already calculated for this season. Skipping calculation.")
            return pd.read_csv(os.path.join(self.output_dir, 'results_with_elo.csv'))
        try:
            # Set logging level
            if debug_mode:
                self.logger.setLevel(logging.DEBUG)
            
            # Get last season ELO data
            self._get_last_race_elo_prev_season()
            print("Last year ELO data:")
            print(self.last_year_elo[['code', 'elo']].head())
            
            # Calculate ELO for new season
            results_with_elo = self.calculate_elo_for_new_season()
            
            if results_with_elo is not None:
                print("\nCurrent ELO standings:")
                print(self.get_current_standings())
            else:
                print("No new season results to process yet.")
                
                # If results file doesn't exist, suggest what to do
                if not os.path.exists(self.results):
                    print(f"\nThe results file is missing: {self.results}")
                    print("Please make sure you have the correct path and that the file exists.")
                    print("You might need to run the data acquisition scripts first to generate this file.")
                    
                    # Check if the directory exists
                    if not os.path.exists(self.output_dir):
                        print(f"\nThe output directory does not exist: {self.output_dir}")
                        print("Creating the directory...")
                        os.makedirs(self.output_dir, exist_ok=True)
            
            return results_with_elo
            
        except Exception as e:
            self.logger.error(f"Error in run method: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
