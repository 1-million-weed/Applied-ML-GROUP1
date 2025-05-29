import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import logging

#https://www.kaggle.com/code/lorenzojayd/elo-system-in-formula-1/notebook

class F1DataLoader:
    """Handles loading and initial processing of F1 data"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
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
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all required CSV files"""
        files = {
            'results': 'results.csv',
            'drivers': 'drivers.csv',
            'constructors': 'constructors.csv',
            'races': 'races.csv',
            'driver_standings': 'driver_standings.csv'
        }
        
        data = {}
        for key, filename in files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                data[key] = pd.read_csv(filepath)
                self.logger.info(f"Loaded {filename}: {data[key].shape}")
            else:
                self.logger.warning(f"File not found: {filepath}")
                
        return data
    
    def create_master_dataframe(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create the master dataframe by merging all relevant data"""
        
        # Define features to keep from each dataset
        features_races = ['raceId', 'year', 'round', 'name']
        features_drivers = ['driverId', 'driverRef', 'code', 'forename', 'surname']
        features_constructors = ['constructorId', 'constructorRef', 'name']
        
        # Merge datasets
        df_master = pd.merge(data['results'], data['races'][features_races], 
                           how='left', on='raceId')
        df_master = df_master.merge(data['drivers'][features_drivers], 
                                  how='left', on='driverId')
        df_master = df_master.merge(data['constructors'][features_constructors], 
                                  how='left', on='constructorId')
        
        # Create derived columns
        df_master['race_yearAndName'] = df_master['year'].astype(str) + ' ' + df_master['name_x']
        
        # Select and rename columns
        features_master = ['resultId', 'year', 'round', 'name_x', 'raceId', 
                          'race_yearAndName', 'driverRef', 'code', 'forename', 'surname', 
                          'constructorRef', 'name_y', 'positionOrder', 'position', 'driverId']
        
        df_master = df_master[features_master].rename(columns={
            'name_x': 'race_name',
            'year': 'race_year',
            'round': 'race_round',
            'forename': 'driver_firstName',
            'surname': 'driver_lastName', 
            'name_y': 'constructor_name'
        })
        
        # Sort and initialize ELO column
        df_master = df_master.sort_values(by=['race_year', 'race_round'], 
                                        ascending=True).reset_index(drop=True)
        df_master['elo'] = None
        
        self.logger.info(f"Master dataframe created: {df_master.shape}")
        return df_master


class F1Visualizer:
    """Handles all visualization functionality"""
    
    def __init__(self, style: str = 'whitegrid'):
        sns.set_style(style)
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
    
    def plot_driver_to_race_ratio(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Plot driver to race ratio analysis"""
        
        # Calculate yearly statistics
        df_yearly = self._calculate_yearly_stats(df)
        
        # Calculate drivers per race
        df_drivers_per_race = self._calculate_drivers_per_race(df)
        
        # Create the visualization
        self._create_ratio_plots(df_yearly, df_drivers_per_race)
        
        return df_yearly, df_drivers_per_race
    
    def _calculate_yearly_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate yearly driver and race statistics"""
        df_yearly = df[['race_year']].drop_duplicates(subset='race_year').reset_index(drop=True)
        
        for i, year in df_yearly.iterrows():
            year_data = df[df['race_year'] == year['race_year']]
            df_yearly.loc[i, 'count_drivers'] = year_data['driverRef'].nunique()
            df_yearly.loc[i, 'count_races'] = year_data['race_round'].max()
            
        return df_yearly
    
    def _calculate_drivers_per_race(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate drivers per race statistics"""
        df_drivers_per_race = pd.DataFrame({
            'race_yearAndName': df['race_yearAndName'].unique().astype(str),
            'count': 0
        })
        
        for i, race in df_drivers_per_race.iterrows():
            count = df[df['race_yearAndName'] == race['race_yearAndName']]['driverRef'].nunique()
            df_drivers_per_race.loc[i, 'count'] = count
            
        return df_drivers_per_race
    
    def _create_ratio_plots(self, df_yearly: pd.DataFrame, df_drivers_per_race: pd.DataFrame):
        """Create the ratio visualization plots"""
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
            3, 2, gridspec_kw={'width_ratios': [1, 7]}, figsize=(16, 9))
        fig.tight_layout(pad=3)

        # Drivers plots
        sns.boxplot(data=df_yearly, y='count_drivers', ax=ax1)
        ax1.set_ylim(0, 120)
        ax1.set_ylabel('Number of drivers')

        sns.barplot(data=df_yearly, x='race_year', y='count_drivers', ax=ax2)
        ax2.set_xticks(range(0, 71, 5))
        ax2.set_ylim(0, 120)
        ax2.set_ylabel('Number of drivers')
        ax2.set_xlabel('Year')

        # Races plots
        sns.boxplot(data=df_yearly, y='count_races', ax=ax3)
        ax3.set_ylim(0, 30)
        ax3.set_ylabel('Number of races')

        sns.barplot(data=df_yearly, x='race_year', y='count_races', ax=ax4)
        ax4.set_xticks(range(0, 71, 5))
        ax4.set_ylim(0, 30)
        ax4.set_ylabel('Number of races')
        ax4.set_xlabel('Year')

        # Drivers per race plots
        sns.boxplot(data=df_drivers_per_race, y='count', ax=ax5)
        ax5.set_ylim(0, 50)
        ax5.set_ylabel('Number of drivers')

        sns.scatterplot(data=df_drivers_per_race, x='race_yearAndName', y='count', ax=ax6)
        ax6.set_xticks(range(0, 1100, 200))
        ax6.set_ylim(0, 50)
        ax6.set_ylabel('Number of Drivers')
        ax6.set_xlabel('Race')

        fig.suptitle('Driver-Race ratio over time')
        plt.show()
    
    def plot_yearly_elo_distribution(self, df: pd.DataFrame):
        """Plot yearly ELO distribution using ridge plots"""
        sns.set_theme(style='white', rc={'axes.facecolor': (0, 0, 0, 0), 'axes.linewidth': 2})
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)

        g = sns.FacetGrid(df, palette=pal, row='race_year', hue='race_year', 
                         aspect=15, height=.5)
        
        g.map(sns.kdeplot, 'elo', bw_adjust=.5, clip_on=False, fill=True, 
              alpha=1, linewidth=1.5)
        g.refline(y=0, linewidth=2, linestyle='-', color=None, clip_on=False)

        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .1, label, fontweight='bold', color=color, fontsize=13, 
                   ha='left', va='center', transform=ax.transAxes)

        g.map(label, 'elo')
        g.figure.subplots_adjust(hspace=-0.35)
        
        g.set_titles('')
        g.set(yticks=[], xlabel='Elo Rating', ylabel='')
        g.despine(left=True, bottom=True)
    
    def plot_driver_vs_teammates(self, df: pd.DataFrame, driver_ref: str):
        """Plot driver ELO vs teammates over time"""
        if driver_ref not in df['driverRef'].values:
            self.logger.error(f"Driver {driver_ref} not found in data")
            return
            
        driver_first_name = df[df['driverRef'] == driver_ref]['driver_firstName'].iloc[0]
        driver_last_name = df[df['driverRef'] == driver_ref]['driver_lastName'].iloc[0]
        
        df_driver_races = df[df['driverRef'] == driver_ref].reset_index(drop=True)
        df_driver_vs_teammate = self._prepare_teammate_comparison_data(df, df_driver_races, driver_ref)
        
        # Create the plot
        fig = plt.figure(figsize=(20, 6))
        sns.lineplot(data=df_driver_vs_teammate, x='race', y='driver_elo', color='black', label=driver_ref)
        sns.lineplot(data=df_driver_vs_teammate, x='race', y='teammate_elo', hue='teammate')
        
        plt.ylim(800, 1500)
        length = len(df_driver_races)
        plt.xticks(range(0, length, max(1, length // 4)))
        plt.tick_params('both', length=10, width=2, which='major')
        plt.ylabel('Elo Rating')
        plt.xlabel('Race')
        plt.title(f"{driver_first_name} {driver_last_name}'s Elo rating over time")
        
        legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        legend.get_frame().set_facecolor('white')
        plt.tight_layout()
        plt.show()
    
    def _prepare_teammate_comparison_data(self, df: pd.DataFrame, df_driver_races: pd.DataFrame, 
                                        driver_ref: str) -> pd.DataFrame:
        """Prepare data for driver vs teammate comparison"""
        df_comparison = pd.DataFrame({
            'race': df_driver_races['race_yearAndName'].values,
            'driver': driver_ref,
            'teammate': None,
            'driver_elo': df_driver_races['elo'].values,
            'teammate_elo': None,
        })

        for i, race in df_driver_races.iterrows():
            current_constructor = race['constructorRef']
            current_race = race['race_yearAndName']
            
            teammates = df[
                (df['race_yearAndName'] == current_race) & 
                (df['constructorRef'] == current_constructor) & 
                (df['driverRef'] != driver_ref)
            ]
            
            if not teammates.empty:
                teammate = teammates.iloc[0]
                df_comparison.loc[df_comparison['race'] == current_race, 'teammate'] = teammate['driverRef']
                df_comparison.loc[df_comparison['race'] == current_race, 'teammate_elo'] = teammate['elo']

        return df_comparison


class F1EloCalculator:
    """Enhanced F1 ELO calculation system"""
    
    def __init__(self, df_main: pd.DataFrame, k_factor: int = 32, c_factor: int = 400):
        self.df_main = df_main.copy()
        self.k_factor = k_factor
        self.c_factor = c_factor
        self.logger = self._setup_logger()
        
        # Initialize ELO tracking dataframe
        self.df_elo = pd.DataFrame({
            'driverRef': self.df_main['driverRef'].unique(),
            'running_elo': 1000,
            'pre_race_elo': 1000,
        })
        
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

    def compute_expected_outcome(self, elo_a: float, elo_b: float) -> float:
        """Compute expected outcome given ELO ratings of driver A and B"""
        if elo_a is None or elo_b is None:
            raise ValueError('ELO of driver A or B is invalid')
        
        q_a = 10 ** (elo_a / self.c_factor)
        q_b = 10 ** (elo_b / self.c_factor)
        return q_a / (q_a + q_b)

    def compute_actual_outcome(self, position_a: int, position_b: int) -> int:
        """Compute actual outcome given finishing positions"""
        if position_a is None or position_b is None:
            raise ValueError('Position of driver A or B is invalid')
        
        return 1 if position_a < position_b else 0

    def compute_elo_gain(self, expected_outcome: float, actual_outcome: int) -> float:
        """Calculate ELO gain/loss"""
        if expected_outcome is None or actual_outcome is None:
            raise ValueError('Expected or actual outcome is invalid')
        
        return self.k_factor * (actual_outcome - expected_outcome)

    def get_elo(self, driver_ref: str) -> float:
        """Get driver's current pre-race ELO"""
        if not driver_ref:
            raise ValueError('No driver input')
        
        elo_data = self.df_elo[self.df_elo['driverRef'] == driver_ref]['pre_race_elo']
        
        if elo_data.empty:
            raise ValueError(f'Driver {driver_ref} not found')
        
        return elo_data.iloc[0]

    def get_teammates(self, driver_ref: str, race_id: int, constructor_ref: str) -> pd.DataFrame:
        """Get driver's teammates for a specific race"""
        teammates = self.df_main[
            (self.df_main['raceId'] == race_id) & 
            (self.df_main['constructorRef'] == constructor_ref) & 
            (self.df_main['driverRef'] != driver_ref)
        ]
        return teammates.reset_index(drop=True)

    def update_running_elo(self, driver_ref: str, new_elo: float):
        """Update driver's running ELO"""
        self.df_elo.loc[self.df_elo['driverRef'] == driver_ref, 'running_elo'] = new_elo

    def update_pre_race_elo(self):
        """Update pre-race ELO for all drivers"""
        self.df_elo['pre_race_elo'] = self.df_elo['running_elo']

    def update_main_dataframe(self, race_id: int, driver_ref: str, new_elo: float):
        """Update main dataframe with new ELO"""
        mask = (self.df_main['raceId'] == race_id) & (self.df_main['driverRef'] == driver_ref)
        self.df_main.loc[mask, 'elo'] = new_elo

    def reset_elo(self):
        """Reset all ELO ratings to 1000"""
        self.df_elo['pre_race_elo'] = 1000
        self.df_elo['running_elo'] = 1000

    def run_elo_calculation(self) -> pd.DataFrame:
        """Execute the complete ELO calculation"""
        self.logger.info("Starting ELO calculation...")
        
        previous_race_id = self.df_main.loc[0, 'raceId']
        processed_races = 0

        for i, data in self.df_main.iterrows():
            current_race_id = data['raceId']
            
            # Update pre-race ELO when moving to next race
            if previous_race_id != current_race_id:
                self.update_pre_race_elo()
                previous_race_id = current_race_id
                processed_races += 1
                
                if processed_races % 100 == 0:
                    self.logger.info(f"Processed {processed_races} races...")

            # Process current driver
            driver_ref = data['driverRef']
            position_class = data['position']
            position_order = data['positionOrder']
            current_elo = self.get_elo(driver_ref)

            # Only calculate ELO for drivers who finished the race
            if position_class != '\\N':
                teammates = self.get_teammates(driver_ref, current_race_id, data['constructorRef'])
                elo_gain = 0
                
                for _, teammate in teammates.iterrows():
                    teammate_position_class = teammate['position']
                    
                    if teammate_position_class != '\\N':
                        teammate_elo = self.get_elo(teammate['driverRef'])
                        expected = self.compute_expected_outcome(current_elo, teammate_elo)
                        actual = self.compute_actual_outcome(position_order, teammate['positionOrder'])
                        elo_gain += self.compute_elo_gain(expected, actual)
                
                new_elo = round(current_elo + elo_gain)
            else:
                new_elo = current_elo

            # Update ELO values
            self.update_running_elo(driver_ref, new_elo)
            self.update_main_dataframe(current_race_id, driver_ref, new_elo)

        # Final update
        self.update_pre_race_elo()
        self.reset_elo()
        
        self.logger.info(f"ELO calculation completed. Processed {processed_races} races.")
        return self.df_main

    def get_history(self) -> pd.DataFrame:
        """Get the complete ELO history"""
        return self.df_main


class F1EloAnalyzer:
    """Main analyzer class that coordinates all components"""
    
    def __init__(self):
        currentdir = os.path.dirname(os.path.abspath(__file__))
        f1_dir = os.path.dirname(currentdir)
        data_dir = os.path.join(os.path.dirname(f1_dir), 'data')
        self.data_dir = data_dir
        self.data_loader = F1DataLoader(data_dir)
        self.visualizer = F1Visualizer()
        self.logger = self._setup_logger()
        
        # Data storage
        self.raw_data = None
        self.master_df = None
        self.elo_calculator = None
        self.elo_history = None
        
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
    
    def load_and_prepare_data(self, exclude_races: Optional[List[str]] = None):
        """Load and prepare all data"""
        self.logger.info("Loading and preparing data...")
        
        # Load raw data
        self.raw_data = self.data_loader.load_data()
        
        # Create master dataframe
        self.master_df = self.data_loader.create_master_dataframe(self.raw_data)
        
        # Exclude specific races if specified
        if exclude_races:
            for race_name in exclude_races:
                initial_count = len(self.master_df)
                self.master_df = self.master_df[self.master_df['race_name'] != race_name]
                excluded_count = initial_count - len(self.master_df)
                self.logger.info(f"Excluded {excluded_count} records from {race_name}")
        
        self.logger.info("Data preparation completed")
    
    def calculate_elo_ratings(self, k_factor: int = 32, c_factor: int = 400):
        """Calculate ELO ratings for all drivers"""
        if self.master_df is None:
            raise ValueError("Data must be loaded first. Call load_and_prepare_data()")
        
        self.logger.info("Calculating ELO ratings...")
        self.elo_calculator = F1EloCalculator(self.master_df, k_factor, c_factor)
        self.elo_history = self.elo_calculator.run_elo_calculation()
        
    def get_top_drivers(self, n: int = 30) -> pd.DataFrame:
        """Get top N drivers by maximum ELO achieved"""
        if self.elo_history is None:
            raise ValueError("ELO ratings must be calculated first")
        
        df_top = self.elo_history.sort_values(by='elo', ascending=False).drop_duplicates(subset='driverRef')
        df_top = df_top.drop(columns=['race_name', 'resultId', 'race_year', 'race_round', 
                                    'raceId', 'driverRef', 'constructorRef', 'positionOrder']).reset_index(drop=True)
        
        df_top.rename(columns={
            'driver_firstName': 'Driver First Name',
            'driver_lastName': 'Driver Last Name',
            'constructor_name': 'Constructor',
            'race_yearAndName': 'Race Achieved',
            'elo': 'Elo'
        }, inplace=True)
        
        df_top['position'] = df_top.index + 1
        df_top = df_top.set_index('position', drop=True)
        
        return df_top.head(n)
    
    def save_results(self, output_filename: str = "driver_standings_with_elo.csv"):
        """Save results with ELO ratings merged"""
        if self.elo_history is None or self.raw_data is None:
            raise ValueError("ELO ratings and data must be available")
        
        # Merge with driver standings
        merged_standings = pd.merge(
            self.raw_data['driver_standings'], 
            self.elo_history[['raceId', 'driverId', 'elo', 'driverRef', 'constructor_name', 'code']], 
            on=['raceId', 'driverId'], 
            how='left'
        )
        
        output_path = Path(self.data_dir) / output_filename
        merged_standings.to_csv(output_path, index=False)
        self.logger.info(f"Results saved to {output_path}")
        
        return merged_standings
    
    def run_complete_analysis(self, exclude_races: Optional[List[str]] = None, 
                            k_factor: int = 32, c_factor: int = 400):
        """Run the complete analysis pipeline"""
        self.logger.info("Starting complete F1 ELO analysis...")
        
        # Load and prepare data
        self.load_and_prepare_data(exclude_races or ['Indianapolis 500'])
        
        # Calculate ELO ratings
        self.calculate_elo_ratings(k_factor, c_factor)
        
        # Save results
        self.save_results()
        
        self.logger.info("Complete analysis finished!")
        
        return {
            'elo_history': self.elo_history,
            'top_drivers': self.get_top_drivers(),
            'master_df': self.master_df
        }
    
    # Visualization methods
    def plot_driver_race_ratio(self):
        """Plot driver to race ratio analysis"""
        if self.master_df is None:
            raise ValueError("Data must be loaded first")
        return self.visualizer.plot_driver_to_race_ratio(self.master_df)
    
    def plot_yearly_elo_distribution(self):
        """Plot yearly ELO distribution"""
        if self.elo_history is None:
            raise ValueError("ELO ratings must be calculated first")
        self.visualizer.plot_yearly_elo_distribution(self.elo_history)
    
    def plot_driver_vs_teammates(self, driver_ref: str):
        """Plot specific driver vs teammates"""
        if self.elo_history is None:
            raise ValueError("ELO ratings must be calculated first")
        self.visualizer.plot_driver_vs_teammates(self.elo_history, driver_ref)
