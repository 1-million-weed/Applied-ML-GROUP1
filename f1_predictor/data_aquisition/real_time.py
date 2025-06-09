import requests
import pandas as pd
import os
from collections import defaultdict
import time
from fastf1.livetiming.client import SignalRClient

parent_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(parent_dir, '2025_data')
results = pd.read_csv(os.path.join(data_dir, 'results_with_elo_with_championships.csv'))
last_race_number = results['RoundNumber'].max()
last_race = results[results['RoundNumber'] == last_race_number]
results_last_race = last_race.drop(columns=['elo_before', 'DriverWinsCount_before', 'DriverWinsCount_before', 'TeamPointsBefore', 'DriverWinsCount_before'])

client = SignalRClient("unused.txt")
