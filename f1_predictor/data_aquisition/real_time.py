import requests
import pandas as pd
import os
from collections import defaultdict
import time

parent_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(parent_dir, '2025_data')
results = pd.read_csv(os.path.join(data_dir, 'results_with_elo_with_championships.csv'))
last_race_number = results['RoundNumber'].max()
last_race = results[results['RoundNumber'] == last_race_number]
results_last_race = last_race.drop(columns=['elo_before', 'DriverWinsCount_before', 'DriverWinsCount_before', 'TeamPointsBefore', 'DriverWinsCount_before'])


import requests

races = requests.get("https://api.openf1.org/v1/meetings").json()
for race in races:
    if race["meeting_name"] == "Australian Grand Prix" and race["session_type"] == "Race":
        print(race["session_key"], race["date_start"])