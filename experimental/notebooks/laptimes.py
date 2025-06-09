import asyncio
import json
import logging
import os
import time
from datetime import datetime
import pandas as pd
import fastf1
from fastf1.livetiming.client import SignalRClient
import types

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Create a directory to store lap time data
parent_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(parent_dir, '2025_data', 'live_laptimes')
os.makedirs(data_dir, exist_ok=True)

# Dictionary to store lap times
lap_times = {}

def fix_json(line):
    # Fix F1's not JSON compliant data
    line = line.replace("'", '"') \
        .replace('True', 'true') \
        .replace('False', 'false')
    return line

def process_timing_data(msg):
    """Process timing data to extract lap times"""
    try:
        data = json.loads(msg)
        
        # Extract lap time information
        if isinstance(data, dict):
            if "Lines" in data:
                for driver_code, driver_data in data["Lines"].items():
                    if "Sectors" in driver_data and "Speeds" in driver_data:
                        # This looks like lap time data
                        if "LastLapTime" in driver_data:
                            lap_time = driver_data["LastLapTime"]["Value"]
                            lap_number = driver_data.get("NumberOfLaps", 0)
                            
                            # Store lap time
                            if driver_code not in lap_times:
                                lap_times[driver_code] = []
                            
                            lap_times[driver_code].append({
                                "Lap": lap_number,
                                "LapTime": lap_time,
                                "Timestamp": datetime.now().isoformat()
                            })
                            
                            log.info(f"Driver {driver_code} completed lap {lap_number} with time {lap_time}")
    except Exception as e:
        log.warning(f"Error processing timing data: {str(e)}")

def _handle_message_overwrite(self, msg):
    """Custom message handler for the SignalRClient"""
    msg = fix_json(msg)
    try:
        cat, msg, dt = json.loads(msg)
        if cat == "TimingData":
            process_timing_data(msg)
    except (json.JSONDecodeError, ValueError):
        log.warning("JSON parse error")

def _start_overwrite(self):
    """Connect to the data stream and start collecting data"""
    try:
        asyncio.run(self._async_start())
    except KeyboardInterrupt:
        log.warning("Keyboard interrupt - exiting...")
        raise KeyboardInterrupt

def start_collecting_laptimes():
    """
    Collects live lap time data from F1 race
    This function only works if a F1 live session is active.
    """
    try:
        log.info("Starting lap time collection...")
        retries = 0
        while retries < 5:
            retries += 1
            client = SignalRClient("unused.txt")
            client.topics = ["TimingData"]  # Only get timing data
            
            # Override client methods
            client._handle_message = types.MethodType(_handle_message_overwrite, client)
            client.start = types.MethodType(_start_overwrite, client)
            
            # Start collecting data
            client.start()
    except KeyboardInterrupt:
        # Save lap times to CSV before exiting
        save_laptimes_to_csv()
        print('Data collection interrupted. Lap times saved to CSV.')

def save_laptimes_to_csv():
    """Save collected lap times to CSV files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a DataFrame with all lap times
    all_laps = []
    for driver, laps in lap_times.items():
        for lap in laps:
            lap_info = lap.copy()
            lap_info["Driver"] = driver
            all_laps.append(lap_info)
    
    if all_laps:
        df = pd.DataFrame(all_laps)
        csv_path = os.path.join(data_dir, f"laptimes_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        log.info(f"Lap times saved to {csv_path}")

def get_latest_laptimes():
    """Get the latest lap times as a pandas DataFrame"""
    all_laps = []
    for driver, laps in lap_times.items():
        for lap in laps:
            lap_info = lap.copy()
            lap_info["Driver"] = driver
            all_laps.append(lap_info)
    
    if all_laps:
        return pd.DataFrame(all_laps)
    else:
        return pd.DataFrame(columns=["Driver", "Lap", "LapTime", "Timestamp"])

if __name__ == '__main__':
    try:
        start_collecting_laptimes()
    except KeyboardInterrupt:
        # Save lap times to CSV before exiting
        save_laptimes_to_csv()
        print('Data collection interrupted. Lap times saved to CSV.')