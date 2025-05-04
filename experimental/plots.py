import os
import random
import sys
import matplotlib.pyplot as plt

from experimental.data_preprocessing.dataset import Dataset

folder_path = os.path.join(os.path.dirname(__file__), "data")
dataset_files = os.listdir(folder_path)
dataset_dict = {}
for file in dataset_files:
    dataset_name = file.split(".")[0]
    dataset_dict[dataset_name] = Dataset(file)

races = dataset_dict['races']
laptimes = dataset_dict['lap_times']


def main_menu():
    print("Select an option:")
    print("1. Plot driver positions in a random race")
    print("2. ")
    print("3. Exit")
    choice = input("Enter your choice: ")
    return choice

while True:
    user_choice = main_menu()
    if user_choice == "1":
        amount_of_races = races.get_amount_of_rows()
        random_race_id = random.randint(1, amount_of_races)
        random_race = races.get_row("raceId", random_race_id)
        title = random_race['name'].values[0]
        date = random_race['date'].values[0]
        laptimes_race = laptimes.get_row("raceId", random_race_id)
        amount_of_laps = len(laptimes_race['lap'].unique())
        if amount_of_laps == 0:
            print(ValueError("No laps found")) 
            sys.exit(1)

        drivers = laptimes_race['driverId'].unique()
        plt.figure(figsize=(12, 8))
        drivers_set = dataset_dict['drivers']
        for driver in drivers:
            driver_name = drivers_set.get_row("driverId", driver)['surname'].values[0]
            driver_laps = laptimes_race[laptimes_race['driverId'] == driver]
            plt.plot(driver_laps['lap'], driver_laps['position'], label=driver_name)

        plt.gca().invert_yaxis()  # Invert y-axis to show 1st position at the top
        plt.title(f"Driver Positions Over Time for Race: {title} ({date})")
        plt.xlabel("Lap")
        plt.ylabel("Position")
        plt.legend(title="Drivers", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    elif user_choice == "2":
        pass

    elif user_choice == "3":
        print("Exiting program.")
        break
    else:
        print("Invalid choice. Please try again.")

