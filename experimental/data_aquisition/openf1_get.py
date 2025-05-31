import json
import os
from pathlib import Path
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError, HTTPError


class OpenF1DataFetcher:
    """
    A class to fetch Formula 1 data from the OpenF1 API in CSV format.

    This class provides methods to retrieve meeting data, session data, and lap data
    from the OpenF1 API using the native CSV format support.
    """

    VALID_SESSION_TYPES = [
        'Practice 1', 'Practice 2', 'Practice 3',
        'Qualifying', 'Sprint Qualifying', 'Sprint', 'Race'
    ]

    def __init__(self, data_directory: str = "openf1_data"):
        """
        Initialize the OpenF1DataFetcher.

        Args:
            data_directory (str): Name of the directory to store CSV files.
        """
        self._base_url = "https://api.openf1.org/v1/"
        self._data_dir = self._create_data_directory(data_directory)

    @property
    def base_url(self) -> str:
        """Get the base URL for the OpenF1 API."""
        return self._base_url

    @property
    def data_directory(self) -> Path:
        """Get the data directory path."""
        return self._data_dir

    @staticmethod
    def _create_data_directory(directory_name: str) -> Path:
        """
        Create a directory to save the OpenF1 data if it doesn't exist.

        Args:
            directory_name (str): Name of the directory to create.

        Returns:
            Path: Path object representing the data directory.
        """
        # Get the project root directory (3 levels up from current file)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        data_path = project_root / directory_name

        # Create directory if it doesn't exist
        data_path.mkdir(exist_ok=True)

        return data_path

    # i want to clear the data directory if it exists, so that we can start fresh each time
    def clear_data_directory(self) -> None:
        """
        Clear the data directory by removing all files in it.
        """
        if self._data_dir.exists() and self._data_dir.is_dir():
            for file in self._data_dir.iterdir():
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Error removing file {file}: {e}")
        else:
            print(f"Data directory {self._data_dir} does not exist or is not a directory.")

    @staticmethod
    def _fetch_csv_data(url: str) -> Optional[str]:
        """
        Fetch CSV data from the OpenF1 API.

        Args:
            url (str): The URL to fetch data from.

        Returns:
            Optional[str]: The CSV data as a string or None if an error occurred.
        """
        try:
            with urlopen(url) as response:
                csv_data = response.read().decode('utf-8')
                return csv_data
        except (URLError, HTTPError) as e:
            print(f"Network error fetching data from {url}: {e}")
            return None
        except UnicodeDecodeError as e:
            print(f"Encoding error for {url}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error fetching data from {url}: {e}")
            return None

    @staticmethod
    def _fetch_json_data(url: str) -> Optional[str]:
        """
        Fetch JSON data from the OpenF1 API.

        Args:
            url (str): The URL to fetch data from.

        Returns:
            Optional[str]: The JSON data as a string or None if an error occurred.
        """
        try:
            with urlopen(url) as response:
                json_data = json.loads(response.read().decode('utf-8'))
                return json_data
        except (URLError, HTTPError) as e:
            print(f"Network error fetching data from {url}: {e}")
            return None
        except UnicodeDecodeError as e:
            print(f"Encoding error for {url}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error fetching data from {url}: {e}")
            return None

    def _save_csv_to_file(self, csv_data: str, filename: str) -> bool:
        """
        Save CSV data to a file.

        Args:
            csv_data (str): The CSV data to save.
            filename (str): The filename for the CSV file.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not csv_data.strip():
            print(f"No data to save for {filename}")
            return False

        file_path = self._data_dir / filename

        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(csv_data)

            print(f"Data saved to {file_path}")
            return True

        except (IOError, OSError) as e:
            print(f"File error saving {filename}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error saving {filename}: {e}")
            return False

    def _validate_session_type(self, session_type: str) -> None:
        """
        Validate the session type.

        Args:
            session_type (str): The session type to validate.

        Raises:
            ValueError: If the session type is invalid.
        """
        if session_type not in self.VALID_SESSION_TYPES:
            raise ValueError(
                f"Invalid session type '{session_type}'. "
                f"Valid types: {', '.join(self.VALID_SESSION_TYPES)}"
            )

    def _build_csv_url(self, endpoint: str, **params) -> str:
        """
        Build a URL for CSV data retrieval with the csv=true parameter.

        Args:
            endpoint (str): The API endpoint (e.g., 'meetings', 'sessions', 'laps').
            **params: Query parameters to include in the URL.

        Returns:
            str: The complete URL with csv=true parameter.
        """
        url = f"{self._base_url}{endpoint}?csv=true"

        for key, value in params.items():
            if value is not None:
                # URL encode spaces and special characters
                encoded_value = str(value).replace(' ', '%20')
                url += f"&{key}={encoded_value}"

        return url

    def get_current_session_id(self):
        """Fetch the current session ID from the OpenF1 API."""


    def get_meeting_data(self, year: int, as_csv: bool = True, save_to_file: bool = True) -> Optional[str]:
        """
        Fetch meeting data for a given year from the OpenF1 API.
        Saves to file_name 'openf1_meetings_{year}.csv'

        Args:
            as_csv: CSV = True, JSON = False
            year (int): The year to fetch meeting data for.
            save_to_file (bool): Whether to save the data to a CSV file.

        Returns:
            Optional[str]: The CSV data if save_to_file is False, None otherwise.
        """
        url = self._build_csv_url('meetings', year=year)
        if as_csv:
            csv_data = self._fetch_csv_data(url)
        else:
            csv_data = self._fetch_json_data(url)

        if csv_data is None:
            return None

        if save_to_file:
            if as_csv:
                filename = f"openf1_meetings_{year}.csv"
            else:
                filename = f"openf1_meetings_{year}.json"
            self._save_csv_to_file(csv_data, filename)
            return None

        return csv_data

    def get_session_data(self, meeting_key: int = None, session_type: str = None,
                         year: int = None, save_to_file: bool = True) -> Optional[str]:
        """
        Fetch session data from the OpenF1 API.

        Args:
            meeting_key (int, optional): The meeting key to fetch session data for.
            session_type (str, optional): The type of session to fetch.
            year (int, optional): The year to fetch sessions for.
            save_to_file (bool): Whether to save the data to a CSV file.

        Returns:
            Optional[str]: The CSV data if save_to_file is False, None otherwise.

        Raises:
            ValueError: If session_type is provided but invalid, or if no parameters are provided.
        """
        if meeting_key is None and session_type is None and year is None:
            raise ValueError("At least one of meeting_key, session_type, or year must be provided")

        if session_type is not None:
            self._validate_session_type(session_type)

        # Build parameters dictionary
        params = {}
        if meeting_key is not None:
            params['meeting_key'] = meeting_key
        if session_type is not None:
            params['session_type'] = session_type
        if year is not None:
            params['year'] = year

        url = self._build_csv_url('sessions', **params)
        csv_data = self._fetch_csv_data(url)

        if csv_data is None:
            return None

        if save_to_file:
            # Create a descriptive filename
            filename_parts = ['openf1_sessions']
            if year:
                filename_parts.append(str(year))
            if meeting_key:
                filename_parts.append(str(meeting_key))
            if session_type:
                filename_parts.append(session_type.replace(' ', '_'))

            filename = '_'.join(filename_parts) + '.csv'
            self._save_csv_to_file(csv_data, filename)
            return None

        return csv_data

    def get_lap_data(self, session_key: int = None, driver_number: int = None,
                     save_to_file: bool = True) -> Optional[str]:
        """
        Fetch lap data from the OpenF1 API.

        Args:
            session_key (int, optional): The session key to fetch lap data for.
            driver_number (int, optional): The driver number to fetch lap data for.
            save_to_file (bool): Whether to save the data to a CSV file.

        Returns:
            Optional[str]: The CSV data if save_to_file is False, None otherwise.

        Raises:
            ValueError: If no parameters are provided.
        """
        if session_key is None and driver_number is None:
            raise ValueError("At least one of session_key or driver_number must be provided")

        # Build parameters dictionary
        params = {}
        if session_key is not None:
            params['session_key'] = session_key
        if driver_number is not None:
            params['driver_number'] = driver_number

        url = self._build_csv_url('laps', **params)
        csv_data = self._fetch_csv_data(url)

        if csv_data is None:
            return None

        if save_to_file:
            # Create a descriptive filename
            filename_parts = ['openf1_laps']
            if session_key:
                filename_parts.append(f"session_{session_key}")
            if driver_number:
                filename_parts.append(f"driver_{driver_number}")

            filename = '_'.join(filename_parts) + '.csv'
            self._save_csv_to_file(csv_data, filename)
            return None

        return csv_data

    def get_driver_data(self, driver_number: int = None, session_key: int = None,
                        save_to_file: bool = True) -> Optional[str]:
        """
        Fetch driver data from the OpenF1 API.

        Args:
            driver_number (int, optional): The driver number to fetch data for.
            session_key (int, optional): The session key to fetch driver data for.
            save_to_file (bool): Whether to save the data to a CSV file.

        Returns:
            Optional[str]: The CSV data if save_to_file is False, None otherwise.
        """
        # Build parameters dictionary
        params = {}
        if driver_number is not None:
            params['driver_number'] = driver_number
        if session_key is not None:
            params['session_key'] = session_key

        url = self._build_csv_url('drivers', **params)
        csv_data = self._fetch_csv_data(url)

        if csv_data is None:
            return None

        if save_to_file:
            # Create a descriptive filename
            filename_parts = ['openf1_drivers']
            if driver_number:
                filename_parts.append(f"driver_{driver_number}")
            if session_key:
                filename_parts.append(f"session_{session_key}")

            filename = '_'.join(filename_parts) + '.csv'
            self._save_csv_to_file(csv_data, filename)
            return None

        return csv_data

    def get_position_data(self, session_key: int = None, driver_number: int = None,
                          save_to_file: bool = True) -> Optional[str]:
        """
        Fetch position data from the OpenF1 API.

        Args:
            session_key (int, optional): The session key to fetch position data for.
            driver_number (int, optional): The driver number to fetch position data for.
            save_to_file (bool): Whether to save the data to a CSV file.

        Returns:
            Optional[str]: The CSV data if save_to_file is False, None otherwise.
        """
        # Build parameters dictionary
        params = {}
        if session_key is not None:
            params['session_key'] = session_key
        if driver_number is not None:
            params['driver_number'] = driver_number

        url = self._build_csv_url('position', **params)
        csv_data = self._fetch_csv_data(url)

        if csv_data is None:
            return None

        if save_to_file:
            # Create a descriptive filename
            filename_parts = ['openf1_positions']
            if session_key:
                filename_parts.append(f"session_{session_key}")
            if driver_number:
                filename_parts.append(f"driver_{driver_number}")

            filename = '_'.join(filename_parts) + '.csv'
            self._save_csv_to_file(csv_data, filename)
            return None

        return csv_data


def main():
    """Example usage of the OpenF1DataFetcher class."""
    try:
        # Initialize the fetcher
        fetcher = OpenF1DataFetcher()

        # Fetch meeting data for 2024
        print("Fetching meeting data for 2024...")
        fetcher.get_meeting_data(2024)

        # Fetch all sessions for 2023
        print("Fetching all sessions for 2023...")
        fetcher.get_session_data(year=2023)

        # Fetch race session data for a specific meeting
        print("Fetching race session data for meeting 1254...")
        fetcher.get_session_data(meeting_key=1254, session_type="Race")

        # Fetch lap data for a specific session
        print("Fetching lap data for session 9693...")
        fetcher.get_lap_data(session_key=9693)

        # Fetch driver data for a specific session
        print("Fetching driver data for session 9693...")
        fetcher.get_driver_data(session_key=9693)

        # Fetch position data for a specific driver in a session
        print("Fetching position data for driver 1 in session 9693...")
        fetcher.get_position_data(session_key=9693, driver_number=1)

        print("Data fetching completed successfully!")

    except ValueError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    fetcher = OpenF1DataFetcher()
    fetcher.get_meeting_data(year=2025)

    #fetcher.clear_data_directory()  # Clear the data directory before fetching new data
    #main()