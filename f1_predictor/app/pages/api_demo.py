from ..page import Page
import streamlit as st
import logging
import httpx

class ApiDemoPage(Page):
    def __init__(self):
        super().__init__("Api Demo Page")
        self.logger = logging.getLogger(__name__)
        self.meeting_dict = self._get_all_meetings()

    def display(self):
        st.title("Api Demo Page")
        st.write("This page demonstrates the basic functionality of the F1 Predictor api.")

        # Fetch available years dynamically
        years = [2025]  # Replace with dynamic fetching if needed
        year = st.selectbox("Select Year", years, index=0)

        # Check if meetings are available
        if not self.meeting_dict:
            st.error("No meetings available. Please check the API or try again later.")
            self.logger.error("Meeting data is empty. Unable to proceed.")
            return

        # Create a dictionary for meetings
        if isinstance(self.meeting_dict, list) and all(isinstance(meeting, dict) for meeting in self.meeting_dict):
            meeting_dict = {meeting["name"]: meeting["id"] for meeting in self.meeting_dict}
        else:
            raise TypeError("self.meeting_dict must be a list of dictionaries.")

        # Use the dictionary keys for the select box and retrieve the selected value
        selected_meeting_name = st.selectbox("Select Meeting", list(meeting_dict.keys()), index=0)
        meeting = meeting_dict[selected_meeting_name]

        self.logger.info(f"Selected year: {year}, meeting: {meeting}")

        laps = st.slider(
            "Select Number of Laps",
            min_value=1,
            max_value=self._get_laps_for_meeting(meeting),  # Adjust max based on meeting
            value=10,
            step=1
        )

        # Add Predict button
        if st.button("Predict"):
            response = self._get_meeting_data(laps, meeting)
            if response and response.status_code == 200:
                data = response.json()
                predictions = data.get("predictions", [])
                race_info = data.get("race_info", {})
                metadata = data.get("metadata", {})

                # Display predictions in a table
                st.subheader("Predictions")
                if predictions:
                    st.table(predictions)
                else:
                    st.write("No predictions available.")

                # Display race information
                st.subheader("Race Information")
                st.write(f"Meeting Name: {race_info.get('meeting_name', 'N/A')}")
                st.write(f"Current Lap: {race_info.get('current_lap', 'N/A')}")
                st.write(f"Total Laps: {race_info.get('total_laps', self._get_laps_for_meeting(meeting))}")

                # Display metadata
                st.subheader("Metadata")
                st.write(f"Prediction Timestamp: {metadata.get('prediction_timestamp', 'N/A')}")
                st.write(f"Model Version: {metadata.get('model_version', 'N/A')}")
            else:
                st.error("Failed to fetch prediction data.")

    def _get_meeting_data(self, lap, meeting):
        try:
            return httpx.post('http://localhost:8000/predict', json={
                "current_lap": lap,
                "meeting": meeting
            })
        except httpx.RequestError as e:
            self.logger.error(f"Failed to fetch prediction data: {e}")
            return None

    def _get_all_meetings(self):
        try:
            response = httpx.get('http://localhost:8000/meetings')
            response.raise_for_status()
            meetings_dict = response.json().get("meetings", {})
            if not meetings_dict:
                self.logger.error("No meetings found in API response.")
                return []
            meetings = [{"id": key, "name": value} for key, value in meetings_dict.items()]
            return meetings
        except httpx.RequestError as e:
            self.logger.error(f"Failed to fetch meetings: {e}")
            return []

    def _get_laps_for_meeting(self, meeting: int = 1):
        try:
            response = httpx.get(f'http://localhost:8000/meetings/{meeting}/max-laps')
            response.raise_for_status()
            data = response.json()
            max_laps = data.get('max_laps', 1)
            if not isinstance(max_laps, (int, float)) or max_laps <= 0:
                self.logger.error(f"Invalid max laps value received for meeting {meeting}: {max_laps}")
                return 1
            return int(max_laps)
        except httpx.RequestError as e:
            self.logger.error(f"Failed to fetch max laps for meeting {meeting}: {e}")
            return 1
        except ValueError as e:
            self.logger.error(f"Error parsing response for meeting {meeting}: {e}")
            return 1    