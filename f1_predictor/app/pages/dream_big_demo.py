from ..page import Page
import streamlit as st
import logging
import httpx

class DreamBigDemoPage(Page):
    def __init__(self):
        super().__init__("Dream Big Demo")
        self.logger = logging.getLogger(__name__)

    def display(self):
        st.title("Dream Big Demo")
        st.write("This page demonstrates the basic functionality of the F1 Predictor api.")

        # Select year from dropdown (only 2025 is available)
        year = st.selectbox("Select Year", [2025], index=0)

        # Options 1-8 for meeting
        meeting = st.selectbox(
            "Select Meeting",
            [
                "Australian Grand Prix",
                "Bahrain Grand Prix",
                "Saudi Arabian Grand Prix",
                "Azerbaijan Grand Prix",
                "Miami Grand Prix",
                "Monaco Grand Prix",
                "Spanish Grand Prix",
                "Canadian Grand Prix"
            ],
            index=0
        )

        self.logger.info(f"Selected year: {year}, meeting: {meeting}")

        # based on meeting, slider for number of laps (max meeting specific)

        laps = st.slider(
            "Select Number of Laps",
            min_value=1,
            max_value=100,  # TODO: Adjust this based on the meeting
            value=10,
            step=1
        )

    def _get_meeting_data(self, year, meeting):
        # This function should return the data for the selected meeting
        # For now, we will just return a placeholder
        return {
            "year": year,
            "meeting": meeting,
            "laps": 10  # Placeholder value
        }

    def _get_all_meetings(self, year):
        # This function should return all meetings for the selected year
        # For now, we will just return a placeholder list
        return [
            "Australian Grand Prix",
            "Bahrain Grand Prix",
            "Saudi Arabian Grand Prix",
            "Azerbaijan Grand Prix",
            "Miami Grand Prix",
            "Monaco Grand Prix",
            "Spanish Grand Prix",
            "Canadian Grand Prix"
        ]