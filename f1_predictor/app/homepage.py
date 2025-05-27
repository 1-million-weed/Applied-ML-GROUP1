import streamlit as st
import os

class HomePage:
    """
    Home page of the F1 Predictor App built with Streamlit.

    This class sets the page configuration and displays an introduction
    to the app in the main interface, guiding users to the sidebar navigation.
    """
    def __init__(self) -> None:
        """
        Constructor Method to initialize the home page by setting the Streamlit page configuration.
        """
        st.set_page_config(
        page_title="HomePage",
        page_icon="👋",
        )

    def display(self) -> None:
        """
        Renders the homepage content to the Streamlit interface.
        """
        st.sidebar.success("Select a page above.")
        st.title("Welcome to the F1 Predictor App")
        st.write(
            "This application predicts the finishing position of Formula 1 drivers based on various features."
        )
        st.write(
            "To get started, please select an option from the sidebar."
        )