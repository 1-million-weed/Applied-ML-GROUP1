import streamlit as st
import os

class HomePage:
    def __init__(self):
        st.set_page_config(
        page_title="HomePage",
        page_icon="👋",
        )

    def display(self):
        st.sidebar.success("Select a page above.")
        st.title("Welcome to the F1 Predictor App")
        st.write(
            "This application predicts the finishing position of Formula 1 drivers based on various features."
        )
        st.write(
            "To get started, please select an option from the sidebar."
        )