from .page import Page
from .pages.api_demo import ApiDemoPage
from .pages.dream_big_demo import DreamBigDemoPage
import streamlit as st

class HomePage(Page):
    def __init__(self, training_features: list):
        super().__init__("Home")
        self.pages = {
            "API Demo": ApiDemoPage(),
            "Dream Big Demo": DreamBigDemoPage()
        }

    def display_sidebar(self):
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", list(self.pages.keys()))
        self.pages[page].display()


    def display(self):
        st.title("Welcome to the F1 Predictor App")
        st.write("This app uses machine learning to predict Formula 1")
        self.display_sidebar()