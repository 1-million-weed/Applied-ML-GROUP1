import streamlit as st

# Abstract class for pages
class Page:
    def __init__(self, title):
        self.title = title

    def display(self):
        raise NotImplementedError("Each page must implement a display method.")
