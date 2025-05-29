from ..page import Page
import streamlit as st
import httpx

class ApiDemoPage(Page):
    def __init__(self):
        super().__init__("API Demo")

    def display(self):
        st.title("API Demo")
        st.write("This page demonstrates how to use the API for predictions.")
        
        # Upload CSV files
        laptimes = st.file_uploader("Upload Laptimes.csv", type=["csv"])
        constructors = st.file_uploader("Upload Constructors.csv", type=["csv"])
        qualifying = st.file_uploader("Upload Qualifying.csv", type=["csv"])

        feature_values = {
            "laptimes": laptimes,
            "constructors": constructors,
            "qualifying": qualifying
        }
        
        if st.button("Call API"):
            self.call_api(feature_values)

    def call_api(self, feature_values):
        # Only send files that have been uploaded (are not None)
        files = {}
        for key, file_obj in feature_values.items():
            if file_obj is not None:
                files[key] = (file_obj.name, file_obj, "text/csv")
                
        if not files:
            st.error("No CSV files were uploaded.")
            return

        st.write("Uploading the following files:")
        st.write(list(files.keys()))
        
        # Use httpx to send a POST request with multiple CSV files
        try:
            response = httpx.post("http://localhost:8000/predict", files=files)
            if response.status_code == 200:
                st.write("Files uploaded successfully!")
                st.json(response.json())
            else:
                st.error(f"API call failed with status code {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
