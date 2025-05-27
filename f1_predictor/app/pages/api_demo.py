from ..page import Page
import streamlit as st
import httpx

class ApiDemoPage(Page):
    def __init__(self, training_features: list = []):
        super().__init__("API Demo")
        self.training_features = training_features

    def display(self):
        st.title("API Demo")
        st.write("This page demonstrates how to use the API for predictions.")
        
        if not self.training_features:
            st.warning("No training features available. Please run the training pipeline first.")
            return
        st.write("Model Input Features:")
        st.write(self.training_features)

        #make sliding bar for each feature between 0 and 1
        feature_values = {}
        for feature in self.training_features:
            feature_values[feature] = st.slider(f"Select value for {feature}", 0.0, 1.0, 0.5)
        if st.button("Predict"):
            self.call_api(feature_values)

    def call_api(self, feature_values):
        # Placeholder for API call logic
        st.write("API called with the following values:")
        st.json(feature_values)
        # Here you would typically use httpx or requests to call your API
        # Example:
        response = httpx.post("http://localhost:8000/predict", json=feature_values)
        if response.status_code == 200:
            prediction = response.json()
            st.write("Prediction Result:")
            st.json(prediction)
        else:
            st.error(f"API call failed with status code {response.status_code}: {response.text}")