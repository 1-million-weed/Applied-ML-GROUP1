import streamlit as st
from fastapi.testclient import TestClient
from ...ml.api import API


class ApiDemo:
    def __init__(self):
        self.title = "API Demo"
        self.description = "This is a demo of the API."
        self.api_url = "/predict"

    def run(self):
        st.title(self.title)
        st.write(self.description)
        #example data
        # Input fields for the API
        input_data = st.text_input("Input Data", "Enter your data here")
        
        if st.button("Submit"):
            response = self.call_api(input_data)
            st.write("Response from API:", response)

    def call_api(self, input_data):
        # Create a TestClient instance
        client = TestClient(self.api_url)
        
        # Prepare the request payload
        payload = {
            "input_data": input_data,
        }
        
        # Make the API call
        response = client.post(self.api_url, json=payload)
        
        # Check if the response is successful
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "API call failed", "status_code": response.status_code}
    
