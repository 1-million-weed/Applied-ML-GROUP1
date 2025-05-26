import streamlit as st
from fastapi.testclient import TestClient
from ...ml.api import API


class ApiDemo:
    """
    A demo interface for interacting with a FastAPI endpoint using Streamlit.

    :raises APIFailedError: raises an error if it has failed to connect to the API.
    """
    def __init__(self) -> None:
        """
        Constructor method to initialize the API demo with title, description, and path.
        """
        self.title = "API Demo"
        self.description = "This is a demo of the API."
        self.api_url = "/predict"

    def run(self) -> None:
        """
        Runs the api demo and renders the Streamlit user interface.
        """
        st.title(self.title)
        st.write(self.description)
        #example data
        # Input fields for the API
        input_data = st.text_input("Input Data", "Enter your data here")
        
        if st.button("Submit"):
            response = self.call_api(input_data)
            st.write("Response from API:", response)

    def call_api(self, input_data) -> Dict[str, Any]:
        """
        Calls the api sends input data to the API and returns the response.

        :param input_data: String data to be sent as input to the prediction API.
        :type input_data: str
        :return: JSON-formatted API response containing prediction or error message.
        :rtype: Dict[str, Any]
        """    
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
    
