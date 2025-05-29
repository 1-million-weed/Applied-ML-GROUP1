from fastapi import FastAPI, HTTPException
from .dataset_manager import DatasetManager
import pandas as pd
from .api_pipeline import ApiNormalizer

class API:
    """Class for the API interface.

    Initializes a API, wraps a trained model for inference with input data.
    """
    def __init__(self, model) -> None:
        """Initialize the API wrapper for the trained model.

        :param model: Trained model object with a `.predict()` method.
        :type model: object
        :raises HTTPException: If the input data format is invalid.
        :return: Dictionary with prediction results.
        :rtype: dict
        """    
        self.model = model
        self.dataset_manager = DatasetManager()
        self.app = FastAPI(
            title="F1 Predictor API",
            description="API for F1 position prediction using a Multi-Layer Perceptron model.",
            version="1.0.0",
        )

        @self.app.post("/predict_normalized")
        async def predict_normalized(data: dict):
            #alright so we are getting a dict with the name of each feature and its value
            """Endpoint for making predictions.
            :param data: Input data for prediction.
            :type data: dict
            :raises HTTPException: If the input data format is invalid.
            :return: Dictionary with prediction results.
            :rtype: dict
            """
            
            # Convert input data to a DataFrame
            try:
                input_df = pd.DataFrame.from_dict([data], orient='columns')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error converting input to DataFrame: {str(e)}")
            if not self.dataset_manager.validate_data(input_df):
                raise HTTPException(status_code=400, detail="Invalid data format")
            prediction = self.model.predict(input_df)
            if hasattr(prediction, "tolist"):
                prediction = prediction.tolist()
            return {"prediction": prediction}
        
        @self.app.post("/predict")
        async def predict(data: dict):
            """Endpoint for making predictions with unnormalized data.
            :param data: Input data for prediction.
            :type data: dict
            :raises HTTPException: If the input data format is invalid.
            :return: Dictionary with prediction results.
            :rtype: dict
            """
            # Convert input data to a DataFrame
            if not isinstance(data, dict):
                raise HTTPException(status_code=400, detail="Input data must be a dictionary.")
            try:
                normalizer = ApiNormalizer(data)
                data = normalizer.normalize()
                input_df = pd.DataFrame.from_dict([data], orient='columns')
                if not self.dataset_manager.validate_data(input_df):
                    raise HTTPException(status_code=400, detail="Invalid data format")
                prediction = self.model.predict(input_df)
                if hasattr(prediction, "tolist"):
                    prediction = prediction.tolist()
                return {"prediction": prediction}
        
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error normalizing input data: {str(e)}")
            