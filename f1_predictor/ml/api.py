from fastapi import FastAPI, HTTPException
from .dataset_manager import DatasetManager

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

        @self.app.get("/predict")
        async def predict(data: dict):
            data = data.get("input_data")
            if not self.dataset_manager.validate_data(data):
                raise HTTPException(status_code=400, detail="Invalid data format")
            prediction = self.model.predict(data)
            return {"prediction": prediction}