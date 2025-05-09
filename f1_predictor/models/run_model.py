from model_manager import Modelmanager

class ModelRunner:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.available_models = ["RandomForestClassifier", "XGBRegressor", "MultilayerPerceptron"]
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} is not available. Available models are: {self.available_models}")
        else:
            model_manager = Modelmanager(model_name)
            self.model = model_manager.load_model()

    def predict(self, observations):
        """
        Predict using the loaded model.
        :param observations: The observations to predict on.
        :return: The predictions.
        """
        return self.model.predict(observations)
        