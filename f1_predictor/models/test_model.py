from model_manager import Modelmanager
from dataset_manager import DatasetManager

class ModelTester:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.available_models = ["RadnomForestClassifier", "XGBRegressor", "MultilayerPerceptron"]
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} is not available. Available models are: {self.available_models}")
        else:
            model_manager = Modelmanager(model_name)
            self.model = model_manager.load_model()
        dataset_manager = DatasetManager()
        self.validation_data = dataset_manager.get_validation_data()

    def test(self):
        """
        Test the model using the validation data.
        """
        pass