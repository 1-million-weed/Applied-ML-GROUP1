from .model_manager import Modelmanager
from .dataset_manager import DatasetManager

class ModelTester:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.available_models = ["RandomForestClassifier", "XGBRegressor", "MultilayerPerceptron"]
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} is not available. Available models are: {self.available_models}")
        else:
            model_manager = Modelmanager(model_name)
            self.model = model_manager.load_model()
        dataset_manager = DatasetManager()
        self.training_data = dataset_manager.get_training_data()
        self.x_train = self.training_data.drop(columns=['finishing_position', 'race_id', 'driver_id', 'lap'])
        self.y_train = self.training_data['finishing_position']
        self.validation_data = dataset_manager.get_validation_data()
        self.x_val = self.validation_data.drop(columns=['finishing_position', 'race_id', 'driver_id', 'lap'])
        self.y_val = self.validation_data['finishing_position']

    def test(self):
        """
        Test the model using the validation data.
        """
        print(self.model.evaluate(x_test=self.x_val, y_test=self.y_val))