from .model_manager import Modelmanager
from .dataset_manager import DatasetManager
from f1_predictor.models.model_generation.xgb_classifier import XGBClassifier
from f1_predictor.models.model_generation.xgb_regressor import XGBRegressor
from f1_predictor.models.model_generation.random_forest_model import RandomForest
from f1_predictor.models.model_generation.multi_layer_perceptron import MultiLayerPerceptron

class Modeltrainer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.available_models = ["RandomForestClassifier", "XGBClassifier", "XGBRegressor", "MultiLayerPerceptron"]
        dataset_manager = DatasetManager()
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} is not available. Available models are: {self.available_models}")
        else:
            self.model_manager = Modelmanager(model_name)
        
        self.training_data = dataset_manager.get_training_data()
        self.training_features = self.training_data.drop(columns=['finishing_position', 'race_id', 'driver_id', 'lap'])
        self.ground_truth = self.training_data['finishing_position']

        if model_name == "XGBClassifier":
            self.model = XGBClassifier()
        elif model_name == "XGBRegressor":
            self.model = XGBRegressor()
        elif model_name == "RandomForestClassifier":
            self.model = RandomForest()
        elif model_name == "MultiLayerPerceptron":
            self.model = MultiLayerPerceptron(input_shape=self.training_features.shape[1])

    def train(self):
        """
        Train the model using the training data.
        """
        self.model.fit(self.training_features, self.ground_truth)
        self.model_manager.save_model(self.model)
        if hasattr(self.model, 'plot_feature_importance'):
            self.model.plot_feature_importance(feature_names=self.training_features.columns)
        elif hasattr(self.model, 'plot_loss'):
            self.model.plot_loss()
        else:
            print("Model does not have a plot method.")



        
