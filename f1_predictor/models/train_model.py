from model_manager import Modelmanager
from dataset_manager import DatasetManager
from f1_predictor.models.model_generation.xgboost_regressor import XGBRegressor
from f1_predictor.models.model_generation.random_forest_model import RandomForest
from f1_predictor.models.model_generation.multi_layer_perceptron import MultiLayerPerceptron

class Modeltrainer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.available_models = ["RadnomForestClassifier", "XGBRegressor", "MultilayerPerceptron"]
        dataset_manager = DatasetManager()
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} is not available. Available models are: {self.available_models}")
        else:
            self.model_manager = Modelmanager(model_name)
        
        if model_name == "XGBRegressor":
            self.model = XGBRegressor()
        elif model_name == "RadnomForestClassifier":
            self.model = RandomForest()
        elif model_name == "MultilayerPerceptron":
            self.model = MultiLayerPerceptron()
        self.training_data = dataset_manager.get_training_data()

        self.training_features = self.training_data.drop(columns=['finishing_position', 'race_id', 'driver_id'])
        self.ground_truth = self.training_data['finishing_position']

    def train(self):
        """
        Train the model using the training data.
        """
        self.model.fit()
        self.model_manager.save_model(self.model)
        if hasattr(self.model, 'plot_feature_importance'):
            self.model.plot_feature_importance(feature_names=self.training_features.columns, max_num_features=10)
        elif hasattr(self.model, 'plot_loss'):
            self.model.plot_loss()
        else:
            print("Model does not have a plot method.")



        
