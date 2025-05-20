from .model_manager import Modelmanager
from .dataset_manager import DatasetManager
from ..features.make_features import FeatureGenerator
from ..models.xgb_classifier import XGBClassifier
from ..models.xgb_regressor import XGBRegressor
from ..models.random_forest_model import RandomForest
from ..models.multi_layer_perceptron import MultiLayerPerceptron
from .api import API
from ..app.homepage import HomePage


class Pipeline:
    def __init__(self, 
        model_config,
        dataset_config,
        training_config,
        test_config,
        inference_config,
    ):
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.test_config = test_config
        self.inference_config = inference_config

        self.dataset_manager = DatasetManager()
        self.model_name = model_config['name']
        self.model_manager = self._get_model_manager(self.model_name)

        self.train_plots = training_config['show_plot']
        self.test_plots = test_config['show_plot']

        self.gen_features = dataset_config['generate']
        self.train_model = self.training_config['enabled']
        self.test_model = test_config['enabled']
        self.run_model = inference_config['enabled']

        self.api = inference_config['api']
        self.streamlit = inference_config['streamlit']

    def _get_model_manager(self, model_name):
        available_models = ["RandomForestClassifier", "XGBClassifier", "XGBRegressor", "MultiLayerPerceptron"]
        if model_name not in available_models:
            raise ValueError(f"Model {model_name} is not available. Available models are: {available_models}")
        else:
            return Modelmanager(model_name)


    def run(self):
        if self.gen_features:
            self.make_features()
        
        if self.train_model:
            self.train()

        if self.test_model:
            self.test()
        
        if self.run_model:
            self.inference()

    def train(self):
        if self.model_name == "XGBClassifier":
            model = XGBClassifier()
        elif self.model_name == "XGBRegressor":
            model = XGBRegressor()
        elif self.model_name == "RandomForestClassifier":
            model = RandomForest()
        elif self.model_name == "MultiLayerPerceptron":
            model = MultiLayerPerceptron(input_shape=len(self.training_config["training_features"]))
        model.fit(*self._load_training_data())
        self.model_manager.save_model(model)
        if self.train_plots:
            if hasattr(model, 'plot_feature_importance'):
                model.plot_feature_importance(feature_names=self.training_config["training_features"])
            elif hasattr(model, 'plot_loss'):
                model.plot_loss()
            else:
                print("Model does not have a plot method.")


    def test(self):
        model = self.model_manager.load_model()
        metrics = model.evaluate(*self._load_validation_data())

    def make_features(self):
        random_seed = self.dataset_config['random_state']
        test_size = self.dataset_config['test_size']
        empty_folder = self.dataset_config['empty_folder']
        feature_generator = FeatureGenerator(random_seed, test_size, empty_folder)
        feature_generator.run()

    def _load_training_data(self):
        training_data = self.dataset_manager.get_training_data()
        x_train = training_data[self.training_config["training_features"]]
        y_train = training_data[self.training_config['ground_truth']]
        return x_train, y_train

    def _load_validation_data(self):
        validation_data = self.dataset_manager.get_validation_data()
        x_val = validation_data[self.training_config["training_features"]]
        y_val = validation_data[self.training_config['ground_truth']]
        return x_val, y_val

    def inference(self):
        model = self.model_manager.load_model()
        API(model)
        app = HomePage()
        app.display()

        
