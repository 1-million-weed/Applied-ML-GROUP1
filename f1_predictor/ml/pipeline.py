import logging

from .model_manager import Modelmanager
from .dataset_manager import DatasetManager
from ..features.make_features import FeatureGenerator
from ..features.elo_calculator import F1EloAnalyzer
from ..models.xgb_classifier import XGBClassifier
from ..models.xgb_regressor import XGBRegressor
from ..models.random_forest_model import RandomForest
from ..models.multi_layer_perceptron import MultiLayerPerceptron
from ..models.multi_layer_regression import MultiLayerRegression
from ..models.random_model import RandomModel
from ..data_aquisition.data_aquisition_pipeline import DataAquisitionPipeline
from .api import API
from ..app.homepage import HomePage
import pandas as pd
import uvicorn 
import threading

class Pipeline:
    """The pipeline of the model for Formula 1 predictions.

    This includes feature generation, model training, evaluation, and deployment
    through an API.
    """
    def __init__(self, 
        model_config,
        dataset_config,
        training_config,
        eval_config,
        inference_config,
    ) -> None:
        """
        Constructor Method to initialize the pipeline with all necessary configuration sections.

        :param model_config: Configuration dictionary for model architecture and name.
        :type model_config: dict
        :param dataset_config: Dataset settings, such as random state and test size.
        :type dataset_config: dict
        :param training_config: Training parameters like features and ground truth.
        :type training_config: dict
        :param test_config: Evaluation parameters including test enablement and plots.
        :type test_config: dict
        :param inference_config: Deployment flags for API and Streamlit inference.
        :type inference_config: dict
        """

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing F1 Prediction Pipeline.")

        self.model_config = model_config
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.eval_config = eval_config
        self.inference_config = inference_config

        self.dataset_manager = DatasetManager(self.training_config["training_features"])
        self.model_name = model_config['name']
        self.model_manager = self._get_model_manager(self.model_name)

        self.logger.info(f"Using model: {self.model_name}")

        self.train_plots = training_config['show_plot']
        self.test_plots = eval_config['show_plot']

        self.gen_features = dataset_config['generate']
        self.gen_2025 = dataset_config['get_2025_data']
        self.calculte_elo = dataset_config['calculate_elo']
        self.train_model = self.training_config['enabled']
        self.test_model = eval_config['enabled']
        self.run_model = inference_config['enabled']

        self.api = inference_config['api']
        self.streamlit = inference_config['streamlit']

        self.logger.info("Pipeline initialized with configuration: %s", self._get_log_context())

    def _get_log_context(self):
        return {
            "model_name": self.model_name,
            "dataset_features": self.training_config["training_features"],
            "ground_truth": self.training_config['ground_truth'],
            "random_state": self.dataset_config['random_state'],
            "test_size": self.dataset_config['test_size'],
            "empty_folder": self.dataset_config['empty_folder'],
            "train_enabled": self.train_model,
            "test_enabled": self.test_model,
            "inference_enabled": self.run_model,
            "api_enabled": self.api,
            "streamlit_enabled": self.streamlit,
            "elo_calculator_enabled": self.calculte_elo,
            "gen_features_enabled": self.gen_features,
            "gen_2025_data_enabled": self.gen_2025,
            "train_plots_enabled": self.train_plots,
            "test_plots_enabled": self.test_plots
        }

    def _get_model_manager(self, model_name):
        available_models = ["RandomForestClassifier", "XGBClassifier", "XGBRegressor", "MultiLayerPerceptron", "MultiLayerRegression", 'RandomModel']
        if model_name not in available_models:
            self.logger.error(f"Model {model_name} is not available. Available models are: {available_models}")
            raise ValueError(f"Model {model_name} is not available. Available models are: {available_models}")
        else:
            self.logger.info(f"Model manager initialized for {model_name}")
            return Modelmanager(model_name)


    def run(self):
        if self.calculte_elo:
            self.elo_calculator()
        if self.gen_features:
            self.make_features()
        if self.gen_2025:
            self.gen_2025_data()

        if self.train_model:
            self.train()

        if self.test_model:
            self.test()
        
        if self.run_model:
            self.inference()

    def train(self) -> None:
        """Train the model saves it.

        Also includes to visualize feature importance or training loss.
        """
        if self.model_name == "XGBClassifier":
            model = XGBClassifier()
        elif self.model_name == "XGBRegressor":
            model = XGBRegressor()
        elif self.model_name == "RandomForestClassifier":
            model = RandomForest()
        elif self.model_name == "MultiLayerPerceptron":
            model = MultiLayerPerceptron(input_shape=len(self.training_config["training_features"]))
        elif self.model_name == "MultiLayerRegression":
            model = MultiLayerRegression(input_shape=len(self.training_config["training_features"]))
        elif self.model_name == "RandomModel":
            model = RandomModel()

        model.fit(*self._load_training_data())
        self.model_manager.save_model(model)
        if self.train_plots:
            if hasattr(model, 'plot_feature_importance'):
                model.plot_feature_importance(feature_names=self.training_config["training_features"])
            elif hasattr(model, 'plot_loss'):
                model.plot_loss()
            else:
                print("Model does not have a plot method.")


    def test(self) -> None:
        """
        Evaluates the trained model on validation data and return metrics
        """    
        model = self.model_manager.load_model()
        metrics = model.evaluate(*self._load_validation_data())

    def make_features(self) -> None:
        """
        Generate features and split them into training and test datasets.
        """   
        random_seed = self.dataset_config['random_state']
        test_size = self.dataset_config['test_size']
        empty_folder = self.dataset_config['empty_folder']
        feature_generator = FeatureGenerator(random_seed, test_size, empty_folder)
        feature_generator.run()

    def _load_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load training data and extract input features and labels.

        :return: Tuple (x_train, y_train) of features and ground truth.
        :rtype: tuple[pd.DataFrame, pd.Series]
        """    
        training_data = self.dataset_manager.get_training_data()
        x_train = training_data[self.training_config["training_features"]]
        y_train = training_data[self.training_config['ground_truth']]
        return x_train, y_train

    def _load_validation_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load validation data and extract input features and labels.

        :return: Tuple (x_val, y_val) of features and ground truth.
        :rtype: tuple[pd.DataFrame, pd.Series]
        """    
        validation_data = self.dataset_manager.get_validation_data()
        x_val = validation_data[self.training_config["training_features"]]
        y_val = validation_data[self.training_config['ground_truth']]
        return x_val, y_val

    def inference(self) -> None:
        """
        Launch inference via API or Streamlit.
        """   
        if self.api:
            api_thread = threading.Thread(target=self.start_api)
            api_thread.start()
        
        if self.streamlit:
            app = HomePage(self.training_config["training_features"].copy())
            app.display()

    def elo_calculator(self):
        analyzer = F1EloAnalyzer()
    
        # Run complete analysis
        results = analyzer.run_complete_analysis()
        
        # Display top drivers
        print("Top 30 Drivers by ELO:")
        print(results['top_drivers'])
        
        # Create visualizations
        analyzer.plot_driver_race_ratio()
        analyzer.plot_yearly_elo_distribution()
        analyzer.plot_driver_vs_teammates('max_verstappen')
        
        # Access specific results
        elo_history = results['elo_history']
        print(f"\nTotal races processed: {elo_history['raceId'].nunique()}")
        print(f"Total drivers: {elo_history['driverRef'].nunique()}")
        analyzer.save_results()
        
    def start_api(self):
        model = self.model_manager.load_model()
        api_instance = API(model)
        uvicorn.run(api_instance.app, host="0.0.0.0", port=8000)

    def gen_2025_data(self):
        """
        Generate data for the 2025 season.
        """
        pipeline_2025 = DataAquisitionPipeline(current_year=2025)
        pipeline_2025.run()
