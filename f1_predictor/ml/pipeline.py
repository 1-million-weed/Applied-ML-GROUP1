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
    """The pipeline of the model for Formula 1 predictions.

    This includes feature generation, model training, evaluation, and deployment
    through an API.
    """
    def __init__(self, 
        model_config,
        dataset_config,
        training_config,
        test_config,
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

    def _get_model_manager(self, model_name) -> Modelmanager:
        """model manager instance if the model available.

        :param model_name: The name of the model.
        :type model_name: str
        :raises ValueError: If the model available.
        :return: Modelmanager instance for the given model.
        :rtype: Modelmanager
        """    
        available_models = ["RandomForestClassifier", "XGBClassifier", "XGBRegressor", "MultiLayerPerceptron"]
        if model_name not in available_models:
            raise ValueError(f"Model {model_name} is not available. Available models are: {available_models}")
        else:
            return Modelmanager(model_name)


    def run(self) -> None:
        """
        Run the full pipeline based on configuration.

        Executes steps for feature generation, training, testing, and inference.
        """    
        if self.gen_features:
            self.make_features()
        
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
        model = self.model_manager.load_model()
        if self.api:
            API(model)
        
        if self.streamlit:
            app = HomePage()
            app.display()

        
