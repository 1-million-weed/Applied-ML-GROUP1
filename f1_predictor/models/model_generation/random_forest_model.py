from f1_predictor.models.model_generation.model import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score


class RandomForest(Model):
    """
    RandomForest model for regression tasks with additional features
    like plotting graphs for feature importance and actual vs. predicted values.

    Attributes:
        n_trees (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        _parameters (dict): Hyperparameters for the RandomForestRegressor.
        _model (RandomForestRegressor): The RandomForestRegressor instance.
    """

    def __init__(self, n_trees: int = 100, max_depth: int = None, min_samples_split: int = 2) -> None:
        """
        Initializes the RandomForest model with given hyperparameters.

        Args:
            n_trees (int): Number of trees in the forest.
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum number of samples required to split an internal node.
        """
        super().__init__(type="classification")
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._parameters = {
            "n_estimators": self.n_trees,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
        }
        self._model = RandomForestClassifier(**self._parameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the RandomForest model to the provided data.

        Args:
            observations (np.ndarray): Training data.
            ground_truth (np.ndarray): Target values.
        """
        self._model.fit(observations, ground_truth)
        self._parameters.update(self._model.get_params())

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the provided data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted target values.
        """
        return self._model.predict(x)

    def evaluate(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluates the model and returns performance metrics.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Training target values.
            X_test (np.ndarray): Test data.
            y_test (np.ndarray): Test target values.

        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        y_train_pred = self.predict(x_train)
        y_test_pred = self.predict(x_test)

        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "test_mse": mean_squared_error(y_test, y_test_pred),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred),
        }

        return metrics

    def plot_feature_importance(self, feature_names: list) -> None:
        """
        Plots the top 10 feature importances.

        Args:
            feature_names (list): List of feature names.
        """
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self._model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nFeature Importance:")
        print(feature_importance.head(10))

        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()
        plt.show()

    def plot_actual_vs_predicted(self, y_test: np.ndarray, y_test_pred: np.ndarray) -> None:
        """
        Plots actual vs predicted values.

        Args:
            y_test (np.ndarray): Actual target values.
            y_test_pred (np.ndarray): Predicted target values.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        plt.tight_layout()
        plt.show()