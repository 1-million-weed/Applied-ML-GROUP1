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

    def __init__(self, n_trees: int = 100, max_depth: int = None, min_samples_split: int = 2, max_leaf_nodes: int = 50) -> None:
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
        self.max_leaf_nodes = max_leaf_nodes
        self._parameters = {
            "n_estimators": self.n_trees,
            "max_depth": self.max_depth,
            'max_leaf_nodes': self.max_leaf_nodes,
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

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluates the model on the test data.

        Args:
            x_test (np.ndarray): Test input features.
            y_test (np.ndarray): Test target values.

        Returns:
            dict: Evaluation metrics including MSE and R2 score.
        """
        y_test_pred = self.predict(x_test)
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R2 Score: {r2}")

        return {"mse": mse, "r2": r2}



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

