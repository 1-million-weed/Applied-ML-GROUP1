from f1_predictor.models.model_generation.model import Model
from xgboost import XGBClassifier as WrappedXGBClassifier
from typing import Tuple
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


class XGBClassifier(Model):
    """ XGBoost for classification wrapper """

    def __init__(self,
                 max_depth: int = 10,
                 learning_rate: float = 0.1,
                 n_estimators: int = 1000,
                 gamma: float = 5.0,
                 ) -> None:
        """
        Initialize the XGBoost model with various hyperparameters,
        as defined in the scikit-learn library.
        :param max_depth: Maximum depth
        :param learning_rate: Learning rate
        :param n_estimators: Number of estimators
        :param gamma: Minimum loss reduction
        We did not like how XGBoost handles error messages, so we
        decided to reimplement checking for parameter values.
        """
        max_depth, learning_rate, n_estimators, gamma = \
            self._validate_parameters(max_depth, learning_rate, n_estimators,
                                      gamma)
        self._model = WrappedXGBClassifier(max_depth=max_depth,
                                           learning_rate=learning_rate,
                                           n_estimators=n_estimators,
                                           gamma=gamma)
        super().__init__(type="classification")

    def _validate_parameters(self,
                             max_depth: int,
                             learning_rate: float,
                             n_estimators: int,
                             gamma: float
                             ) -> Tuple[int, float, int, float]:
        """
        Validates the parameters for the model.
        Replaces every wrong parameter with its default
        value while informing the user of the change.
        """
        if not isinstance(max_depth, int):
            print("Max depth must be an integer. Setting to default value 6")
            max_depth = 6
        if not isinstance(learning_rate, float):
            print("Learning rate must be a float. "
                  "Setting to default value 0.1")
            learning_rate = 0.1
        if not isinstance(n_estimators, int):
            print("Number of estimators must be an integer. "
                  "Setting to default value 100")
            n_estimators = 100
        if not isinstance(gamma, float):
            print("Minimum loss reduction 'gamma' must be a float. "
                  "Setting to default value 0.0")
            gamma = 0.0

        if learning_rate < 0.0 or learning_rate > 1.0:
            print("Learning rate must be positive and between [0.0, 1.0]. "
                  "Setting to default value 0.1")
            learning_rate = 0.1
        if max_depth < 0:
            print("Max depth must be positive. Setting to default value 6")
            max_depth = 6
        if n_estimators < 0:
            print("Number of estimators must be positive. "
                  "Setting to default value 100")
            n_estimators = 100
        if gamma < 0.0:
            print("Minimum loss reduction 'gamma' must be positive. "
                  "Setting to default value 0.0")
            gamma = 0.0

        return max_depth, learning_rate, n_estimators, gamma

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on the observations and labels (ground_truth)
        by applying the xgboost method .fit
        """

        ground_truth = ground_truth - ground_truth.min()
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "booster": self._model.get_booster(),
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions based on the observations
        by applying the xgboost method .predict
        """
        return self._model.predict(observations)
    
    def plot_feature_importance(self, feature_names: list, max_num_features: int = 10) -> None:
        """
        Plots the top N feature importances.
        :param feature_names: List of feature names.
        :param max_num_features: Maximum number of features to plot.
        """
        plt.figure(figsize=(10, 6))
        plot_importance(self._model, max_num_features=max_num_features)
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.show()

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model on the test data.
        :param x_test: Test input features as a numpy array.
        :param y_test: Test target values as a numpy array.
        :return: A dictionary containing evaluation metrics.
        """
        y_test = y_test - y_test.min()
        y_pred = self.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }