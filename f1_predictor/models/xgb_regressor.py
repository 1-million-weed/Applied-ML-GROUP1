from f1_predictor.models.model import Model
from xgboost import XGBRegressor as WrappedXGBRegressor
from typing import Tuple
import matplotlib.pyplot as plt
from xgboost import plot_importance
from typing import Dict

import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


class XGBRegressor(Model):
    """
    XGBoost wrapper for regression with parameter 
    validation, evaluatiopn, and feature importance plotting.
    """
    def __init__(self,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 gamma: float = 0.01,
                 ) -> None:
        """
        Constructor method to initialize the XGBoost regressor model 
        with configurable hyperparameters, defined in the scikit-learn library.
        (   We did not like how XGboost handles error messages, so we
        decided to reimplement checking for parameter values.)

        :param max_depth: Maximum tree depth for baase learners.
        :param learning_rate: boosting learning rate.
        :param n_estimators: number of estimators/boosting rounds.
        :param gamma: minimum loss reduction requied for partition.
        """
        max_depth, learning_rate, n_estimators, gamma = \
            self._validate_parameters(max_depth, learning_rate, n_estimators,
                                      gamma)
        self._model = WrappedXGBRegressor(max_depth=max_depth,
                                          learning_rate=learning_rate,
                                          n_estimators=n_estimators,
                                          gamma=gamma)
        super().__init__(type="regression")

    def _validate_parameters(self,
                             max_depth: int,
                             learning_rate: float,
                             n_estimators: int,
                             gamma: float
                             ) -> Tuple[int, float, int, float]:
        """
        Validates hyperparameters and apply default values if valid for the model.
        Replaces every wrong parameter with its default value, 
        while informing the user of the change.

        :param max_depth: Intended tree depth.
        :param learning_rate: Step size shrinkage.
        :param n_estimators: Number of trees.
        :param gamma: Minimum loss reduction threshold.
        :return: Tuple of validated parameters.
        """
        if not isinstance(max_depth, int):
            print("Max depth must be an integer. Setting to default value 6")
            max_depth = 6
        if not isinstance(learning_rate, float):
            print("Learning rate must be a float. "
                  "Setting to defaul value 0.1")
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
        by applying the xgboost method to fit the model on training data.

        :param observations: Feature matrix.
        :param ground_truth: Regression target values.
        """
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "booster": self._model.get_booster(),
        }

    def predict(self, observations: np.ndarray, round:bool =True) -> np.ndarray:
        """
        Make predictions for the target value based on the observations
        by applying the xgboost method .predict on the input data.

        :param observations: Feature matrix for prediction.
        :param round: Whether to round the predictions to the nearest integer.
        :return: Predicted target values
        :return_type: np.ndarray
        """
        predictions = self._model.predict(observations)
        if round:
            return np.round(predictions).astype(int)
        return predictions
    
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

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray, show_plots: bool = True) -> Dict[str, float]:
        """
        Evaluate the model on the test data and print the mse.

        :param x_test: Test input features.
        :type x_test: np.array
        :param y_test: Test target values.
        :type y_test: np.array
        :param show_plots: Whether to display plots (confusion matrix and feature importance).
        :type show_plots: bool
        """
        y_pred = self.predict(x_test, round=False)
        mse = np.mean((y_test - y_pred) ** 2)
        print("Sample predictions:")
        print(y_pred[:5])
        print("Sample ground truth:")
        print(y_test[:5])
        
        # Get rounded predictions for confusion matrix
        y_pred_rounded = self.predict(x_test, round=True)
        if show_plots:
            self.plot_confusion_matrix(y_test, y_pred_rounded)

        print(f"Mean Squared Error: {mse}")
        return {
            "mse": mse,
        }
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot the confusion matrix for the model predictions.
        
        Args:
            y_true: True labels as a numpy array.
            y_pred: Predicted labels as a numpy array.
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()