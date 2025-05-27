from f1_predictor.models.model import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict


class RandomForest(Model):
    """
    RandomForest model for regression tasks with additional features
    like plotting graphs for feature importance and actual vs. predicted values.

    :param n_trees: Number of trees in the forest.
    :type n_trees: int
    :param max_depth: Maximum depth of the tree.
    :type max_depth: Optional[int]
    :param min_samples_split: Minimum number of samples required to split an internal node.
    :type min_samples_split: int
    :param max_leaf_nodes: Maximum number of leaf nodes in the tree.
    :type max_leaf_nodes: int
    """
    def __init__(self, n_trees: int = 50, max_depth: int = None, min_samples_split: int = 2, max_leaf_nodes: int = 500) -> None:
        """
        Constructor method to initializes the RandomForest model with given hyperparameters.

       :param n_trees: Number of trees in the forest.
        :type n_trees: int
        :param max_depth: Maximum depth of the tree.
        :type max_depth: Optional[int]
        :param min_samples_split: Minimum number of samples required to split an internal node.
        :type min_samples_split: int
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
        Train the RandomForest model to fit the RandomForest model to the provided data.


        :param observations: Feature matrix for training data.
        :type observations: np.ndarray
        :param ground_truth: Ground truth labels for target values.
        :type ground_truth: np.ndarray
        """
        self._model.fit(observations, ground_truth)
        self._parameters.update(self._model.get_params())

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for input features.


        :param x: Feature matrix of input data for prediction.
        :type x: np.ndarray
        :return: Predicted class labels of target values.
        :rtype: np.ndarray
        """
        return self._model.predict(x)

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, object]:
        """
        Evaluates the classification model 
        using test data, displaying metrics and a confusion matrix.

        :param x_test: Test input features.
        :type x_test: np.ndarray
        :param y_test: Test target values..
        :type y_test: np.ndarray
        :param return: A dictionary conataining Evaluation metrics including accuracy, precision, recall, and F1-score.
        :type return: dict
        """
        y_test_pred = self.predict(x_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_test_pred)
        
        # Generate classification report
        report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred))
        
        # Plot confusion matrix if desired
        self._plot_confusion_matrix(conf_matrix, classes=np.unique(y_test))
        
        return {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": conf_matrix
        }

    def _plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues) -> None:
        """
        Plots the confusion matrix.
        
        :param cm: Confusion matrix.
        :type cm: np.ndarray
        :param classes: List or array of class names.
        :type classes: np.ndarray
        :param normalize: Whether to normalize the confusion matrix.
        :type normalize: bool
        :param title: Title of the plot.
        :type title: str
        :param cmap: Color map used in the plot.
        :type cmap: matplotlib colormap
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True Position')
        plt.xlabel('Predicted Position')
        plt.show()



    def plot_feature_importance(self, feature_names: list) -> None:
        """        
        Plot the top 10 feature importances from the trained model.

        :param feature_names: List of feature names corresponding to input features.
        :type feature_names: List[str]
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
        Plot actual vs. predicted values to visualize prediction quality.

        :param y_test: Ground truth target values.
        :type y_test: np.ndarray
        :param y_test_pred: Predicted target values.
        :type y_test_pred: np.ndarray
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        plt.tight_layout()
        plt.show()

