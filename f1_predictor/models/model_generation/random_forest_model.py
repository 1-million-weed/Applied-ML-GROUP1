from f1_predictor.models.model_generation.model import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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

    def __init__(self, n_trees: int = 1000, max_depth: int = None, min_samples_split: int = 2, max_leaf_nodes: int = 100) -> None:
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
        Evaluates the classification model on the test data.

        Args:
            x_test (np.ndarray): Test input features.
            y_test (np.ndarray): Test target values.

        Returns:
            dict: Evaluation metrics including accuracy, precision, recall, and F1-score.
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

    def _plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        """
        Plots the confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            classes (list): List of class labels
            normalize (bool): Whether to normalize the confusion matrix
            title (str): Title for the plot
            cmap: Colormap for the plot
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

