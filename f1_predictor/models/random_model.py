import numpy as np
import matplotlib.pyplot as plt
from .model import Model
from sklearn.metrics import confusion_matrix
import seaborn as sns

class RandomModel(Model):
    """
    A simple random predictor model for Formula 1 finishing positions.
    Predicts a random position between 1-20 for each input.
    """
    def __init__(self, type: str = "RandomPredictor", num_classes: int = 20) -> None:
        """
        Constructor method to initialize the RandomPredictor model.

        :param type: Model type identifier.
        :type type: str
        :param num_classes: Number of output classes (positions).
        :type num_classes: int
        """
        super().__init__(type)
        self.num_classes = num_classes
        self._history = None

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray, **kwargs) -> None:
        """
        No actual training is performed since this is a random predictor.
        
        :param observations: Input features (not used).
        :type observations: np.ndarray
        :param ground_truth: Target values (not used).
        :type ground_truth: np.ndarray
        """
        print("No training needed for random predictor.")
        # Create a dummy history object for compatibility
        self._history = {"history": {"loss": [0], "val_loss": [0]}}

    def predict(self, observations: np.ndarray, return_zero_indexed: bool = False, round: bool = True, **kwargs) -> np.ndarray:
        """
        Predict random positions between 1-20 for each observation.
        
        :param observations: Input data (only used for determining output size).
        :type observations: np.ndarray
        :param return_zero_indexed: If True, returns positions 0-19, otherwise returns 1-20.
        :type return_zero_indexed: bool
        :param round: If True, returns integer positions, otherwise returns float positions.
        :type round: bool
        :return: Random predicted positions.
        :rtype: np.ndarray
        """
        # Generate random positions
        num_samples = len(observations)
        
        if round:
            # Return integer predictions
            if return_zero_indexed:
                # Random integers between 0 and num_classes-1
                predictions = np.random.randint(0, self.num_classes, size=num_samples)
            else:
                # Random integers between 1 and num_classes
                predictions = np.random.randint(1, self.num_classes + 1, size=num_samples)
        else:
            # Return float predictions
            if return_zero_indexed:
                # Random floats between 0 and num_classes-1
                predictions = np.random.uniform(0, self.num_classes, size=num_samples)
            else:
                # Random floats between 1 and num_classes
                predictions = np.random.uniform(1, self.num_classes + 1, size=num_samples)
            
        return predictions

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the random model on test data.

        :param x_test: Test features.
        :type x_test: np.ndarray
        :param y_test: Test target values.
        :type y_test: np.ndarray
        :return: Evaluation metrics.
        :rtype: dict
        """
        predictions = self.predict(x_test)
        
        # Calculate simple accuracy by comparing predictions to ground truth
        # Convert y_test to same format as predictions if needed
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            # If one-hot encoded, convert to class labels
            y_test = np.argmax(y_test, axis=1) + 1
        
        accuracy = np.mean(predictions == y_test)
        print(f"Random predictor accuracy: {accuracy:.4f}")
        
        # Plot confusion matrix for visualization
        self.plot_confusion_matrix(y_test, predictions)
        
        return {"accuracy": accuracy}

    def plot_loss(self) -> None:
        """
        Placeholder method for API compatibility.
        """
        print("No meaningful loss plot available for random predictor.")

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot the confusion matrix for the model predictions.

        :param y_true: True labels.
        :type y_true: np.ndarray
        :param y_pred: Predicted labels.
        :type y_pred: np.ndarray
        """
        

        # Ensure y_true is in the right format
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()