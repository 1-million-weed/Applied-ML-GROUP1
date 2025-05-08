import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from .model import Model

class MultiLayerPerceptron(Model):
    def __init__(self, type: str = "MultiLayerPerceptron", input_shape: int = 4, num_classes: int = 20) -> None:
        """
        Initialize the MultiLayerPerceptron model for classification.

        Args:
            type: The type of the model.
            input_shape: The number of input features.
            num_classes: The number of output classes (20 positions).
        """
        super().__init__(type)
        self._model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')  # Output layer for classification
        ])
        self._model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray, epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2) -> None:
        """
        Train the model on the provided data.

        Args:
            observations: Input features as a numpy array.
            ground_truth: Target values as a numpy array (one-hot encoded).
            epochs: Number of epochs for training.
            batch_size: Batch size for training.
            validation_split: Fraction of data to use for validation.
        """
        self._history = self._model.fit(observations, ground_truth, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the class probabilities based on the observations.

        Args:
            observations: Input features as a numpy array.

        Returns:
            Predicted class probabilities as a numpy array.
        """
        return self._model.predict(observations)

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Evaluate the model on the test data.

        Args:
            x_test: Test input features as a numpy array.
            y_test: Test target values as a numpy array (one-hot encoded).
        """
        loss, accuracy = self._model.evaluate(x_test, y_test)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    def plot_loss(self) -> None:
        """
        Plot the training and validation loss over epochs.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self._history.history['loss'], label='Training Loss')
        plt.plot(self._history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()