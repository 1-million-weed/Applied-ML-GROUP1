import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import datetime
import tensorflow as tf
from tensorflow import keras
from .model import Model
import os

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
        self.num_classes = num_classes
        self._model = keras.Sequential([
            keras.Input(shape=(input_shape,)),
            keras.layers.Dense(16, activation='linear', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dense(16, activation='linear'),

            #keras.layers.Dense(8, activation='linear', kernel_regularizer=keras.regularizers.l2(0.005)),
            #keras.layers.Dense(8, activation='linear', kernel_regularizer=keras.regularizers.l2(0.005)),
            #keras.layers.Dense(8, activation='linear', kernel_regularizer=keras.regularizers.l2(0.005)),
            keras.layers.Dense(num_classes, activation='softmax')  # Output layer for classification
        ])
        self._model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray, epochs: int = 300, batch_size: int = 2**12, validation_split: float = 0.2) -> None:
        """
        Train the model on the given observations and ground truth.
        
        Args:
            observations: Input features as a numpy array.
            ground_truth: Target values (finishing positions) as a numpy array.
            epochs: Number of epochs to train for.
            batch_size: Batch size for training.
            validation_split: Fraction of the data to use for validation.
        """
        # Convert ground_truth to one-hot encoding
        # First, ensure ground_truth is 0-indexed for proper one-hot encoding
        ground_truth_array = np.array(ground_truth)
        
        # Check for and handle positions that are outside the expected range
        max_position = ground_truth_array.max()
        if max_position > self.num_classes:
            print(f"Warning: Found finishing positions up to {max_position}, which exceeds the model's output size of {self.num_classes}.")
            print(f"Limiting positions to range 0-{self.num_classes-1}.")
            # Clip values to be within valid range for one-hot encoding
            ground_truth_array = np.clip(ground_truth_array, 0, self.num_classes-1)
        
        # If finishing positions are 1-indexed (1 to 20), convert to 0-indexed (0 to 19)
        if ground_truth_array.min() == 1:
            ground_truth_array = ground_truth_array - 1
            
        # Convert to one-hot encoding
        one_hot_ground_truth = to_categorical(ground_truth_array, num_classes=self.num_classes)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        self._history = self._model.fit(
            observations, 
            one_hot_ground_truth, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            callbacks=[early_stopping, tensorboard_callback]
        )
        os.system("tensorboard --logdir logs/fit")


    def predict(self, observations: np.ndarray, return_zero_indexed: bool = False) -> np.ndarray:
        """
        Predict the most likely class (finishing position) for each observation.
        
        Args:
            observations: Input features as a numpy array.
            return_zero_indexed: If True, returns positions 0-19, otherwise returns 1-20.
            
        Returns:
            Predicted class labels (finishing positions) as a numpy array.
        """
        probs = self._model.predict(observations)
        positions = np.argmax(probs, axis=1)
        
        # Convert from 0-indexed back to 1-indexed if needed
        if not return_zero_indexed:
            positions = positions + 1
            
        return positions

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Evaluate the model on the test data.

        Args:
            x_test: Test input features as a numpy array.
            y_test: Test target values as a numpy array (one-hot encoded).
        """
        # Check if y_test is already one-hot encoded
        if len(y_test.shape) == 1 or y_test.shape[1] == 1:
            # Convert to one-hot if it's not
            y_test_array = np.array(y_test).flatten()
            
            # Handle values outside the expected range
            if y_test_array.max() > self.num_classes:
                print(f"Warning: Test data contains positions up to {y_test_array.max()}, clipping to range 0-{self.num_classes-1}")
                y_test_array = np.clip(y_test_array, 0, self.num_classes-1)
                
            if y_test_array.min() == 1:
                y_test_array = y_test_array - 1
                
            y_test = to_categorical(y_test_array, num_classes=self.num_classes)
        self.plot_confusion_matrix(y_test, self.predict(x_test))
        loss, accuracy = self._model.evaluate(x_test, y_test)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
        return {"loss": loss, "accuracy": accuracy}
        


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

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot the confusion matrix for the model predictions.

        Args:
            y_true: True labels (one-hot encoded or single-label).
            y_pred: Predicted labels (single-label).
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        # Convert y_true from one-hot encoding to single-label format if necessary
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    
