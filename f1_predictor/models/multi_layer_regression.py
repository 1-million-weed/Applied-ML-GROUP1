import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from tensorflow import keras
from .model import Model
from threading import Thread
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class MultiLayerRegression(Model):
    def __init__(self, type: str = "MultiLayerRegression", input_shape: int = 9, num_classes: int = 20) -> None:
        super().__init__(type)
        self.num_classes = num_classes
        inputs = keras.Input(shape=(input_shape,))
        x = keras.layers.Dense(32, activation='relu')(inputs)
        x = keras.layers.Dense(16, activation='relu')(x)
        x = keras.layers.Dense(8, activation='relu')(x)
        x = keras.layers.Dense(4, activation='relu')(x)
        outputs = keras.layers.Dense(1, activation='linear')(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs)
        self._model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        
       
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
        
        
        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        self._history = self._model.fit(
            observations,
            ground_truth_array,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[tensorboard_callback, early_stopping]
        )
        
        # Start TensorBoard in a separate thread
        self.run_tensorboard()


    def predict(self, observations: np.ndarray, return_zero_indexed: bool = False, round:bool = True) -> np.ndarray:
        """
        Predict the most likely class (finishing position) for each observation.
        
        Args:
            observations: Input features as a numpy array.
            return_zero_indexed: If True, returns positions 0-19, otherwise returns 1-20.
            
        Returns:
            Predicted class labels (finishing positions) as a numpy array.
        """
        predictions = self._model.predict(observations)
        # Convert predictions to finishing positions
        if round:
            predicted_positions = np.round(predictions).astype(int)
        
        # If return_zero_indexed is False, convert to 1-indexed
        if not return_zero_indexed:
            predicted_positions += 1
            
        return predicted_positions.flatten()
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
       
       mean_squared_error, mean_absolute_error = self._model.evaluate(x_test, y_test)
       print(f"Mean Squared Error: {mean_squared_error}")
       print(f"Mean Absolute Error: {mean_absolute_error}")
       y_pred = self.predict(x_test)
       self.plot_confusion_matrix(y_test, y_pred)
       accuracy = accuracy_score(y_test, y_pred)
       precision = precision_score(y_test, y_pred, average="weighted")
       recall = recall_score(y_test, y_pred, average="weighted")
       f1 = f1_score(y_test, y_pred, average="weighted")
       metrics = {
              "mean_squared_error": mean_squared_error,
                "mean_absolute_error": mean_absolute_error,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
       }
       print(metrics)
       return metrics

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
            y_true: True labels as a numpy array.
            y_pred: Predicted labels as a numpy array.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def run_tensorboard(self):
        """
        Start TensorBoard in a separate thread.
        """
        # Start TensorBoard in a separate thread
        thread = Thread(target=self._start_tensorboard)
        thread.start()


    def _start_tensorboard(self):
        os.system(f"tensorboard --logdir {self.log_dir}")