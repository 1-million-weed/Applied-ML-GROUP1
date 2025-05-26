import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import datetime
import tensorflow as tf
from tensorflow import keras
from .model import Model
import os

@tf.keras.utils.register_keras_serializable()
class OrdinalCrossentropy:
    """
    Custom loss function for ordinal classification, penalizing predictions by their distance from the true class.
    """
    def __init__(self, num_classes) -> None:
        """
        Constructor method to initialize the loss function.

        :param num_classes: Total number of classes in the classification problem.
        :type num_classes: int
        """    
        self.num_classes = num_classes

    def __call__(self, y_true, y_pred) -> tf.Tensor:
        """
        Compute the ordinal cross-entropy loss.

        :param y_true: One-hot encoded true labels.
        :type y_true: tf.Tensor
        :param y_pred: Predicted probabilities.
        :type y_pred: tf.Tensor
        :return: Computed loss.
        :rtype: tf.Tensor
        """    
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        true_labels = tf.argmax(y_true, axis=-1)

        class_indices = tf.range(self.num_classes, dtype=tf.float32)
        class_indices = tf.reshape(class_indices, (1, -1))  # shape: [1, num_classes]
        true_labels_float = tf.cast(tf.expand_dims(true_labels, axis=-1), tf.float32)
        distances = tf.abs(class_indices - true_labels_float)  # shape: [batch, num_classes]

        # Normalize and square distance to penalize larger mistakes more
        max_distance = tf.constant(self.num_classes - 1, dtype=tf.float32)
        normalized_dist = distances / max_distance
        weighted_ce = -y_true * tf.math.log(y_pred) * (1.0 + tf.square(normalized_dist))

        return tf.reduce_mean(tf.reduce_sum(weighted_ce, axis=-1))


    @classmethod
    def from_config(cls, config) -> self:
        """        
        Instantiate the class from a config dictionary.

        :param config: Configuration dictionary.
        :type config: dict
        :return: Instantiated class object.
        :rtype: cls
        """ 
        return cls(**config)

    def get_config(self) -> dict:
        """
        Returns the configuration of the loss function for serialization.

        :return: Configuration dictionary.
        :rtype: dict
        """
        return {"num_classes": self.num_classes}

    @classmethod
    def from_config(cls, config) -> self:
        """
        Instantiates the class from its configuration dictionary.

        :param config: Configuration dictionary.
        :type config: dict
        :return: Instantiated class object.
        :rtype: cls.
        """
        return cls(**config)

class MultiLayerPerceptron(Model):
    """
    A multi-layer perceptron model for classification of Formula 1 finishing positions.

    :param Model: _description_
    :type Model: _type_
    """
    def __init__(self, type: str = "MultiLayerPerceptron", input_shape: int = 4, num_classes: int = 20) -> None:
        """
        Constructor method to initialize the MultiLayerPerceptron model for classification.

        :param type: Model type identifier.
        :type type: str
        :param input_shape: Number of input features.
        :type input_shape: int
        :param num_classes: Number of output classes (positions).
        :type num_classes: int
        """
        super().__init__(type)
        self.num_classes = num_classes
        self._model = keras.Sequential([
            keras.Input(shape=(input_shape,)),
            keras.layers.Dense(200, activation='relu', kernel_regularizer=keras.regularizers.l2(0.05)),
        ])
        
        # Add 1000 layers dynamically
        for _ in range(3):
            self._model.add(keras.layers.Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001)))

        # Add the output layer
        self._model.add(keras.layers.Dense(num_classes, activation='softmax'))  # Output layer for classification
        
        self._model.compile(optimizer='adam', loss=OrdinalCrossentropy(num_classes), metrics=['accuracy'])

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray, epochs: int = 300, batch_size: int = 2**12, validation_split: float = 0.2) -> None:
        """
        Train the model on the given observations and ground truth.
        
        :param observations: Input features.
        :type observations: np.ndarray
        :param ground_truth: Target values for class labels(finishing positions).
        :type ground_truth: np.ndarray        
        :param epochs: Number of training epochs.
        :type epochs: int
        :param batch_size: Training batch size.
        :type batch_size: int
        :param validation_split: Fraction of training data to use for validation.
        :type validation_split: float
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
        
        :param observations: Input data.
        :type observations: np.ndarray
        :param return_zero_indexed: If True, returns positions 0-19, otherwise returns 1-20.
        :type return_zero_indexed: bool
        :return: Predicted class labels (finishing positions).
        :rtype: np.ndarray
        """
        probs = self._model.predict(observations)
        positions = np.argmax(probs, axis=1)
        
        # Convert from 0-indexed back to 1-indexed if needed
        if not return_zero_indexed:
            positions = positions + 1
            
        return positions

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model on the test data.

        :param x_test: Test features.
        :type x_test: np.ndarray
        :param y_test:  Test target values(one-hot encoded).
        :type y_test: np.ndarray
        :return: Evaluation metrics.
        :rtype: dict
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
        Plot the training and validation loss curve over epochs.
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



        :param y_true: True labels (one-hot encoded or single-label).
        :type y_true: np.ndarray
        :param y_pred: Predicted labels (single-label).
        :type y_pred: np.ndarray
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


