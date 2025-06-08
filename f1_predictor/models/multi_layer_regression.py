import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import keras
from .model import Model
from threading import Thread
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras import backend as K



class MultiLayerRegression(Model):
    def __init__(self, type: str = "MultiLayerRegression", input_shape: int = 9, num_classes: int = 20, dropout_rate: float = 0.1) -> None:
        """
        Initialize the MultiLayerRegression model with MC-dropout for uncertainty quantification.
        
        Args:
            type: Model type identifier
            input_shape: Number of input features
            num_classes: Maximum position number (20 for F1)
            dropout_rate: Dropout rate for MC-dropout layers
        """
        super().__init__(type)
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Create model with MC-dropout
        inputs = keras.Input(shape=(input_shape,))
        x = keras.layers.Dense(32, activation='relu')(inputs)
        x = keras.layers.Dropout(dropout_rate)(x, training=True)  # Always enabled for uncertainty estimation
        x = keras.layers.Dense(16, activation='relu')(x)
        x = keras.layers.Dropout(dropout_rate)(x, training=True)
        x = keras.layers.Dense(8, activation='relu')(x)
        x = keras.layers.Dropout(dropout_rate)(x, training=True)
        x = keras.layers.Dense(4, activation='relu')(x)
        x = keras.layers.Dropout(dropout_rate)(x, training=True)
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


    def predict_with_uncertainty(self, observations: np.ndarray, num_samples: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates using MC-dropout.
        
        Args:
            observations: Input features as a numpy array
            num_samples: Number of Monte Carlo samples to draw
            
        Returns:
            tuple: (mean predictions, standard deviations)
        """
        predictions = []
        for _ in range(num_samples):
            pred = self._model(observations, training=True)  # Enable dropout during inference
            predictions.append(pred)
            
        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        return mean_prediction, std_prediction
        
    def predict(self, observations: np.ndarray, return_zero_indexed: bool = False, with_uncertainty: bool = False, num_samples: int = 100) -> np.ndarray:
        """
        Predict the most likely position for each observation.
        
        Args:
            observations: Input features as a numpy array
            return_zero_indexed: If True, returns positions 0-19, otherwise returns 1-20
            with_uncertainty: If True, returns uncertainty estimates using MC-dropout
            num_samples: Number of Monte Carlo samples when using uncertainty estimation
            
        Returns:
            If with_uncertainty=False: predicted positions
            If with_uncertainty=True: tuple of (predicted positions, uncertainties)
        """
        if with_uncertainty:
            predictions, uncertainties = self.predict_with_uncertainty(observations, num_samples)
        else:
            predictions = self._model.predict(observations)
            uncertainties = None
            
        # Round to nearest integer but don't modify the range yet
        predicted_positions = np.round(predictions).astype(int)
        
        # Only clip the range, don't add or subtract 1
        predicted_positions = np.clip(predicted_positions, 1, self.num_classes)
            
        if with_uncertainty:
            return predicted_positions.flatten(), uncertainties.flatten()
        return predicted_positions.flatten()

    def plot_position_group_uncertainty(self, y_true: np.ndarray, y_pred: np.ndarray, uncertainties: np.ndarray) -> None:
        """
        Plot uncertainty distribution for different position groups.
        
        Args:
            y_true: True positions
            y_pred: Predicted positions
            uncertainties: Uncertainty values for each prediction
        """
        # Create position groups (1-4, 5-8, 9-12, 13-16, 17-20)
        position_groups = [(1,4), (5,8), (9,12), (13,16), (17,20)]
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, height_ratios=[1, 1])
        
        # Violin plot
        ax1 = fig.add_subplot(gs[0, :])
        boxplot_data = []
        
        for start, end in position_groups:
            mask = (y_true >= start) & (y_true <= end)
            group_uncertainties = uncertainties[mask]
            boxplot_data.append(group_uncertainties)
        
        # Create violin plot
        parts = ax1.violinplot(boxplot_data, positions=range(1, len(position_groups) + 1), 
                             showmeans=True, showextrema=True, showmedians=True)
        ax1.set_xticks(range(1, len(position_groups) + 1))
        ax1.set_xticklabels([f'P{s}-P{e}' for s, e in position_groups])
        ax1.set_ylabel('Prediction Uncertainty')
        ax1.set_title('Distribution of Prediction Uncertainty by Position Groups')
        
        # Predicted vs True positions with uncertainty
        ax2 = fig.add_subplot(gs[1, 0])
        for i, (start, end) in enumerate(position_groups):
            mask = (y_true >= start) & (y_true <= end)
            ax2.scatter(y_true[mask], y_pred[mask], 
                       alpha=0.6, c=colors[i], label=f'P{start}-P{end}')
            
        # Add error bars for a few example points from each group
        for i, (start, end) in enumerate(position_groups):
            mask = (y_true >= start) & (y_true <= end)
            indices = np.where(mask)[0]
            if len(indices) > 0:
                # Sample up to 3 points from each group
                sample_idx = np.random.choice(indices, min(3, len(indices)), replace=False)
                ax2.errorbar(y_true[sample_idx], y_pred[sample_idx], 
                           yerr=uncertainties[sample_idx], fmt='none', 
                           alpha=0.3, color=colors[i])
        
        ax2.plot([0, 20], [0, 20], 'r--', alpha=0.5, label='Perfect prediction')
        ax2.set_xlabel('True Position')
        ax2.set_ylabel('Predicted Position')
        ax2.set_title('Predicted vs True Positions\nwith Uncertainty Bars (sample)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_xlim(0, 21)
        ax2.set_ylim(0, 21)
        
        # Mean uncertainty per position
        ax3 = fig.add_subplot(gs[1, 1])
        mean_uncertainties = []
        std_uncertainties = []
        positions = range(1, 21)
        
        for pos in positions:
            mask = (y_true == pos)
            if np.any(mask):
                mean_uncertainties.append(np.mean(uncertainties[mask]))
                std_uncertainties.append(np.std(uncertainties[mask]))
            else:
                mean_uncertainties.append(0)
                std_uncertainties.append(0)
        
        ax3.bar(positions, mean_uncertainties, yerr=std_uncertainties, 
                alpha=0.6, capsize=5)
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Mean Uncertainty')
        ax3.set_title('Average Uncertainty per Position')
        
        plt.tight_layout()
        plt.show()

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        mean_squared_error, mean_absolute_error = self._model.evaluate(x_test, y_test)
        print(f"Mean Squared Error: {mean_squared_error}")
        print(f"Mean Absolute Error: {mean_absolute_error}")
        
        # Get predictions and uncertainties
        y_pred, uncertainties = self.predict(x_test, with_uncertainty=True)
        
        # Plot standard confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        # Plot position group uncertainty
        self.plot_position_group_uncertainty(y_test, y_pred, uncertainties)
        
        # Find most and least certain predictions
        n_examples = 5
        sorted_indices = np.argsort(uncertainties)
        most_certain_idx = sorted_indices[:n_examples]
        least_certain_idx = sorted_indices[-n_examples:]
        
        print("\n5 Most Certain Predictions:")
        print("True Position | Predicted Position | Uncertainty | Absolute Error")
        print("-" * 65)
        for idx in most_certain_idx:
            true_pos = y_test[idx]
            pred_pos = y_pred[idx]
            uncertainty = uncertainties[idx]
            abs_error = abs(true_pos - pred_pos)
            print(f"{true_pos:^13.0f} | {pred_pos:^17.0f} | {uncertainty:^10.4f} | {abs_error:^14.0f}")
            
        print("\n5 Most Uncertain Predictions:")
        print("True Position | Predicted Position | Uncertainty | Absolute Error")
        print("-" * 65)
        for idx in least_certain_idx:
            true_pos = y_test[idx]
            pred_pos = y_pred[idx]
            uncertainty = uncertainties[idx]
            abs_error = abs(true_pos - pred_pos)
            print(f"{true_pos:^13.0f} | {pred_pos:^17.0f} | {uncertainty:^10.4f} | {abs_error:^14.0f}")
        
        # Calculate metrics
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
            "f1_score": f1,
            "mean_uncertainty": np.mean(uncertainties),
            "uncertainty_std": np.std(uncertainties)
        }
        print("\nOverall Metrics:")
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
        # Ensure predictions and ground truth are 1-based
        y_true_1based = np.where(y_true < 1, 1, y_true)
        y_pred_1based = np.where(y_pred < 1, 1, y_pred)
        
        # Clip values to maximum position
        y_true_1based = np.clip(y_true_1based, 1, self.num_classes)
        y_pred_1based = np.clip(y_pred_1based, 1, self.num_classes)
        
        # Create position labels (1 to num_classes)
        labels = np.arange(1, self.num_classes + 1)
        
        cm = confusion_matrix(y_true_1based, y_pred_1based, labels=labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Position')
        plt.ylabel('True Position')
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