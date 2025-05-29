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
from collections import defaultdict
from typing import Union



class MultiLayerRegression(Model):
    def __init__(self, type: str = "MultiLayerRegression", input_shape: int = 11, num_classes: int = 20) -> None:
        super().__init__(type)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self._initialize_model()
        self.metrics_history = defaultdict(list)
        
    def _initialize_model(self):
        inputs = keras.Input(shape=(self.input_shape,))
        x = keras.layers.Dense(32, activation='relu')(inputs)
        x = keras.layers.Dense(16, activation='relu')(x)
        x = keras.layers.Dense(8, activation='relu')(x)
        x = keras.layers.Dense(4, activation='relu')(x)
        outputs = keras.layers.Dense(1, activation='linear')(x)

        self._model = keras.Model(inputs=inputs, outputs=outputs)
        self._model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
    def fit_sequential_years(self, train_data: tuple, epochs: int = 50, 
                           batch_size: int = 2**12, validation_split: float = 0.2,
                           early_stopping_patience: int = 10) -> None:
        """
        Train the model sequentially on data year by year.
        
        Args:
            train_data: Tuple of (features_df, ground_truth) where features_df is a DataFrame 
                       containing features and 'year' column, and ground_truth is a 1D array 
                       of finishing positions
            epochs: Number of epochs per year
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Number of epochs with no improvement after which training will stop
        """
        features_df, all_ground_truth = train_data
          # Get unique years and sort them
        years = sorted(features_df['year'].unique())
        
        # Use exactly the features specified in config.yaml
        feature_cols = ['normalized_lap', 'average_normalized_lap', 'lap_progress', 
                       'current_position_norm', 'normalized_driver_standing', 
                       'normalized_fastest_qualifying', 'position_quali', 
                       'normalized_driver_elo', 'amount_of_wins', 'points_team', 'year']
        
        # Try to load initial weights from earliest available year
        if len(years) > 0:
            self._initialize_weights(years[0])
        
        for year in years:
            print(f"\nTraining on year {year}")
            
            # Get mask for current year's data
            year_mask = features_df['year'] == year
            
            # Get features and targets for current year
            X_year = features_df[year_mask][feature_cols].values
            y_year = all_ground_truth[year_mask]
            
            # Create TensorBoard callback for this year
            log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_year_{year}"
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            )
            
            # Early stopping callback
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            )
            
            # Train on current year's data
            history = self._model.fit(
                X_year, y_year,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[tensorboard_callback, early_stopping],
                verbose=1
            )
            
            # Store metrics for this year
            self.metrics_history[year] = {
                'loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1],
                'mae': history.history['mae'][-1],
                'val_mae': history.history['val_mae'][-1]
            }
            
            # Evaluate on all data up to current year
            historical_mask = features_df['year'] <= year
            X_hist = features_df[historical_mask][feature_cols].values
            y_hist = all_ground_truth[historical_mask]
            
            eval_metrics = self._model.evaluate(X_hist, y_hist, verbose=0)
            self.metrics_history[f'cumulative_{year}'] = {
                'loss': eval_metrics[0],
                'mae': eval_metrics[1]
            }
              # Save weights for this year
            self._save_weights(year)
            
    def plot_training_metrics(self):
        """Plot the training metrics across years"""
        years = sorted([k for k in self.metrics_history.keys() if not str(k).startswith('cumulative')])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot per-year metrics
        ax1.plot([self.metrics_history[year]['loss'] for year in years], label='Training Loss')
        ax1.plot([self.metrics_history[year]['val_loss'] for year in years], label='Validation Loss')
        ax1.set_title('Loss by Year')
        ax1.set_xlabel('Year Index')
        ax1.set_ylabel('Mean Squared Error')
        ax1.legend()
        
        # Plot cumulative metrics
        cumulative_years = sorted([k for k in self.metrics_history.keys() if str(k).startswith('cumulative')])
        ax2.plot([self.metrics_history[year]['loss'] for year in cumulative_years], label='Cumulative MSE')
        ax2.plot([self.metrics_history[year]['mae'] for year in cumulative_years], label='Cumulative MAE')
        ax2.set_title('Cumulative Metrics')
        ax2.set_xlabel('Year Index')
        ax2.set_ylabel('Error')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('logs/sequential_training_metrics.png')
        plt.close()

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get feature column names from training config"""
        return [col for col in df.columns if col not in ['finishing_position', 'year', 'race_id', 'driver_id']]
        
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray, epochs: int = 300, 
            batch_size: int = 2**12, validation_split: float = 0.2) -> None:
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
    def predict(self, observations: Union[np.ndarray, pd.DataFrame], return_zero_indexed: bool = False) -> np.ndarray:
        """
        Predict the most likely class (finishing position) for each observation.
        
        Args:
            observations: Input features as either numpy array or pandas DataFrame.
            return_zero_indexed: If True, returns positions 0-19, otherwise returns 1-20.
            
        Returns:
            Predicted class labels (finishing positions) as a numpy array.
        """
        # Handle DataFrame input
        if isinstance(observations, pd.DataFrame):
            feature_cols = ['normalized_lap', 'average_normalized_lap', 'lap_progress', 
                          'current_position_norm', 'normalized_driver_standing', 
                          'normalized_fastest_qualifying', 'position_quali', 
                          'normalized_driver_elo', 'amount_of_wins', 'points_team', 'year']
            observations = observations[feature_cols].values
        
        predictions = self._model.predict(observations)
        predicted_positions = np.round(predictions).astype(int)
        # Clip to valid range
        if return_zero_indexed:
            predicted_positions = np.clip(predicted_positions, 0, self.num_classes - 1)
        else:
            predicted_positions = np.clip(predicted_positions, 1, self.num_classes)
        return predicted_positions.flatten()
    
    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            x_test: Test features as a pandas DataFrame
            y_test: Test targets as a pandas Series
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Use the same feature columns as in training
        feature_cols = ['normalized_lap', 'average_normalized_lap', 'lap_progress', 
                       'current_position_norm', 'normalized_driver_standing', 
                       'normalized_fastest_qualifying', 'position_quali', 
                       'normalized_driver_elo', 'amount_of_wins', 'points_team', 'year']
        
        # Verify all required features are present
        missing_cols = [col for col in feature_cols if col not in x_test.columns]
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")
            
        # Ensure we use the same features in the same order
        x_test_array = x_test[feature_cols].values
        y_test_array = y_test.values
            
        # Evaluate the model
        mean_squared_error, mean_absolute_error = self._model.evaluate(x_test_array, y_test_array, verbose=0)
        print(f"Mean Squared Error: {mean_squared_error}")
        print(f"Mean Absolute Error: {mean_absolute_error}")
        
        # Get predictions for additional metrics
        y_pred = self.predict(x_test_array)
        
        # Calculate classification metrics (since we're predicting positions)
        accuracy = accuracy_score(y_test_array, y_pred)
        precision = precision_score(y_test_array, y_pred, average="weighted")
        recall = recall_score(y_test_array, y_pred, average="weighted")
        f1 = f1_score(y_test_array, y_pred, average="weighted")
        
        # Create metrics dictionary
        metrics = {
            "mean_squared_error": mean_squared_error,
            "mean_absolute_error": mean_absolute_error,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        print(metrics)
        
        if hasattr(self, 'plot_confusion_matrix'):
            self.plot_confusion_matrix(y_test_array, y_pred)
            
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
        labels = np.arange(1, self.num_classes + 1)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels
        )
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

    def _save_weights(self, year: int) -> None:
        """Save model weights for a specific year"""
        weights_path = f'models/checkpoints/sequential_mlp_{year}.weights.h5'
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        self._model.save_weights(weights_path)

    def _load_weights(self, year: int) -> None:
        """Load model weights for a specific year"""
        weights_path = f'models/checkpoints/sequential_mlp_{year}.weights.h5'
        if os.path.exists(weights_path):
            self._model.load_weights(weights_path)
        else:
            raise FileNotFoundError(f"No weights found for year {year}")    
        
    def _initialize_weights(self, initial_year: int) -> None:
        """Initialize weights from a previous year if available"""
        try:
            # Ensure model is initialized with correct input shape before loading weights
            if not hasattr(self, '_model'):
                self._initialize_model()
            
            self._load_weights(initial_year - 1)
            print(f"\nInitialized weights from year {initial_year - 1}")
            
            # Verify input shape after loading weights
            expected_shape = (None, self.input_shape)
            actual_shape = self._model.layers[0].input_shape
            if actual_shape != expected_shape:
                raise ValueError(f"Loaded model expects input shape {actual_shape}, but we need {expected_shape}")
                
        except (FileNotFoundError, ValueError) as e:
            print(f"\nStarting with fresh weights: {str(e)}")
            self._initialize_model()  # Reinitialize with correct shape