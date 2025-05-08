from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy



class Model(ABC):
    """
    Abstract base class for machine learning models.

    Attributes:
        _model: The machine learning model instance.
        type: The type of the model.
        _hyperparameters: The hyperparameters for the model.
        parameters: The parameters of the model after fitting.
    """

    def __init__(self, type: str = None) -> None:
        """
        Initialize the model with given hyperparameters.

        Args:
            hyperparameters: Arbitrary keyword arguments for hyperparameters.
        """
        self._parameters: dict = {}
        self._type = type

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on observations and ground truth.

        Args:
            observations: Input features as a numpy array.
            ground_truth: Target values as a numpy array.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict based on the observations.

        Args:
            observations: Input features as a numpy array.
        """
        pass

    @property
    def parameters(self) -> dict:
        """ Returns a copy of parameters to prevent leakage. """
        return deepcopy(self._parameters)

    @property
    def type(self) -> str:
        """ Returns the model type. """
        return self._type

    @property
    def model(self) -> None:
        """
        Returns a deepcopy of the model instance to prevent leakage.
        """
        return deepcopy(self._model)
