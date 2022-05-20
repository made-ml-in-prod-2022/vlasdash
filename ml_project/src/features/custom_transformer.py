import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
    """Custom MinMaxScaler."""

    def __init__(self):
        """Init transformer. """

        self.min = None
        self.max = None

    def fit(self, dataset: np.ndarray):
        """Fit custom transformer.

        :param dataset: input dataset
        :return: trained transformer
        """

        self.min = dataset.min(axis=0)
        self.max = dataset.max(axis=0)

        return self

    def transform(self, dataset: np.ndarray) -> np.ndarray:
        """Transform dataset.

        :param dataset: input dataset
        :return: scaled dataset
        """

        result = (dataset - self.min) / (self.max - self.min + 0.00001)

        return result
