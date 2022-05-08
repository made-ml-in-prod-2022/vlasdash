import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CustomMinMaxScaler(BaseEstimator, TransformerMixin):

    def fit(self, dataset: np.ndarray):
        self.min = dataset.min(axis=0)
        self.max = dataset.max(axis=0)

        return self

    def transform(self, dataset: np.ndarray) -> np.ndarray:
        result = (dataset - self.min) / (self.max - self.min + 0.00001)

        return result
