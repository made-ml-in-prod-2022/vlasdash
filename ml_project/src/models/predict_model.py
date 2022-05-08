from typing import Union
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np
import pickle

SklearnClassifierModel = Union[SGDClassifier, RandomForestClassifier]


def predict_model(
    model: SklearnClassifierModel, features: pd.DataFrame
) -> np.ndarray:
    return model.predict(features)


def write_prediction(
    prediction: np.ndarray, path: str
) -> str:
    with open(path, 'w') as sf:
        sf.write("id,prediction\n")
        for i in range(prediction.size):
            sf.write(f"{i},{prediction[i]}\n")

    return path


def deserialize_model(path: str) -> SklearnClassifierModel:
    with open(path, "rb") as sf:
        model = pickle.load(sf)

    return model
