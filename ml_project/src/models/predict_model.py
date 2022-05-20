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
    """Makes model predictions.

    :param model: model
    :param features: input features
    :return: model predictions
    """

    return model.predict(features)


def write_prediction(
    prediction: np.ndarray, path: str
) -> str:
    """Writes the model predictions to a file.

    :param prediction: model predictions
    :param path: a way to record predictions
    :return:
    """

    with open(path, 'w') as file:
        file.write("id,prediction\n")
        for i in range(prediction.size):
            file.write(f"{i},{prediction[i]}\n")

    return path


def deserialize_model(path: str) -> SklearnClassifierModel:
    """Deserialization of the model.

    :param path: path to model
    :return: model
    """

    with open(path, "rb") as file:
        model = pickle.load(file)

    return model
