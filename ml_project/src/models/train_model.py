from typing import Union, Dict
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import json

from entities.metric_params import MetricParams
from entities.training_params import TrainingParams

SklearnClassifierModel = Union[SGDClassifier, RandomForestClassifier]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassifierModel:
    """Train model.

    :param features: input features
    :param target: target values
    :param train_params: parameters for model training
    :return: trained model
    """

    if train_params.model == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.model_forest_params.n_estimators,
            random_state=train_params.model_forest_params.random_state,
            max_depth=train_params.model_forest_params.max_depth,
        )
    elif train_params.model == "SGDClassifier":
        model = SGDClassifier(
            random_state=train_params.model_sgd_params.random_state,
            penalty=train_params.model_sgd_params.penalty,
            alpha=train_params.model_sgd_params.alpha,
        )

    model.fit(features, target)

    return model


def evaluate_model(
    predicts: np.ndarray, target: pd.Series,
    params: MetricParams
) -> Dict[str, float]:
    """Evaluate model.

    :param predicts:
    :param target: target values
    :param params: parameters for metrics
    :return: model quality metrics
    """

    model_metrics = {}

    if params.precision:
        model_metrics["precision"] = metrics.precision_score(target, predicts)
    if params.recall:
        model_metrics["recall"] = metrics.recall_score(target, predicts)
    if params.f1:
        model_metrics["f1"] = metrics.f1_score(target, predicts)

    return model_metrics


def serialize_model(model: SklearnClassifierModel, path: str) -> str:
    """Serialization of the model.

    :param model: model for serialization
    :param path: recording path model
    :return: recording path model
    """

    with open(path, "wb") as sf:
        pickle.dump(model, sf)

    return path


def write_metrics(model_metrics: Dict[str, float], path: str):
    """Write the metrics of the model

    :param model_metrics: model quality metrics
    :param path: recording path metrics
    :return: recording path metrics
    """

    with open(path, "w") as sf:
        json.dump(model_metrics, sf)

    return path
