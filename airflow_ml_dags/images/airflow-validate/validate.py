import os
import pandas as pd
import click
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
import pickle
import numpy as np
from typing import Dict
from sklearn import metrics
import json


def evaluate_model(
    predicts: np.ndarray, target: pd.Series,
) -> Dict[str, float]:
    """Evaluate model.
    :param predicts:
    :param target: target values
    :return: model quality metrics
    """

    model_metrics = {}

    model_metrics["precision"] = metrics.precision_score(target, predicts)
    model_metrics["recall"] = metrics.recall_score(target, predicts)
    model_metrics["f1"] = metrics.f1_score(target, predicts)

    return model_metrics


def write_metrics(model_metrics: Dict[str, float], path: str):
    """Write the metrics of the model
    :param model_metrics: model quality metrics
    :param path: recording path metrics
    :return: recording path metrics
    """

    with open(path, "w") as file:
        json.dump(model_metrics, file)


def deserialize_model(path: str) -> RandomForestClassifier:
    """Deserialization of the model.
    :param path: path to model
    :return: model
    """

    with open(path, "rb") as file:
        model = pickle.load(file)

    return model


def deserialize_transformer(path: str) -> ColumnTransformer:
    """Deserialize transformer.
        :param path: path to transformer
        :return: deserialized transformer
    """

    with open(path, "rb") as stream:
        return pickle.load(stream)


@click.command("validate")
@click.option("--data-dir")
@click.option("--transformer-dir")
@click.option("--model-dir")
@click.option("--metrics-dir")
def validate_model(
        data_dir: str,
        transformer_dir: str,
        model_dir: str,
        metrics_dir: str,
):
    features = pd.read_csv(os.path.join(data_dir, "valid_data.csv"))
    targets = pd.read_csv(os.path.join(data_dir, "valid_target.csv"))
    model = deserialize_model(os.path.join(model_dir, "model.pkl"))
    transformer = deserialize_transformer(os.path.join(
        transformer_dir,
        "transformer.pkl",
    ))

    preprocess_features = pd.DataFrame(transformer.transform(features))
    predicts = model.predict(preprocess_features)
    model_metrics = evaluate_model(predicts, targets)

    os.makedirs(metrics_dir, exist_ok=True)
    write_metrics(model_metrics, os.path.join(model_dir, "metrics.json"))


if __name__ == '__main__':
    validate_model()
