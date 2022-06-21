import os
import pandas as pd
import numpy as np
import click
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
import pickle


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


def write_prediction(
    prediction: np.ndarray, path: str
):
    """Writes the model predictions to a file.
    :param prediction: model predictions
    :param path: a way to record predictions
    """

    with open(path, 'w') as file:
        file.write("id,prediction\n")
        for i in range(prediction.size):
            file.write(f"{i},{prediction[i]}\n")


@click.command("predict")
@click.option("--data-dir")
@click.option("--model-dir")
@click.option("--predicts-dir")
@click.option("--transformer-dir")
def predict_model(
        data_dir: str,
        model_dir: str,
        predicts_dir: str,
        transformer_dir: str,

):
    features = pd.read_csv(os.path.join(data_dir, "data.csv"))
    model = deserialize_model(os.path.join(model_dir, "model.pkl"))
    transformer = deserialize_transformer(os.path.join(
        transformer_dir,
        "transformer.pkl",
    ))

    preprocess_features = pd.DataFrame(transformer.transform(features))
    predicts = model.predict(preprocess_features)

    os.makedirs(predicts_dir, exist_ok=True)
    write_prediction(predicts, os.path.join(model_dir, "predictions.csv"))


if __name__ == '__main__':
    predict_model()
