import os
import pandas as pd
import click
from sklearn.ensemble import RandomForestClassifier
import pickle


def serialize_model(model: RandomForestClassifier, path: str):
    """Serialization of the model.
    :param model: model for serialization
    :param path: recording path model
    """

    with open(path, "wb") as file:
        pickle.dump(model, file)


@click.command("train")
@click.option("--data-dir")
@click.option("--model-dir")
@click.option("--n-estimators", default=50)
@click.option("--random-state", default=42)
def train_model(
        data_dir: str,
        model_dir: str,
        n_estimators: int,
        random_state: int,
):
    features = pd.read_csv(os.path.join(data_dir, "data.csv"))
    targets = pd.read_csv(os.path.join(data_dir, "target.csv"))

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(features, targets)

    os.makedirs(model_dir, exist_ok=True)
    serialize_model(model, os.path.join(model_dir, "model.pkl"))


if __name__ == '__main__':
    train_model()
