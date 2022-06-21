import os
import click
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

CATEGORICAL_FEATURES = [
            "cp",
            "restecg",
            "slope",
            "ca",
            "thal",
            "sex",
            "fbs",
            "exang"
        ]
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def build_categorical_pipeline() -> Pipeline:
    """Build pipeline for categorical features
    :return: pipeline for categorical features
    """

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(
                missing_values=np.nan,
                strategy='most_frequent'
            )),
            ("encoder", OneHotEncoder()),
        ]
    )

    return categorical_pipeline


def build_numerical_pipeline() -> Pipeline:
    """Build pipeline for numerical features
        :return: pipeline for numerical features
    """

    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
         ("scaler", MinMaxScaler())]
    )

    return num_pipeline


def build_transformer() -> ColumnTransformer:
    """Build a data preprocessing transformer.
    :return: transformer
    """

    return ColumnTransformer(
        [("categorical_pipeline", build_categorical_pipeline(), CATEGORICAL_FEATURES),
         ("numerical_pipeline", build_numerical_pipeline(), NUMERICAL_FEATURES)]
    )


def serialize_transformer(transformer: ColumnTransformer, output: str):
    """Serialization of the transformer.
    :param transformer: transformer for serialization
    :param output: recording path transformer
    """

    with open(output, "wb") as file:
        pickle.dump(transformer, file)


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--transformer-dir")
def preprocess_data(input_dir: str, output_dir: str, transformer_dir: str):
    features = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
    targets = pd.read_csv(os.path.join(input_dir, "train_target.csv"))

    transformer = build_transformer()
    process_features = pd.DataFrame(transformer.fit_transform(features))

    os.makedirs(output_dir, exist_ok=True)
    process_features.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    targets.to_csv(os.path.join(output_dir, "target.csv"), index=False)
    os.makedirs(transformer_dir, exist_ok=True)
    path_to_transformer = os.path.join(transformer_dir, "transformer.pkl")
    serialize_transformer(transformer, path_to_transformer)


if __name__ == '__main__':
    preprocess_data()
