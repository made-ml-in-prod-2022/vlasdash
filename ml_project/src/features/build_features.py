import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from entities.feature_params import FeatureParams
from features.custom_transformer import CustomMinMaxScaler


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
         ("scaler", CustomMinMaxScaler())]
    )

    return num_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    """Build a data preprocessing transformer.

    :param params: parameters for features
    :return: transformer
    """

    return ColumnTransformer(
        [("categorical_pipeline", build_categorical_pipeline(), params.categorical),
         ("numerical_pipeline", build_numerical_pipeline(), params.numerical)]
    )


def make_features(
        transformer: ColumnTransformer, df: pd.DataFrame
) -> pd.DataFrame:
    """Transform the features.

    :param transformer: data preprocessing transformer
    :param df: input features
    :return: transformed features
    """

    return pd.DataFrame(transformer.transform(df))


def get_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    """Extracts target from features.

    :param df: input features
    :param params: parameters for features
    :return: target
    """

    return df[params.target]


def serialize_transformer(transformer: ColumnTransformer, output: str) -> str:
    """Serialization of the transformer.

    :param transformer: transformer for serialization
    :param output: recording path transformer
    :return: recording path transformer
    """

    with open(output, "wb") as sf:
        pickle.dump(transformer, sf)
    return output


def deserialize_transformer(path: str) -> ColumnTransformer:
    """Deserialization of the transformer.

    :param path: path to transformer
    :return: transformer
    """

    with open(path, "rb") as sf:
        transformer = pickle.load(sf)
    return transformer
