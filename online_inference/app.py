import os
import pickle
from typing import List, Optional
from sklearn.compose import ColumnTransformer
import uvicorn
import pandas as pd
from fastapi import FastAPI
from sklearn.pipeline import Pipeline
from fastapi.encoders import jsonable_encoder

from entities import HeartDiseaseRequest, ConditionResponse


model: Optional[Pipeline] = None
transformer: Optional[ColumnTransformer] = None
app = FastAPI()


def load_model(path: str) -> Pipeline:
    """Deserialize model.
        :param path: path to model
        :return: deserialized model
    """

    with open(path, "rb") as stream:
        return pickle.load(stream)


def load_transformer(path: str) -> ColumnTransformer:
    """Deserialize transformer.
        :param path: path to transformer
        :return: deserialized transformer
    """

    with open(path, "rb") as stream:
        return pickle.load(stream)


def make_predict(
        data: List[HeartDiseaseRequest],
        model: Pipeline,
        transformer: ColumnTransformer,
) -> List[ConditionResponse]:
    """Deserialize transformer.
        :param data: input data for predict
        :param model: model
        :param transformer: transformer
        :return: deserialized transformer
    """

    data = pd.DataFrame(jsonable_encoder(data))
    transform_data = transformer.transform(data.drop(["id"], axis=1))
    ids = list(data["id"])

    predicts = model.predict(transform_data)

    return [
        ConditionResponse(id=id_, condition=int(condition_))
        for id_, condition_ in zip(ids, predicts)
    ]


@app.get("/")
def main():
    """Start page."""

    return "This app predicts heart disease"


@app.on_event("startup")
def load_pipline():
    """Load transformer and model."""

    global model
    global transformer
    model_path = os.getenv("PATH_TO_MODEL")
    transformer_path = os.getenv("PATH_TO_TRANSFORMER")
    if model_path is None:
        raise NotImplementedError("Path to model is None")
    if transformer_path is None:
        raise NotImplementedError("Path to transformer is None")

    model = load_model(model_path)
    transformer = load_transformer(transformer_path)


@app.get(
    "/health",
    description="Check if a model is available",
)
def health() -> int:
    if model is None:
        return 503

    return 200


@app.get(
    "/predict/",
    response_description="Heart disease condition",
    description="Get prediction for features",
    response_model=List[ConditionResponse],
)
def predict(
        request: List[HeartDiseaseRequest]
) -> List[ConditionResponse]:
    return make_predict(
        request,
        model,
        transformer,
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
