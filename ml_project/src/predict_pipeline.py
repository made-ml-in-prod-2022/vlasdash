import logging
import os
import hydra
from omegaconf import DictConfig

from data.make_dataset import load_dataset
from entities.predict_pipeline_params import (
    PredictPipelineParams,
    read_predict_params_from_dict,
)
from features.build_features import make_features
from features.build_features import deserialize_transformer
from models.predict_model import (
    deserialize_model,
    write_prediction,
    predict_model,
)

logger = logging.getLogger(__name__)


def predict_pipeline(predict_pipeline_params: PredictPipelineParams) -> str:
    """Prediction model pipelining.

    :param predict_pipeline_params: parameters for model prediction
    :return: path to the prediction file
    """

    logger.info("Start predict pipeline with model "
                f"from {predict_pipeline_params.model_path}")
    test_df = load_dataset(predict_pipeline_params.input_data_path)
    logger.info(f"data shape is {test_df.shape}")

    model = deserialize_model(predict_pipeline_params.model_path)
    logger.info(f"Model is {model.__class__}")

    transformer = deserialize_transformer(
        predict_pipeline_params.transformer_path
    )
    transformed_data = make_features(transformer, test_df)

    prediction = predict_model(
        model,
        transformed_data
    )
    logger.info(f"Prediction shape is {prediction.shape}")
    path_to_prediction = write_prediction(
        prediction,
        predict_pipeline_params.prediction_path
    )
    logger.info(f"Prediction was recorded to {path_to_prediction}")

    logger.info("End predict pipeline")

    return path_to_prediction


@hydra.main(config_path="../configs", config_name="predict_config")
def run_pipeline(cfg: DictConfig) -> None:
    params = read_predict_params_from_dict(cfg)
    working_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..',
    ))
    os.chdir(working_dir)
    predict_pipeline(params)


if __name__ == "__main__":
    run_pipeline()
