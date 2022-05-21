import logging
import os
from typing import Dict, Tuple
import hydra
from omegaconf import DictConfig

from data.make_dataset import load_dataset, split_dataset
from entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_train_params_from_dict,
)
from features.build_features import (
    get_target,
    build_transformer,
    serialize_transformer,
    make_features,
)
from models.train_model import (
    train_model,
    serialize_model,
    evaluate_model,
    write_metrics,
)
from models.predict_model import predict_model

logger = logging.getLogger(__name__)


def train_pipeline(
        training_pipeline_params: TrainingPipelineParams
) -> Tuple[str, str, Dict[str, float]]:
    """Train model pipeline.

    :param training_pipeline_params: parameters for model training
    :return: the paths to the model, transformer and metrics on validation
    """

    logger.info("Start train pipeline with model "
                f"{training_pipeline_params.train_params.model}")
    data = load_dataset(training_pipeline_params.input_data_path)
    logger.info(f"data shape is {data.shape}")
    train_df, valid_df = split_dataset(
        data,
        training_pipeline_params.splitting_params
    )
    logger.info(f"train_df shape is {train_df.shape}")
    logger.info(f"valid_df shape is {valid_df.shape}")

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    train_features = make_features(
        transformer,
        train_df.drop(training_pipeline_params.feature_params.target, axis=1)
    )
    train_target = get_target(
        train_df,
        training_pipeline_params.feature_params
    )
    logger.info(f"train_features shape is {train_features.shape}")
    logger.info(f"train_target shape is {train_target.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    valid_features = make_features(
        transformer,
        valid_df.drop(training_pipeline_params.feature_params.target, axis=1)
    )
    valid_target = get_target(
        valid_df,
        training_pipeline_params.feature_params
    )
    logger.info(f"valid_features shape is {valid_features.shape}")
    logger.info(f"valid_target shape is {valid_target.shape}")

    predicts = predict_model(
        model,
        valid_features,
    )
    metrics = evaluate_model(
        predicts,
        valid_target,
        training_pipeline_params.metric_params,
    )
    logger.info(f"Metrics: {metrics}")
    path_to_metrics = write_metrics(
        metrics,
        training_pipeline_params.metric_path
    )
    logger.info(f"Metrics was recorded to {path_to_metrics}")

    path_to_model = serialize_model(
        model,
        training_pipeline_params.output_model_path
    )
    logger.info(f"Model was recorded to {path_to_model}")
    path_to_transformer = serialize_transformer(
        transformer,
        training_pipeline_params.transformer_path
    )
    logger.info(f"Transformer was recorded to {path_to_transformer}")

    logger.info("End train pipeline")

    return path_to_model, path_to_transformer, metrics


@hydra.main(config_path="../configs", config_name="train_forest_config")
def run_pipeline(cfg: DictConfig) -> None:
    params = read_train_params_from_dict(cfg)
    working_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..'
    ))
    os.chdir(working_dir)
    train_pipeline(params)


if __name__ == "__main__":
    run_pipeline()
