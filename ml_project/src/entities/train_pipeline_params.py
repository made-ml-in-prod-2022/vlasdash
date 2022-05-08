from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig

from .splitting_params import SplittingParams
from .feature_params import FeatureParams
from .training_params import TrainingParams
from .metric_params import MetricParams


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    transformer_path: str
    splitting_params: SplittingParams
    metric_params: MetricParams
    feature_params: FeatureParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_train_params_from_dict(cfg: DictConfig) -> TrainingPipelineParams:
    return TrainingPipelineParamsSchema().load(cfg)
