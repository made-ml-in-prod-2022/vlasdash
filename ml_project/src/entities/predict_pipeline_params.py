from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig


@dataclass()
class PredictPipelineParams:
    input_data_path: str
    model_path: str
    transformer_path: str
    prediction_path: str


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_params_from_dict(cfg: DictConfig) -> PredictPipelineParams:
    """Reading parameters for model prediction pipeline.

        :param cfg: config
        :return: prediction pipeline parameters
    """

    return PredictPipelineParamsSchema().load(cfg)
