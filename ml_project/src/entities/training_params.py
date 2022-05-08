from dataclasses import dataclass, field
from typing import Optional

from .model_forest_params import ModelForestParams
from .model_sgd_params import ModelSGDParams


@dataclass()
class TrainingParams:
    model: str = field(default="SGDClassifier")
    model_forest_params: Optional[ModelForestParams] = None
    model_sgd_params: Optional[ModelSGDParams] = None
