from dataclasses import dataclass, field


@dataclass()
class ModelSGDParams:
    random_state: int = field(default=42)
    penalty: str = field(default="l2")
    alpha: float = field(default=0.0001)
