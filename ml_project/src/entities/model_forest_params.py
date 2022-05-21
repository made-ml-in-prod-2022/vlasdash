from dataclasses import dataclass, field


@dataclass()
class ModelForestParams:
    random_state: int = field(default=42)
    n_estimators: int = field(default=50)
    max_depth: int = field(default=5)
