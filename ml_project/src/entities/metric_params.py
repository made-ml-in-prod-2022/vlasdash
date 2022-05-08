from dataclasses import dataclass, field


@dataclass()
class MetricParams:
    precision: bool = field(default=True)
    recall: bool = field(default=True)
    f1: bool = field(default=True)
