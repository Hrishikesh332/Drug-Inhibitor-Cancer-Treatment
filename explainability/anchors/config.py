from dataclasses import dataclass
from explaination_config.explaination_config import ExplainationConfig


@dataclass
class AnchorConfig(ExplainationConfig):
    maximize: bool = True
    num_explanations: int = 50
    threshold: float = 0.9
    seed: int = 42