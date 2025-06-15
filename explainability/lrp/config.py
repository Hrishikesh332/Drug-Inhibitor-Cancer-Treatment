from typing import Literal, Optional

from dataclasses import dataclass
from explainability.explaination_config import ExplainationConfig

@dataclass
class LRPExplanationConfig(ExplainationConfig):
    subsample: bool = False
    num_samples: int = 1000




