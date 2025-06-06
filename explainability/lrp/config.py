from typing import Literal, Optional

from dataclasses import dataclass
from explainability.explaination_config import ExplainationConfig

@dataclass
class LRPExplanationConfig(ExplainationConfig):
    relative: bool = False
    baseline: Optional[Literal["zero", "mean", "random", "mean_per_cell_line"]] = None
    subsample: bool = True
    num_samples: int = 1000




