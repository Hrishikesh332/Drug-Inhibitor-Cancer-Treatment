from dataclasses import dataclass
from typing import Literal, Optional

from explainability.explaination_config import ExplainationConfig

@dataclass
class ActivationMaximizationConfig(ExplainationConfig):
    maximize: bool = True
    num_trials: int = 5
    input_bounds: tuple[float, float] = (0, 1)
    steps: int = 2000
    lr: float = 0.01
    early_stopping: bool = True
    patience: int = 50
    regularization: Optional[Literal["l1", "l2"]] = None
    l1_lambda: float = 1e-3 # hyperparams for regularisation
    l2_lambda: float = 1e-3 # hyperparams for regularisation