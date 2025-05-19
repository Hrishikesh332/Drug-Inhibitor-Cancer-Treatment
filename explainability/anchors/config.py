from dataclasses import dataclass
from typing import Optional
from explainability.explaination_config import ExplainationConfig


@dataclass
class AnchorConfig(ExplainationConfig):
    maximize: bool = True
    num_explanations: Optional[int] = None
    threshold: float = 0.9
    seed: int = 42
    highest_quantile_for_binning: float = 0.85 # HYPERPARAM: Anchors do require categorical variables so we have to bin out target synergy score
    lowest_quantile_for_binning: float = 0.15 # HYPERPARAM: Anchors do require categorical variables so we have to bin out target synergy score