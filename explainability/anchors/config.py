from dataclasses import dataclass
from typing import Optional, Literal
from explainability.explaination_config import ExplainationConfig


@dataclass
class AnchorConfig(ExplainationConfig):
    num_explanations: Optional[int] = None
    threshold: float = 0.9
    fraction_explained: Literal["all", "bottom_10_percent", "top_10_percent", "random"] = "random"
    highest_quantile_for_binning: float = 0.85 # HYPERPARAM: Anchors do require categorical variables so we have to bin out target synergy score
    lowest_quantile_for_binning: float = 0.15 # HYPERPARAM: Anchors do require categorical variables so we have to bin out target synergy score