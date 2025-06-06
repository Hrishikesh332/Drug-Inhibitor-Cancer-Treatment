from dataclasses import dataclass

from explainability.explaination_config import ExplainationConfig

@dataclass
class SHAPExplanationConfig(ExplainationConfig):
    samples_percentage: float = 0.25
    #how many samples to represent the baseline input distribution for SHAP comparisons
    min_background: int = 500
    max_background: int = 2000
    # how many input examples will receive SHAP explanations
    min_test: int = 500
    max_test: int = 2000