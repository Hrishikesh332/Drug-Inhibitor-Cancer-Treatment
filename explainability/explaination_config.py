from dataclasses import dataclass
from typing import Literal

@dataclass
class ExplainationConfig:
    paper: Literal["biomining", "transynergy"]
    cell_drug_feat_len_transynergy : int = 2402
    cell_drug_feat_len_biomining: int = 33 