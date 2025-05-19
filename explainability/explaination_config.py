from dataclasses import dataclass
from typing import Literal

@dataclass
class ExplainationConfig:
    paper: Literal["biomining", "transynergy"]