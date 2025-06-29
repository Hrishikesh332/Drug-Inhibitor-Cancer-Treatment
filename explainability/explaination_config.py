import pandas as pd
from dataclasses import dataclass
from typing import Literal

@dataclass
class ExplainationConfig:
    paper: Literal["biomining", "transynergy"]
    transynergy_features_parquet_path: str | None = "external/drug_combination/data/final_X.parquet"
    seed: int = 42

    def __post_init__(self):
        if self.paper == "biomining":
            self.feature_names = [
            "ABL1", "ABL1b", "CSF1R", "CSF1Rb", "EGFR", "EGFRb", "FLT1", "FLT1b", "FLT4", "FLT4b",
            "KDR", "KDRb", "KIT", "KITb", "MCL1", "MCL1b", "NR1I2", "NR1I2b", "PDGFRB", "PDGFRBb",
            "RET", "RETb", "TOP2A", "TOP2Ab", "TUBB1", "TUBB1b", "GATA3", "NF1", "NF2", "P53", "PI3K", "PTEN", "RAS"
        ]
            self.feature_length = 33
        elif self.paper == "transynergy":
            if self.transynergy_features_parquet_path is None:
                raise ValueError("Please provide path to the features parquet file for paper transynergy")
            X_df = pd.read_parquet(self.transynergy_features_parquet_path)
            self.feature_names = X_df.columns
            num_cols = X_df.shape[1]
            if num_cols % 3 != 0:
                raise ValueError("The number of columns for the transynergy data has to be divisible by 3 - "
                                 "in order to account for fisrt drug, second drug, and the cell lines!")
            self.feature_length = X_df.shape[1] // 3
        else:
            raise ValueError(f"Unknown paper={self.paper}")
