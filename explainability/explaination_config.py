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
            "ABL", "ABLb", "CSF1R", "CSF1Rb", "EGFR", "EGFRb", "FLT1", "FLT1b", "FLT4", "FLT4b",
            "KDR", "KDRb", "KIT", "KITb", "MCL1", "MCL1b", "NR1I2", "NR1I2b", "PDGFRB", "PDGFRBb",
            "RET", "RETb", "TOP2", "TOP2b", "TUB1", "TUB1b", "GATA3", "NF1", "NF2", "P53", "PI3K", "PTEN", "RAS"
        ]
            self.feature_length = 33
        elif self.paper == "transynergy":
            if self.transynergy_features_parquet_path is None:
                raise ValueError("Please provide path to the features parquet file for paper transynergy")
            X_df = pd.read_csv(self.transynergy_features_parquet_path)
            self.feature_names = X_df.columns
            self.feature_length = X_df.shape[0]
        else:
            raise ValueError(f"Unknown paper={self.paper}")
