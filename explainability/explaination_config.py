import pandas as pd
from dataclasses import dataclass
from typing import Literal

@dataclass
class ExplainationConfig:
    paper: Literal["biomining", "transynergy"]
    transynergy_gene_csv_path: str | None = "external\drug_combination\data\genes\genes_2401_df.csv"
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
            if self.transynergy_gene_csv_path is None:
                raise ValueError("Please provide path to gene csv for paper transynergy")
            gene_df = pd.read_csv(self.transynergy_gene_csv_path)
            gene_symbols = gene_df['symbol'].tolist()
            feature_names_drugs_a = [f"{g}_A" for g in gene_symbols] + ["pIC50_A"]
            feature_names_drugs_b = [f"{g}_B" for g in gene_symbols] + ["pIC50_B"]
            feature_names_cell_lines = gene_df['symbol'].tolist() + ["padding_feature"]
            
            self.feature_names = feature_names_drugs_a + feature_names_drugs_b + feature_names_cell_lines
            self.feature_length = 2402 
            # this is very important to "fold" the model imput correctly before forward pass
        else:
            raise ValueError(f"Unknown paper={self.paper}")
