import pandas as pd
from dataclasses import dataclass
from typing import Literal

@dataclass
class ExplainationConfig:
    paper: Literal["biomining", "transynergy"]
    
    def __post_init__(self):
        if self.paper == "biomining":
            self.feature_names = [
            "ABL", "ABLb", "CSF1R", "CSF1Rb", "EGFR", "EGFRb", "FLT1", "FLT1b", "FLT4", "FLT4b",
            "KDR", "KDRb", "KIT", "KITb", "MCL1", "MCL1b", "NR1I2", "NR1I2b", "PDGFRB", "PDGFRBb",
            "RET", "RETb", "TOP2", "TOP2b", "TUB1", "TUB1b", "GATA3", "NF1", "NF2", "P53", "PI3K", "PTEN", "RAS"
        ]
            self.feature_length = 33
        elif self.paper == "transynergy":
            gene_df = pd.read_csv("external\drug_combination\data\genes\genes_2401_df.csv")
            feature_names_drugs = gene_df.columns.tolist() + ["pIC50"]
            feature_names_cell_lines = gene_df.columns.tolist() + ["padding_feature"]
            
            self.feature_names = feature_names_drugs + feature_names_drugs + feature_names_cell_lines
            self.feature_length = 2402 
            # this is very important to "fold" the model imput correctly before forward pass