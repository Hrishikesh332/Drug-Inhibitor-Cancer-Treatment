from pathlib import Path
from typing import Literal

import pandas as pd
import torch

from explainability.explaination_config import ExplainationConfig

REGULARIZATIONS = ["l2", "l1", "none"]
MINIMAXIS = ["max", "min"]


def load_results(
    paper: Literal["biomining", "transynergy"],
    base_dir_path: str = "./explainability/am",
    transynergy_genes_csv_path: (
        str | None
    ) = "../external/drug_combination/data/genes/genes_2401_df.csv",
) -> pd.DataFrame:
    """
    Load and aggregate feature importance results for a given paper from disk.

    Parameters:
        paper (str): Either "biomining" or "transynergy".
        base_dir_path (str): Base directory path where results are stored.

    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'value', 'regularization', 'minimax'].
    """
    base_path = Path(base_dir_path)
    if not base_path.exists():
        raise ValueError(f"{base_path=} doesn't exist")
    feature_names = ExplainationConfig(
        paper=paper, transynergy_gene_csv_path=transynergy_genes_csv_path
    ).feature_names

    data = []

    for minimax in MINIMAXIS:
        for reg in REGULARIZATIONS:
            dir_path = base_path / "results" / f"{paper}_{minimax}_reg_{reg}"
            feature_importance = torch.zeros(len(feature_names))

            for pt_file in dir_path.glob("*.pt"):
                tensor = torch.load(pt_file)
                feature_importance += tensor

            values = feature_importance.tolist()

            for i, value in enumerate(values):
                data.append(
                    {
                        "feature": feature_names[i],
                        "value": value,
                        "regularization": reg,
                        "minimax": minimax,
                    }
                )

    return pd.DataFrame(data)
