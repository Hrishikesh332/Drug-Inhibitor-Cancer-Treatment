from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import numpy as np

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


def load_am_results_by_cell_line(
    paper: Literal["biomining", "transynergy"],
    cell_line: str,
    regularization: str = "l2_input",
    minimax: str = "max",
    base_dir_path: str = "./explainability/am",
    transynergy_genes_csv_path: (
        str | None
    ) = "../external/drug_combination/data/genes/genes_2401_df.csv",
) -> np.ndarray:
    """
    Load AM results for a specific cell line.
    
    Parameters:
        paper (str): Either "biomining" or "transynergy".
        cell_line (str): Name of the cell line.
        regularization (str): Regularization method used.
        minimax (str): Whether to maximize or minimize.
        base_dir_path (str): Base directory path where results are stored.
        
    Returns:
        np.ndarray: Average AM attributions across trials for the cell line.
    """
    base_path = Path(base_dir_path)
    if not base_path.exists():
        raise ValueError(f"{base_path=} doesn't exist")
    
    feature_names = ExplainationConfig(
        paper=paper
    ).feature_names
    
    if cell_line is "prototypeAM":
        dir_path = base_path / "results"/ f"{paper}_{minimax}_reg_{regularization}"
    else:
        # Construct the directory path for this specific cell line
        dir_path = base_path / "results" / "by_cell_line" / f"{paper}_{cell_line}_{minimax}_reg_{regularization}"
    
    if not dir_path.exists():
        raise ValueError(f"Directory {dir_path} doesn't exist")
    
    # Load all trial results and average them
    feature_importance = torch.zeros(len(feature_names))
    trial_count = 0
    
    for pt_file in dir_path.glob("best_input_trial_*.pt"):
        tensor = torch.load(pt_file)
        feature_importance += tensor
        trial_count += 1
    
    if trial_count == 0:
        raise ValueError(f"No trial files found in {dir_path}")
    
    # Average across trials
    feature_importance = feature_importance / trial_count
    
    return feature_importance.detach().cpu().numpy()


def load_all_am_results_by_cell_line(
    paper: Literal["biomining", "transynergy"],
    regularization: str = "l2_input",
    minimax: str = "max",
    base_dir_path: str = "./explainability/am",
    transynergy_genes_csv_path: (
        str | None
    ) = "../external/drug_combination/data/genes/genes_2401_df.csv",
) -> dict:
    """
    Load AM results for all cell lines.
    
    Parameters:
        paper (str): Either "biomining" or "transynergy".
        regularization (str): Regularization method used.
        minimax (str): Whether to maximize or minimize.
        base_dir_path (str): Base directory path where results are stored.
        
    Returns:
        dict: Dictionary mapping cell line names to their AM attributions.
    """
    base_path = Path(base_dir_path)
    cell_line_dir = base_path / "results" / "by_cell_line"
    
    if not cell_line_dir.exists():
        raise ValueError(f"Cell line directory {cell_line_dir} doesn't exist")
    
    results = {}
    
    # Find all directories that match the pattern for this paper and settings
    pattern = f"{paper}_*_{minimax}_reg_{regularization}"
    
    for cell_line_dir_path in cell_line_dir.glob(pattern):
        # Extract cell line name from directory name
        dir_name = cell_line_dir_path.name
        cell_line = dir_name.replace(f"{paper}_", "").replace(f"_{minimax}_reg_{regularization}", "")
        
        try:
            am_attributions = load_am_results_by_cell_line(
                paper=paper,
                cell_line=cell_line,
                regularization=regularization,
                minimax=minimax,
                base_dir_path=base_dir_path,
                transynergy_genes_csv_path=transynergy_genes_csv_path
            )
            results[cell_line] = am_attributions
        except Exception as e:
            print(f"Warning: Could not load results for cell line {cell_line}: {e}")
    
    return results
