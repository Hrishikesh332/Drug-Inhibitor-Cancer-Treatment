import numpy as np
from typing import Literal
from pathlib import Path

def load_shap_data(
    paper: Literal["biomining", "transynergy"],
    base_dir_path: str = "./explainability/shapley/results",
):
    npz_path = Path(base_dir_path) / paper / "shap_complete.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"SHAP complete data file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    shap_values = data["shap_values"]
    inputs = data["inputs"]
    feature_names = data["feature_names"].tolist()
    test_indices = data['test_indices']
    return shap_values, inputs, feature_names, test_indices



    
