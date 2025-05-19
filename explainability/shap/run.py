import os
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Literal, Optional
import wandb
from logging import Logger

@dataclass
class SHAPExplanationConfig:
    paper: Literal["biomining", "transynergy"]
    max_background_samples: int = 200
    max_test_samples: int = 200

def run_shap_explanation(
    model: torch.nn.Module,
    paper: Literal["biomining", "transynergy"],
    X: torch.Tensor,
    Y: Optional[torch.Tensor],
    logger: Logger,
    config: Optional[SHAPExplanationConfig] = None,
):
    config = config or SHAPExplanationConfig(paper=paper)
    device = next(model.parameters()).device
    model.eval()
    model = model.to(device)

    wandb.init(
        project=f"EXPLAINABILITY on SHAP",
        config=asdict(config),
        name=f"{paper}_shap_explanation",
    )

    logger.info(f"Running SHAP (GradientExplainer) for model: {paper}")

    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    X = X.to(device)

    if paper == "transynergy":
        if X.ndim == 2:
            batch_size, total_features = X.shape
            if total_features % 3 != 0:
                raise ValueError(f"Expected transynergy input features divisible by 3, got {total_features}")
            feature_dim = total_features // 3
            logger.info(f"Reshaping transynergy input: ({batch_size}, {total_features}) → ({batch_size}, 3, {feature_dim})")
            X = X.view(batch_size, 3, feature_dim)

    background_size = min(config.max_background_samples, X.shape[0])
    test_size = min(config.max_test_samples, X.shape[0])
    background = X[:background_size]
    test_inputs = X[:test_size]

    explainer = shap.GradientExplainer(model, background)

    shap_values = []
    for i in tqdm(range(test_inputs.shape[0]), desc="SHAP explanation"):
        val = explainer.shap_values(test_inputs[i:i+1])
        shap_values.append(val)

    if paper == "transynergy":
        shap_values_matrix = np.array([sample.reshape(-1) for sample in shap_values])
    elif paper == "biomining":
        shap_values_matrix = np.vstack([sv[0] for sv in shap_values])

    input_numpy = test_inputs.detach().cpu().numpy()
    if input_numpy.ndim == 3:
        input_numpy = input_numpy.reshape(input_numpy.shape[0], -1)

    if shap_values_matrix.shape != input_numpy.shape:
        raise ValueError("Mismatch between SHAP values and input data shape.")

    output_dir = Path(f"explainability/shap/results/{paper}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "summary_plot.png"

    if paper == "biomining":
        biomining_feature_names = [
            "ABL", "ABLb", "CSF1R", "CSF1Rb", "EGFR", "EGFRb", "FLT1", "FLT1b", "FLT4", "FLT4b",
            "KDR", "KDRb", "KIT", "KITb", "MCL1", "MCL1b", "NR1I2", "NR1I2b", "PDGFRB", "PDGFRBb",
            "RET", "RETb", "TOP2", "TOP2b", "TUB1", "TUB1b", "GATA3", "NF1", "NF2", "P53", "PI3K", "PTEN", "RAS"
        ]
        shap.summary_plot(shap_values_matrix, input_numpy, feature_names=biomining_feature_names, show=False)

    elif paper == "transynergy":
        gene_csv_path = os.path.join("external", "drug_combination", "data", "genes", "genes_2401_df.csv")

        gene_df = pd.read_csv(gene_csv_path)
        gene_symbols = gene_df['symbol'].tolist()

        drug_feature_names = gene_symbols + ['pIC50']
        cellline_feature_names = gene_symbols

        feature_names = (
            [f"drugA_{name}" for name in drug_feature_names] +
            [f"drugB_{name}" for name in drug_feature_names] +
            [f"cellline_{name}" for name in cellline_feature_names]
        )

        shap.summary_plot(shap_values_matrix, input_numpy, feature_names=feature_names, show=False)

    plt.savefig(plot_path)
    logger.info(f"SHAP summary plot saved to {plot_path}")
    wandb.log({"shap_summary_plot": wandb.Image(str(plot_path))})

    wandb.finish()