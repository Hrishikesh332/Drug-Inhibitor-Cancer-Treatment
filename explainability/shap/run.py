import os
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from dataclasses import asdict
from typing import Literal
import wandb
from logging import Logger
from explainability.utils import save_with_wandb
from explainability.shap.utils import select_representative_samples
from explainability.shap.config import SHAPExplanationConfig

def run_shap_explanation(
    model: torch.nn.Module,
    paper: Literal["biomining", "transynergy"],
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    logger: Logger,
    **kwargs
):
    config = SHAPExplanationConfig(paper=paper, **kwargs)
    device = next(model.parameters()).device
    model.eval()
    model = model.to(device)

    wandb.init(
        project=f"EXPLAINABILITY on SHAP",
        config=asdict(config),
        name=f"{paper}_shap_explanation",
    )

    logger.info(f"Running SHAP (GradientExplainer) for model: {paper}")

    if isinstance(X_train, np.ndarray):
        X_train = torch.tensor(X_train, dtype=torch.float32)
    if isinstance(X_test, np.ndarray):
        X_test = torch.tensor(X_test, dtype=torch.float32)

    X_train = X_train.to(device)
    X_test = X_test.to(device)

    if config.paper == "transynergy":
        if X_train.ndim == 2:
            batch_size_train, total_features = X_train.shape
            if total_features % 3 != 0:
                raise ValueError(f"Expected transynergy input features divisible by 3, got {total_features}")
            feature_dim = total_features // 3
            logger.info(f"Reshaping transynergy X_train: ({batch_size_train}, {total_features}) → ({batch_size_train}, 3, {feature_dim})")
            X_train = X_train.view(batch_size_train, 3, feature_dim)
        if X_test.ndim == 2:
            batch_size_test, total_features = X_test.shape
            if total_features % 3 != 0:
                raise ValueError(f"Expected transynergy input features divisible by 3, got {total_features}")
            feature_dim = total_features // 3
            logger.info(f"Reshaping transynergy X_test: ({batch_size_test}, {total_features}) → ({batch_size_test}, 3, {feature_dim})")
            X_test = X_test.view(batch_size_test, 3, feature_dim)

    # sample size is equal to percentage but at least the minimum and no more than the maximum configured limits
    background_size = min(max(config.min_background, int(X_train.shape[0] * config.samples_percentage)), config.max_background) 
    test_size = min(max(config.min_test, int(X_test.shape[0] * config.samples_percentage)), config.max_test)

    background = select_representative_samples(X_train, background_size)
    test_inputs = select_representative_samples(X_test, test_size)

    explainer = shap.GradientExplainer(model, background)

    shap_values = []
    for i in tqdm(range(test_inputs.shape[0]), desc="SHAP explanation"):
        val = explainer.shap_values(test_inputs[i:i+1])
        shap_values.append(val)

    if config.paper == "transynergy":
        shap_values_matrix = np.array([sample.reshape(-1) for sample in shap_values])
    elif config.paper == "biomining":
        shap_values_matrix = np.vstack([sv[0] for sv in shap_values])

    output_dir = Path(f"explainability/shap/results/{paper}")
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_values_tensor = torch.tensor(shap_values_matrix)
    shap_values_path = output_dir / "shap_values.pt"
    torch.save(shap_values_tensor, shap_values_path)

    logger.info(f"SHAP values saved locally at {shap_values_path}")
    save_with_wandb(str(shap_values_path), name_of_object=f"{paper}_shap_values.pt")

    wandb.finish()
